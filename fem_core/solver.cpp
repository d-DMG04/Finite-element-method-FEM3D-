// =============================================================================
// solver.cpp
// -----------------------------------------------------------------------------
// Реализация PCG (Preconditioned Conjugate Gradient) с диагональным
// предобусловливателем Якоби.
//
// Алгоритм для системы K T = F, K симметричная положительно определённая.
//
//   T_0 произвольное (берём 0)
//   r_0 = F - K T_0
//   z_0 = M^{-1} r_0,        M = diag(K) (Якоби)
//   p_0 = z_0
//   for k = 0, 1, 2, ...
//       alpha_k = (r_k, z_k) / (p_k, K p_k)
//       T_{k+1} = T_k + alpha_k * p_k
//       r_{k+1} = r_k - alpha_k * K p_k
//       если |r_{k+1}| / |F| < tol — стоп
//       z_{k+1} = M^{-1} r_{k+1}
//       beta_k  = (r_{k+1}, z_{k+1}) / (r_k, z_k)
//       p_{k+1} = z_{k+1} + beta_k * p_k
//
// Все скалярные произведения и AXPY-операции — параллельные через OpenMP.
// SpMV — горячий цикл, параллелен в CSRMatrix::multiply.
//
// Раздел 3.1.4 пояснительной записки.
// =============================================================================

#include "solver.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace fem {

namespace {

// -----------------------------------------------------------------------------
// Скалярное произведение a^T b с OpenMP-редукцией.
// -----------------------------------------------------------------------------
inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    const std::int32_t n = static_cast<std::int32_t>(a.size());
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (std::int32_t i = 0; i < n; ++i) {
        s += a[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
    }
    return s;
}

// -----------------------------------------------------------------------------
// y <- y + alpha * x
// -----------------------------------------------------------------------------
inline void axpy(double alpha,
                 const std::vector<double>& x,
                 std::vector<double>& y) {
    const std::int32_t n = static_cast<std::int32_t>(x.size());
    #pragma omp parallel for schedule(static)
    for (std::int32_t i = 0; i < n; ++i) {
        y[static_cast<std::size_t>(i)] += alpha * x[static_cast<std::size_t>(i)];
    }
}

// -----------------------------------------------------------------------------
// p <- z + beta * p
// -----------------------------------------------------------------------------
inline void xpby(const std::vector<double>& z, double beta,
                 std::vector<double>& p) {
    const std::int32_t n = static_cast<std::int32_t>(z.size());
    #pragma omp parallel for schedule(static)
    for (std::int32_t i = 0; i < n; ++i) {
        p[static_cast<std::size_t>(i)] =
            z[static_cast<std::size_t>(i)] + beta * p[static_cast<std::size_t>(i)];
    }
}

// -----------------------------------------------------------------------------
// z <- M^{-1} r,  M = diag(K).  При нулевой диагонали ставим 1 (защита).
// -----------------------------------------------------------------------------
inline void apply_jacobi(const std::vector<double>& diag,
                         const std::vector<double>& r,
                         std::vector<double>& z) {
    const std::int32_t n = static_cast<std::int32_t>(r.size());
    #pragma omp parallel for schedule(static)
    for (std::int32_t i = 0; i < n; ++i) {
        const double d = diag[static_cast<std::size_t>(i)];
        z[static_cast<std::size_t>(i)] =
            (d != 0.0) ? r[static_cast<std::size_t>(i)] / d
                       : r[static_cast<std::size_t>(i)];
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Запуск PCG.
// -----------------------------------------------------------------------------
std::int32_t solve_cg(const CSRMatrix& K,
                      const std::vector<double>& F,
                      const SolverOptions& opts,
                      std::vector<double>& T,
                      SolverResult& result) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const std::int32_t N = K.size();
    T.assign(static_cast<std::size_t>(N), 0.0);

    // Норма правой части — для относительного критерия остановки.
    const double normF = std::sqrt(dot(F, F));
    if (normF == 0.0) {
        // Тривиальный случай F == 0 -> T == 0 уже решение.
        result.iterations    = 0;
        result.final_residual = 0.0;
        result.converged     = 1;
        result.cancelled     = 0;
        result.solve_time_s  = std::chrono::duration<double>(clock::now() - t0).count();
        return 1;
    }

    std::vector<double> diag;
    K.diagonal(diag);

    std::vector<double> r(static_cast<std::size_t>(N), 0.0);
    std::vector<double> z(static_cast<std::size_t>(N), 0.0);
    std::vector<double> p(static_cast<std::size_t>(N), 0.0);
    std::vector<double> Kp(static_cast<std::size_t>(N), 0.0);

    // r0 = F - K * T0,  T0 = 0  =>  r0 = F.
    r = F;

    apply_jacobi(diag, r, z);
    p = z;

    double rz_old = dot(r, z);
    double res_norm = std::sqrt(dot(r, r));
    double rel = res_norm / normF;

    std::int32_t k = 0;
    std::int32_t converged = (rel < opts.tol_rel) ? 1 : 0;
    std::int32_t cancelled = 0;

    for (k = 0; k < opts.max_iter && !converged; ++k) {
        K.multiply(p, Kp);
        const double pKp = dot(p, Kp);
        if (pKp <= 0.0) {
            // Признак потери положительной определённости — выходим аккуратно.
            break;
        }
        const double alpha = rz_old / pKp;
        axpy( alpha, p,  T);
        axpy(-alpha, Kp, r);

        res_norm = std::sqrt(dot(r, r));
        rel = res_norm / normF;
        if (rel < opts.tol_rel) {
            converged = 1;
            break;
        }

        // Callback прогресса: вызов через GIL может быть дорогим, поэтому
        // отбиваем каждые progress_period итераций.
        if (opts.progress_callback && opts.progress_period > 0
            && ((k + 1) % opts.progress_period == 0)) {
            if (!opts.progress_callback(k + 1, rel)) {
                cancelled = 1;
                break;
            }
        }

        apply_jacobi(diag, r, z);
        const double rz_new = dot(r, z);
        const double beta = rz_new / rz_old;
        xpby(z, beta, p);
        rz_old = rz_new;
    }

    // Если вышли по max_iter — converged остаётся 0, но T содержит последнее
    // приближение. Это согласуется с поведением, описанным в разделе 3.1.4.
    if (converged && k == 0 && rel < opts.tol_rel) {
        result.iterations = 0;
    } else if (converged) {
        result.iterations = k + 1;
    } else {
        result.iterations = k;  // прервано или max_iter
    }
    result.final_residual = rel;
    result.converged      = converged;
    result.cancelled      = cancelled;
    result.solve_time_s   = std::chrono::duration<double>(clock::now() - t0).count();
    return converged;
}

} // namespace fem
