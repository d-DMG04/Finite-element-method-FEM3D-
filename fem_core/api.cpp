// =============================================================================
// api.cpp
// -----------------------------------------------------------------------------
// Реализация C-API. Ядро хранит единственное глобальное состояние (одна
// активная задача за раз) — это упрощает интерфейс и согласуется со
// сценарием использования из Python: один пользователь — одна задача.
//
// Раздел 3.2.2 пояснительной записки.
// =============================================================================

#ifndef FEM_CORE_BUILD
#  define FEM_CORE_BUILD
#endif
#include "api.hpp"

#include "assembly.hpp"
#include "flux.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "sparse.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace {

// Глобальный флаг отмены — устанавливается из другого потока через
// fem_request_cancel(); проверяется в callback CG.
std::atomic<bool> g_cancel_requested{false};

// Указатель на текущий progress-callback из Python (NULL — без прогресса).
using ProgressCallback = std::int32_t (*)(std::int32_t, double);
ProgressCallback g_progress_cb = nullptr;

// -----------------------------------------------------------------------------
// Глобальное состояние ядра.
// -----------------------------------------------------------------------------
struct CoreState {
    fem::Mesh                mesh;
    fem::ProblemSpec         spec;
    fem::CSRMatrix           K;
    std::vector<double>      F;
    std::vector<double>      T;
    fem::SolverResult        solver_info;
    bool                     mesh_ready = false;
    bool                     solved     = false;
};

CoreState* g_state = nullptr;

CoreState& ensure_state() {
    if (!g_state) g_state = new CoreState();
    return *g_state;
}

} // namespace

// -----------------------------------------------------------------------------
// 1) Генерация сетки на параллелепипеде.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_generate_box(
    double x_min, double x_max,
    double y_min, double y_max,
    double z_min, double z_max,
    std::int32_t nx, std::int32_t ny, std::int32_t nz) {
    try {
        auto& s = ensure_state();
        s.mesh.generate_box(x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz);
        s.mesh_ready = true;
        s.solved     = false;
        return 0;
    } catch (const std::exception&) {
        return -1;
    } catch (...) {
        return -1;
    }
}

// -----------------------------------------------------------------------------
// 2) Загрузка готовой сетки.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_load_mesh(
    const double* nodes_xyz, std::int32_t n_nodes,
    const std::int32_t* elements, std::int32_t n_elements,
    const std::int32_t* boundary_nodes,
    const std::int32_t* boundary_face_ids,
    std::int32_t n_boundary_faces) {
    if (!nodes_xyz || !elements) return -1;
    try {
        auto& s = ensure_state();
        s.mesh.load(nodes_xyz, n_nodes, elements, n_elements,
                    boundary_nodes, boundary_face_ids, n_boundary_faces);
        s.mesh_ready = true;
        s.solved     = false;
        return 0;
    } catch (...) {
        return -1;
    }
}

// -----------------------------------------------------------------------------
// 3) Число узлов.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_get_node_count(void) {
    if (!g_state || !g_state->mesh_ready) return 0;
    return g_state->mesh.n_nodes();
}

// -----------------------------------------------------------------------------
// 4) Координаты узлов.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_get_nodes(double* out_xyz) {
    if (!g_state || !g_state->mesh_ready || !out_xyz) return -1;
    const auto& nodes = g_state->mesh.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        out_xyz[3 * i + 0] = nodes[i].x;
        out_xyz[3 * i + 1] = nodes[i].y;
        out_xyz[3 * i + 2] = nodes[i].z;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// 5) Связность элементов.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_get_elements(std::int32_t* out_conn) {
    if (!g_state || !g_state->mesh_ready || !out_conn) return -1;
    const auto& els = g_state->mesh.elements();
    for (std::size_t e = 0; e < els.size(); ++e) {
        out_conn[4 * e + 0] = els[e].nodes[0];
        out_conn[4 * e + 1] = els[e].nodes[1];
        out_conn[4 * e + 2] = els[e].nodes[2];
        out_conn[4 * e + 3] = els[e].nodes[3];
    }
    return 0;
}

// -----------------------------------------------------------------------------
// 6) Материал и объёмные источники.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_set_material(double lambda_val, double Q) {
    if (lambda_val <= 0.0) return -1; // требование положительности
    auto& s = ensure_state();
    s.spec.lambda = lambda_val;
    s.spec.lambda_x = s.spec.lambda_y = s.spec.lambda_z = lambda_val;
    s.spec.is_anisotropic = false;
    s.spec.Q      = Q;
    s.solved      = false;
    return 0;
}

// -----------------------------------------------------------------------------
// Установка плотности и теплоёмкости (для нестационарной задачи).
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_set_thermal_capacity(double rho, double cp) {
    if (rho < 0.0 || cp < 0.0) return -1;
    auto& s = ensure_state();
    s.spec.rho = rho;
    s.spec.cp  = cp;
    return 0;
}

// -----------------------------------------------------------------------------
// 7) Граничное условие на одной грани.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_set_boundary_condition(
    std::int32_t face_id, std::int32_t bc_type,
    double T0, double q0, double alpha, double T_inf) {
    if (face_id < 0 || face_id >= fem::FACE_COUNT) return -1;
    // Принимаем все типы BC_NONE..BC_RADIATION (0..4). BC_RADIATION на этом
    // уровне трансформируется в BC_ROBIN на стороне Python (Picard).
    if (bc_type < 0 || bc_type > 4) return -1;
    auto& s = ensure_state();
    auto& bc = s.spec.bc[face_id];
    bc.type  = bc_type;
    bc.T0    = T0;
    bc.q0    = q0;
    bc.alpha = alpha;
    bc.T_inf = T_inf;
    s.solved = false;
    return 0;
}

// -----------------------------------------------------------------------------
// 8) Сборка и решение.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_solve(double tol, std::int32_t max_iter) {
    if (!g_state || !g_state->mesh_ready) return -1;
    if (tol <= 0.0 || max_iter <= 0) return -1;
    try {
        auto& s = *g_state;
        fem::build_sparsity_pattern(s.mesh, s.K);
        fem::assemble(s.mesh, s.spec, s.K, s.F);

        std::vector<std::int32_t> dnodes;
        std::vector<double>       dvals;
        fem::apply_dirichlet(s.mesh, s.spec, s.K, s.F, dnodes, dvals);

        // Сбрасываем cancel-флаг перед решением.
        g_cancel_requested.store(false);

        fem::SolverOptions opts;
        opts.tol_rel  = tol;
        opts.max_iter = max_iter;

        // Подключаем callback, если задан.
        if (g_progress_cb != nullptr) {
            opts.progress_callback = [](std::int32_t it, double res) -> bool {
                if (g_cancel_requested.load()) return false;
                if (g_progress_cb != nullptr) {
                    // Возвращаемое значение из Python: 1 = продолжать, 0 = отмена.
                    return g_progress_cb(it, res) != 0;
                }
                return true;
            };
            opts.progress_period = 5;
        } else {
            // Даже без Python-callback проверяем cancel.
            opts.progress_callback = [](std::int32_t, double) -> bool {
                return !g_cancel_requested.load();
            };
            opts.progress_period = 20;
        }

        const std::int32_t ok = fem::solve_cg(s.K, s.F, opts, s.T, s.solver_info);
        s.solved = true;
        if (s.solver_info.cancelled) {
            return 2;  // прервано
        }
        // Возвращаем 0 при сходимости, 1 при отказе сходимости (предупреждение).
        return (ok == 1) ? 0 : 1;
    } catch (...) {
        return -1;
    }
}

// -----------------------------------------------------------------------------
// Регистрация Python-callback прогресса. Передаётся указатель на функцию
// int32(*)(int32 iteration, double residual). Возврат 1 = продолжать, 0 = отмена.
// Чтобы снять — передать NULL.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_set_progress_callback(
        std::int32_t (*cb)(std::int32_t, double)) {
    g_progress_cb = cb;
    return 0;
}

// -----------------------------------------------------------------------------
// Запросить прерывание текущего расчёта. Можно вызывать из другого потока.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_request_cancel(void) {
    g_cancel_requested.store(true);
    return 0;
}

extern "C" std::int32_t fem_clear_cancel(void) {
    g_cancel_requested.store(false);
    return 0;
}

// -----------------------------------------------------------------------------
// 9) Узловой вектор температур.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_get_temperature(double* out_T) {
    if (!g_state || !g_state->solved || !out_T) return -1;
    const auto& T = g_state->T;
    std::memcpy(out_T, T.data(), T.size() * sizeof(double));
    return 0;
}

// -----------------------------------------------------------------------------
// 10) Тепловые потоки (узловые).
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_compute_fluxes(double* out_flux) {
    if (!g_state || !g_state->solved || !out_flux) return -1;
    try {
        auto& s = *g_state;
        std::vector<double> qe;
        fem::compute_element_fluxes(s.mesh, s.spec, s.T, qe);
        std::vector<double> qn;
        fem::average_fluxes_to_nodes(s.mesh, qe, qn);
        std::memcpy(out_flux, qn.data(), qn.size() * sizeof(double));
        return 0;
    } catch (...) {
        return -1;
    }
}

// -----------------------------------------------------------------------------
// 11) Диагностика решателя.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_get_solver_info(
    std::int32_t* out_iterations,
    double*       out_residual,
    double*       out_time_seconds,
    std::int32_t* out_converged) {
    if (!g_state) return -1;
    const auto& info = g_state->solver_info;
    if (out_iterations)   *out_iterations   = info.iterations;
    if (out_residual)     *out_residual     = info.final_residual;
    if (out_time_seconds) *out_time_seconds = info.solve_time_s;
    if (out_converged)    *out_converged    = info.converged;
    return 0;
}

// -----------------------------------------------------------------------------
// Нестационарный решатель: одиночный запуск с возвратом серии T(t) во время.
//
// Реализована неявная схема Эйлера 1-го порядка:
//     (M/Δt + K) T^{n+1} = (M/Δt) T^n + F
// где M — диагональная (lumped) масс-матрица, K — обычная матрица жёсткости.
// Дирихле применяется к (M/Δt + K) на каждом шаге.
//
// Параметры:
//   t_end     — финальное время, с
//   dt        — шаг по времени, с
//   T_init    — начальная температура (одинаковая во всех узлах), °C
//   n_save    — желаемое число сохранённых снимков (включая начальный и конечный)
//   out_times — буфер длины n_save для моментов времени
//   out_T     — буфер длины n_save * n_nodes для T(t_i) подряд
//   tol       — относительная точность CG на каждом шаге
//   max_iter  — максимум итераций CG на каждом шаге
//
// Возврат: 0 — успех, 1 — частичная сходимость, 2 — прерывание, -1 — ошибка.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_solve_transient(
    double t_end, double dt, double T_init,
    std::int32_t n_save,
    double* out_times,
    double* out_T,
    double tol, std::int32_t max_iter)
{
    if (!g_state || !g_state->mesh_ready) return -1;
    if (dt <= 0.0 || t_end <= 0.0 || n_save < 2 || tol <= 0.0 || max_iter <= 0)
        return -1;
    if (!out_times || !out_T) return -1;

    try {
        auto& s = *g_state;
        const std::int32_t N = s.mesh.n_nodes();

        // 1. Собираем K, F и диагональную M.
        fem::build_sparsity_pattern(s.mesh, s.K);
        fem::assemble(s.mesh, s.spec, s.K, s.F);

        std::vector<double> M;
        fem::assemble_lumped_mass(s.mesh, s.spec, M);
        if ((std::int32_t)M.size() != N) return -1;

        // 2. Готовим A = K + M/Δt (копия K с добавлением M на диагональ).
        fem::CSRMatrix A = s.K;
        const double inv_dt = 1.0 / dt;
        const auto& rp = A.row_ptr();
        const auto& ci = A.col_indices();
        auto& vals = A.values();
        for (std::int32_t i = 0; i < N; ++i) {
            const std::int32_t row_start = rp[static_cast<std::size_t>(i)];
            const std::int32_t row_end   = rp[static_cast<std::size_t>(i + 1)];
            for (std::int32_t k = row_start; k < row_end; ++k) {
                if (ci[static_cast<std::size_t>(k)] == i) {
                    vals[static_cast<std::size_t>(k)] += M[i] * inv_dt;
                    break;
                }
            }
        }
        // Применяем Дирихле к A.
        std::vector<std::int32_t> dnodes;
        std::vector<double>       dvals;
        std::vector<double> rhs0 = s.F;  // временно, чтобы apply_dirichlet модифицировал rhs0
        fem::apply_dirichlet(s.mesh, s.spec, A, rhs0, dnodes, dvals);
        // rhs0 теперь содержит F + Дирихле-вклад.

        // 3. Инициализация T.
        s.T.assign(static_cast<std::size_t>(N), T_init);
        // На Дирихле-узлах сразу ставим значения.
        for (std::size_t i = 0; i < dnodes.size(); ++i) {
            s.T[static_cast<std::size_t>(dnodes[i])] = dvals[i];
        }

        // 4. Сохраняем t=0 как первый snapshot.
        const std::int32_t total_steps = static_cast<std::int32_t>(
            std::ceil(t_end / dt));
        // Моменты сохранения: равномерно от 0 до total_steps включительно.
        std::vector<std::int32_t> save_at(static_cast<std::size_t>(n_save));
        for (std::int32_t k = 0; k < n_save; ++k) {
            save_at[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(
                std::round((double)k * total_steps / (n_save - 1)));
        }
        std::int32_t saved_idx = 0;
        auto save_snapshot = [&](std::int32_t step) {
            if (saved_idx < n_save
                && save_at[static_cast<std::size_t>(saved_idx)] == step) {
                out_times[saved_idx] = step * dt;
                std::memcpy(out_T + saved_idx * N, s.T.data(),
                             static_cast<std::size_t>(N) * sizeof(double));
                saved_idx++;
            }
        };
        save_snapshot(0);

        g_cancel_requested.store(false);
        std::vector<double> rhs(static_cast<std::size_t>(N));
        fem::SolverOptions opts;
        opts.tol_rel  = tol;
        opts.max_iter = max_iter;
        opts.progress_callback = [](std::int32_t, double) -> bool {
            return !g_cancel_requested.load();
        };
        opts.progress_period = 50;

        // 5. Цикл шагов по времени.
        for (std::int32_t step = 1; step <= total_steps; ++step) {
            // rhs = rhs0 + (M/Δt) * T^n
            // На узлах Дирихле rhs уже содержит правильное значение T_dir
            // благодаря apply_dirichlet — но для них нужно тоже учесть M·T/Δt,
            // что мы НЕ хотим: apply_dirichlet установил A[k,k]=1 и F[k]=T_dir,
            // поэтому решение CG автоматически даст T[k]=T_dir.
            // Для не-Дирихле узлов rhs[i] = rhs0[i] + M[i]/Δt * T_old[i].
            // Чтобы не путать, делаем единообразно:
            //   for i: rhs[i] = (Дирихле[i] ? T_dir : rhs0[i] + M[i]/Δt*T_old[i])
            // Но apply_dirichlet выставил A[k,k]=1 для Дирихле, так что
            // rhs0[i] для Дирихле уже = T_dir. Просто добавим M*T_old/dt
            // только для не-Дирихле строк.
            for (std::int32_t i = 0; i < N; ++i) {
                rhs[i] = rhs0[i] + M[i] * inv_dt * s.T[i];
            }
            // Перезапишем Дирихле-строки: для них rhs должен = T_dir.
            for (std::size_t i = 0; i < dnodes.size(); ++i) {
                rhs[static_cast<std::size_t>(dnodes[i])] = dvals[i];
            }

            // CG.
            const std::int32_t ok = fem::solve_cg(A, rhs, opts, s.T, s.solver_info);
            (void)ok;
            if (s.solver_info.cancelled) {
                return 2;
            }
            save_snapshot(step);
            if (saved_idx >= n_save) break;
        }
        s.solved = true;
        return 0;
    } catch (...) {
        return -1;
    }
}

// -----------------------------------------------------------------------------
// 12) Освобождение ресурсов.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_free(void) {
    delete g_state;
    g_state = nullptr;
    return 0;
}

// -----------------------------------------------------------------------------
// Доп.: число элементов и граничных граней.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_get_element_count(void) {
    if (!g_state || !g_state->mesh_ready) return 0;
    return g_state->mesh.n_elements();
}

extern "C" std::int32_t fem_get_boundary_face_count(void) {
    if (!g_state || !g_state->mesh_ready) return 0;
    return g_state->mesh.n_boundary_faces();
}

// =============================================================================
// Локальные источники тепла (раздел 3.3.11 ПЗ).
// =============================================================================

extern "C" std::int32_t fem_clear_sources(void) {
    auto& s = ensure_state();
    s.spec.point_sources.clear();
    s.spec.volume_sources.clear();
    s.solved = false;
    return 0;
}

extern "C" std::int32_t fem_add_point_source(std::int32_t node_idx, double power) {
    if (!g_state || !g_state->mesh_ready) return -1;
    if (node_idx < 0 || node_idx >= g_state->mesh.n_nodes()) return -1;
    auto& s = *g_state;
    s.spec.point_sources.push_back(fem::PointSource{node_idx, power});
    s.solved = false;
    return 0;
}

extern "C" std::int32_t fem_add_volume_source(std::int32_t shape,
                                              const double* params,
                                              double Q0) {
    if (!g_state) return -1;
    if (!params) return -1;
    if (shape != fem::VOLSRC_BOX && shape != fem::VOLSRC_SPHERE) return -1;
    auto& s = ensure_state();
    fem::VolumeSource vs;
    vs.shape = shape;
    for (int i = 0; i < 6; ++i) vs.params[i] = params[i];
    vs.Q0 = Q0;
    // Базовая валидация:
    if (shape == fem::VOLSRC_BOX) {
        if (vs.params[0] >= vs.params[3] ||
            vs.params[1] >= vs.params[4] ||
            vs.params[2] >= vs.params[5]) return -1;
    } else { // sphere
        if (vs.params[3] <= 0.0) return -1;
    }
    s.spec.volume_sources.push_back(vs);
    s.solved = false;
    return 0;
}

// -----------------------------------------------------------------------------
// Поузельные переопределения Дирихле (раздел 3.4.3 ПЗ).
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_set_node_dirichlet(std::int32_t node_idx,
                                                double value) {
    auto& s = ensure_state();
    if (!s.mesh_ready) return -1;
    if (node_idx < 0 || node_idx >= s.mesh.n_nodes()) return -1;
    // Если узел уже есть в списке — обновляем значение; иначе добавляем.
    for (std::size_t i = 0; i < s.spec.dirichlet_node_overrides.size(); ++i) {
        if (s.spec.dirichlet_node_overrides[i] == node_idx) {
            s.spec.dirichlet_value_overrides[i] = value;
            s.solved = false;
            return 0;
        }
    }
    s.spec.dirichlet_node_overrides.push_back(node_idx);
    s.spec.dirichlet_value_overrides.push_back(value);
    s.solved = false;
    return 0;
}

extern "C" std::int32_t fem_clear_node_dirichlet(void) {
    auto& s = ensure_state();
    s.spec.dirichlet_node_overrides.clear();
    s.spec.dirichlet_value_overrides.clear();
    s.solved = false;
    return 0;
}

// -----------------------------------------------------------------------------
// Регионы материалов.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_clear_materials(void) {
    auto& s = ensure_state();
    s.spec.materials.clear();
    if (s.mesh_ready) {
        s.mesh.clear_material_assignments();
    }
    s.solved = false;
    return 0;
}

extern "C" std::int32_t fem_add_material(double lambda_val, double Q_val) {
    auto& s = ensure_state();
    if (lambda_val <= 0.0) return -1;
    fem::MaterialProperties mp;
    mp.lambda = lambda_val;
    mp.Q = Q_val;
    s.spec.materials.push_back(mp);
    s.solved = false;
    return static_cast<std::int32_t>(s.spec.materials.size());
}

// Расширенная версия с плотностью и теплоёмкостью.
extern "C" std::int32_t fem_add_material_with_thermal(
    double lambda_val, double Q_val, double rho, double cp) {
    auto& s = ensure_state();
    if (lambda_val <= 0.0 || rho < 0.0 || cp < 0.0) return -1;
    fem::MaterialProperties mp;
    mp.lambda = lambda_val;
    mp.Q = Q_val;
    mp.rho = rho;
    mp.cp  = cp;
    s.spec.materials.push_back(mp);
    s.solved = false;
    return static_cast<std::int32_t>(s.spec.materials.size());
}

extern "C" std::int32_t fem_assign_material_in_box(
        std::int32_t material_id,
        double x_min, double x_max,
        double y_min, double y_max,
        double z_min, double z_max) {
    auto& s = ensure_state();
    if (!s.mesh_ready) return -1;
    if (material_id < 0
        || material_id > static_cast<std::int32_t>(s.spec.materials.size())) {
        return -1;
    }
    const std::int32_t count = s.mesh.assign_material_in_box(
        material_id, x_min, x_max, y_min, y_max, z_min, z_max);
    s.solved = false;
    return count;
}

extern "C" std::int32_t fem_assign_material_in_sphere(
        std::int32_t material_id,
        double cx, double cy, double cz, double radius) {
    auto& s = ensure_state();
    if (!s.mesh_ready) return -1;
    if (material_id < 0
        || material_id > static_cast<std::int32_t>(s.spec.materials.size())) {
        return -1;
    }
    if (radius <= 0.0) return -1;
    const std::int32_t count = s.mesh.assign_material_in_sphere(
        material_id, cx, cy, cz, radius);
    s.solved = false;
    return count;
}

extern "C" std::int32_t fem_clear_material_assignments(void) {
    auto& s = ensure_state();
    if (s.mesh_ready) {
        s.mesh.clear_material_assignments();
    }
    s.solved = false;
    return 0;
}

extern "C" std::int32_t fem_get_material_ids(std::int32_t* out_ids) {
    auto& s = ensure_state();
    if (!s.mesh_ready || !out_ids) return -1;
    s.mesh.copy_material_ids_to(out_ids);
    return 0;
}

extern "C" std::int32_t fem_get_material_count(void) {
    auto& s = ensure_state();
    return static_cast<std::int32_t>(s.spec.materials.size());
}

extern "C" std::int32_t fem_set_material_anisotropic(
        double lambda_x, double lambda_y, double lambda_z, double Q_val) {
    auto& s = ensure_state();
    if (lambda_x <= 0.0 || lambda_y <= 0.0 || lambda_z <= 0.0) return -1;
    s.spec.is_anisotropic = true;
    s.spec.lambda_x = lambda_x;
    s.spec.lambda_y = lambda_y;
    s.spec.lambda_z = lambda_z;
    // lambda для совместимости — среднее (на всякий случай).
    s.spec.lambda = (lambda_x + lambda_y + lambda_z) / 3.0;
    s.spec.Q = Q_val;
    s.solved = false;
    return 0;
}

extern "C" std::int32_t fem_add_material_anisotropic(
        double lambda_x, double lambda_y, double lambda_z, double Q_val) {
    auto& s = ensure_state();
    if (lambda_x <= 0.0 || lambda_y <= 0.0 || lambda_z <= 0.0) return -1;
    fem::MaterialProperties mp;
    mp.is_anisotropic = true;
    mp.lambda_x = lambda_x;
    mp.lambda_y = lambda_y;
    mp.lambda_z = lambda_z;
    mp.lambda = (lambda_x + lambda_y + lambda_z) / 3.0;
    mp.Q = Q_val;
    s.spec.materials.push_back(mp);
    s.solved = false;
    return static_cast<std::int32_t>(s.spec.materials.size());
}
