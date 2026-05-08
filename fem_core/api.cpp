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

#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace {

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
    s.spec.Q      = Q;
    s.solved      = false;
    return 0;
}

// -----------------------------------------------------------------------------
// 7) Граничное условие на одной грани.
// -----------------------------------------------------------------------------
extern "C" std::int32_t fem_set_boundary_condition(
    std::int32_t face_id, std::int32_t bc_type,
    double T0, double q0, double alpha, double T_inf) {
    if (face_id < 0 || face_id >= fem::FACE_COUNT) return -1;
    if (bc_type < 0 || bc_type > 3) return -1;
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

        fem::SolverOptions opts;
        opts.tol_rel  = tol;
        opts.max_iter = max_iter;

        const std::int32_t ok = fem::solve_cg(s.K, s.F, opts, s.T, s.solver_info);
        s.solved = true;
        // Возвращаем 0 при сходимости, 1 при отказе сходимости (предупреждение).
        return (ok == 1) ? 0 : 1;
    } catch (...) {
        return -1;
    }
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
        fem::compute_element_fluxes(s.mesh, s.spec.lambda, s.T, qe);
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
