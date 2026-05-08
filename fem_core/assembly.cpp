// =============================================================================
// assembly.cpp
// -----------------------------------------------------------------------------
// Двухпроходная сборка глобальной матрицы жёсткости K и вектора F.
// Параллелизация цикла сборки через OpenMP.
//
// Раздел 3.1.3 пояснительной записки.
// =============================================================================

#include "assembly.hpp"

#include <algorithm>
#include <cmath>
#include <set>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace fem {

// -----------------------------------------------------------------------------
// Первый проход: для каждой строки матрицы — отсортированный список столбцов.
// Узел i связан с узлом j, если они принадлежат хотя бы одному общему элементу.
// -----------------------------------------------------------------------------
void build_sparsity_pattern(const Mesh& mesh, CSRMatrix& K) {
    const std::int32_t N = mesh.n_nodes();
    std::vector<std::set<std::int32_t>> rows(static_cast<std::size_t>(N));

    for (std::int32_t e = 0; e < mesh.n_elements(); ++e) {
        const auto& tet = mesh.elements()[static_cast<std::size_t>(e)];
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                rows[static_cast<std::size_t>(tet.nodes[a])].insert(tet.nodes[b]);
            }
        }
    }

    // Граничные грани (Робен) — добавляют связи между тремя узлами грани.
    for (std::int32_t f = 0; f < mesh.n_boundary_faces(); ++f) {
        const auto& bf = mesh.boundary_faces()[static_cast<std::size_t>(f)];
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                rows[static_cast<std::size_t>(bf.nodes[a])].insert(bf.nodes[b]);
            }
        }
    }

    std::vector<std::vector<std::int32_t>> adj(static_cast<std::size_t>(N));
    for (std::int32_t i = 0; i < N; ++i) {
        adj[static_cast<std::size_t>(i)].assign(
            rows[static_cast<std::size_t>(i)].begin(),
            rows[static_cast<std::size_t>(i)].end());
        // set уже даёт сортированный порядок.
    }

    K.resize(N);
    K.set_pattern(adj);
}

// -----------------------------------------------------------------------------
// Второй проход: числовая сборка матрицы и правой части.
// -----------------------------------------------------------------------------
void assemble(const Mesh& mesh, const ProblemSpec& spec,
              CSRMatrix& K, std::vector<double>& F) {
    const std::int32_t N  = mesh.n_nodes();
    const std::int32_t Ne = mesh.n_elements();
    const std::int32_t Nf = mesh.n_boundary_faces();

    // Перед сборкой обнуляем values; шаблон уже зафиксирован.
    std::fill(K.values().begin(), K.values().end(), 0.0);
    F.assign(static_cast<std::size_t>(N), 0.0);

    const double lambda = spec.lambda;
    const double Q      = spec.Q;

    // -------------------------------------------------------------------------
    // Объёмные вклады: цикл по элементам.
    // Формула локальной K^e (1.21):
    //     K^e_{ij} = lambda / (36 * V_e) * (b_i b_j + c_i c_j + d_i d_j)
    // Локальный вектор от Q (1.23):  F^e_i = Q V_e / 4
    // -------------------------------------------------------------------------
    #pragma omp parallel for schedule(static)
    for (std::int32_t e = 0; e < Ne; ++e) {
        std::array<double, 4> bb, cc, dd;
        const double V6 = mesh.element_gradients(e, bb, cc, dd);
        const double Ve = V6 / 6.0;
        if (Ve <= 0.0) continue;

        // Локальная матрица 4x4.
        double Ke[4][4];
        const double coef = lambda / (36.0 * Ve);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Ke[i][j] = coef * (bb[i] * bb[j] + cc[i] * cc[j] + dd[i] * dd[j]);
            }
        }

        // Локальный вектор от объёмного источника.
        const double Fe_loc = (Q * Ve) / 4.0;

        const auto& tet = mesh.elements()[static_cast<std::size_t>(e)];
        for (int i = 0; i < 4; ++i) {
            const std::int32_t gi = tet.nodes[i];
            #pragma omp atomic
            F[static_cast<std::size_t>(gi)] += Fe_loc;
            for (int j = 0; j < 4; ++j) {
                const std::int32_t gj = tet.nodes[j];
                K.add_atomic(gi, gj, Ke[i][j]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Граничные вклады: цикл по граничным граням.
    // Формулы 1.25 (Нейман) и 1.27, 1.28 (Робен).
    // -------------------------------------------------------------------------
    #pragma omp parallel for schedule(static)
    for (std::int32_t f = 0; f < Nf; ++f) {
        const auto& bf = mesh.boundary_faces()[static_cast<std::size_t>(f)];
        const auto& bc = spec.bc[bf.face_id];

        // Дирихле обрабатывается после сборки; здесь — только Нейман и Робен.
        if (bc.type != BC_NEUMANN && bc.type != BC_ROBIN) continue;

        const double A = mesh.face_area(f);
        if (A <= 0.0) continue;

        if (bc.type == BC_NEUMANN) {
            // F^e_N = -q0 * A / 3 на каждом из трёх узлов грани.
            const double contrib = -bc.q0 * A / 3.0;
            for (int i = 0; i < 3; ++i) {
                const std::int32_t gi = bf.nodes[i];
                #pragma omp atomic
                F[static_cast<std::size_t>(gi)] += contrib;
            }
        } else { // BC_ROBIN
            // K^eR = (alpha A / 12) * [[2,1,1],[1,2,1],[1,1,2]]
            // F^eR_i = alpha * T_inf * A / 3
            const double Kcoef = bc.alpha * A / 12.0;
            const double Fcoef = bc.alpha * bc.T_inf * A / 3.0;
            const int M[3][3] = {{2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
            for (int i = 0; i < 3; ++i) {
                const std::int32_t gi = bf.nodes[i];
                #pragma omp atomic
                F[static_cast<std::size_t>(gi)] += Fcoef;
                for (int j = 0; j < 3; ++j) {
                    const std::int32_t gj = bf.nodes[j];
                    K.add_atomic(gi, gj, Kcoef * static_cast<double>(M[i][j]));
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Локальные источники тепла (раздел 3.3.11 ПЗ).
    //
    // Точечный: F[i] += P  для глобального индекса узла.
    // Объёмный (под)область: F[gi] += Q0 * V_e / 4  (как формула 1.23, но
    //   только для тех элементов, центры которых попадают в подобласть).
    // -------------------------------------------------------------------------
    for (const auto& ps : spec.point_sources) {
        if (ps.node_idx < 0 || ps.node_idx >= N) continue;  // защита
        F[static_cast<std::size_t>(ps.node_idx)] += ps.power;
    }

    if (!spec.volume_sources.empty()) {
        #pragma omp parallel for schedule(static)
        for (std::int32_t e = 0; e < Ne; ++e) {
            // Координаты центра тяжести элемента.
            const auto& tet = mesh.elements()[static_cast<std::size_t>(e)];
            const auto& nodes = mesh.nodes();
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int i = 0; i < 4; ++i) {
                const auto& n = nodes[static_cast<std::size_t>(tet.nodes[i])];
                cx += n.x; cy += n.y; cz += n.z;
            }
            cx *= 0.25; cy *= 0.25; cz *= 0.25;

            // Накопим суммарный Q_loc от всех подобластей, в которые попал центр.
            double Q_loc = 0.0;
            for (const auto& vs : spec.volume_sources) {
                bool inside = false;
                if (vs.shape == VOLSRC_BOX) {
                    inside = (cx >= vs.params[0] && cx <= vs.params[3] &&
                              cy >= vs.params[1] && cy <= vs.params[4] &&
                              cz >= vs.params[2] && cz <= vs.params[5]);
                } else if (vs.shape == VOLSRC_SPHERE) {
                    const double dx = cx - vs.params[0];
                    const double dy = cy - vs.params[1];
                    const double dz = cz - vs.params[2];
                    const double r  = vs.params[3];
                    inside = (dx*dx + dy*dy + dz*dz <= r * r);
                }
                if (inside) Q_loc += vs.Q0;
            }
            if (Q_loc == 0.0) continue;

            // Вклад в правую часть: Q_loc * V_e / 4 на каждый из 4 узлов.
            const double Ve = mesh.element_volume(e);
            if (Ve <= 0.0) continue;
            const double contrib = Q_loc * Ve / 4.0;
            for (int i = 0; i < 4; ++i) {
                const std::int32_t gi = tet.nodes[i];
                #pragma omp atomic
                F[static_cast<std::size_t>(gi)] += contrib;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Применение условий Дирихле к собранной системе.
//
// Стратегия (раздел 1.2.8):
//   1) собираем множество узлов Дирихле и их значения T0;
//   2) переносим вклад столбца k в правую часть: F_i -= K_{ik} * T0  (i != k);
//   3) обнуляем строку и столбец k, ставим 1 на K_{k,k}, F_k = T0.
//
// Обнуление столбца сохраняет симметрию матрицы.
// -----------------------------------------------------------------------------
void apply_dirichlet(const Mesh& mesh, const ProblemSpec& spec,
                     CSRMatrix& K, std::vector<double>& F,
                     std::vector<std::int32_t>& dirichlet_nodes,
                     std::vector<double>& dirichlet_values) {
    const std::int32_t N = mesh.n_nodes();

    // Узел может попасть на несколько граней Дирихле — берём последнее заданное
    // значение; для конфликтующих T0 это даст детерминированный результат.
    std::vector<std::int32_t> mark(static_cast<std::size_t>(N), 0);
    std::vector<double> T0(static_cast<std::size_t>(N), 0.0);

    for (std::int32_t f = 0; f < mesh.n_boundary_faces(); ++f) {
        const auto& bf = mesh.boundary_faces()[static_cast<std::size_t>(f)];
        const auto& bc = spec.bc[bf.face_id];
        if (bc.type != BC_DIRICHLET) continue;
        for (int i = 0; i < 3; ++i) {
            const std::int32_t gi = bf.nodes[i];
            mark[static_cast<std::size_t>(gi)] = 1;
            T0[static_cast<std::size_t>(gi)]   = bc.T0;
        }
    }

    // Поузельные переопределения Дирихле: помечаем узел и записываем значение
    // (даже если ранее он не был помечен через грани).
    for (std::size_t k = 0; k < spec.dirichlet_node_overrides.size(); ++k) {
        const std::int32_t gi = spec.dirichlet_node_overrides[k];
        if (gi < 0 || gi >= N) continue;
        mark[static_cast<std::size_t>(gi)] = 1;
        T0[static_cast<std::size_t>(gi)] = spec.dirichlet_value_overrides[k];
    }

    dirichlet_nodes.clear();
    dirichlet_values.clear();
    for (std::int32_t i = 0; i < N; ++i) {
        if (mark[static_cast<std::size_t>(i)]) {
            dirichlet_nodes.push_back(i);
            dirichlet_values.push_back(T0[static_cast<std::size_t>(i)]);
        }
    }

    // 1) Перенос столбцов в правую часть.
    //    Для каждого узла Дирихле k проходим по строкам, в которых есть K_{i,k},
    //    и вычитаем K_{i,k} * T0_k из F_i (для i != k).
    //    Эффективнее всего — пройти по всем ненулевым (i, j) один раз и для
    //    каждой j, помеченной как Дирихле, перекинуть вклад.
    const auto& row_ptr = K.row_ptr();
    const auto& cols    = K.col_indices();
    const auto& vals    = K.values();
    for (std::int32_t i = 0; i < N; ++i) {
        if (mark[static_cast<std::size_t>(i)]) continue; // строки Дирихле обнулим ниже
        const std::int32_t beg = row_ptr[static_cast<std::size_t>(i)];
        const std::int32_t end = row_ptr[static_cast<std::size_t>(i) + 1];
        double acc = 0.0;
        for (std::int32_t p = beg; p < end; ++p) {
            const std::int32_t j = cols[static_cast<std::size_t>(p)];
            if (mark[static_cast<std::size_t>(j)]) {
                acc += vals[static_cast<std::size_t>(p)] *
                       T0[static_cast<std::size_t>(j)];
            }
        }
        F[static_cast<std::size_t>(i)] -= acc;
    }

    // 2) Обнуление строк/столбцов и установка диагонали.
    for (std::int32_t k : dirichlet_nodes) {
        K.zero_row_and_column(k);
        F[static_cast<std::size_t>(k)] = T0[static_cast<std::size_t>(k)];
    }
}

} // namespace fem
