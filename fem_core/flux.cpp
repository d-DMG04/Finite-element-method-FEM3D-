// =============================================================================
// flux.cpp
// -----------------------------------------------------------------------------
// Вычисление тепловых потоков по закону Фурье. Поскольку базисные функции
// P1 имеют постоянный градиент в пределах элемента, поток также постоянен
// на элементе:
//
//     q^e = -lambda * grad(T_h)
//         = -lambda * sum_i T_i * grad(phi_i)
//         = -(lambda / (6 V_e)) * sum_i T_i * (b_i, c_i, d_i)^T
//
// Раздел 3.1.5 пояснительной записки.
// =============================================================================

#include "flux.hpp"

#include <array>
#include <cstddef>

namespace fem {

void compute_element_fluxes(const Mesh& mesh, double lambda,
                            const std::vector<double>& T,
                            std::vector<double>& flux_per_element) {
    const std::int32_t Ne = mesh.n_elements();
    flux_per_element.assign(static_cast<std::size_t>(3 * Ne), 0.0);

    #pragma omp parallel for schedule(static)
    for (std::int32_t e = 0; e < Ne; ++e) {
        std::array<double, 4> bb, cc, dd;
        const double V6 = mesh.element_gradients(e, bb, cc, dd);
        if (V6 == 0.0) continue;

        const auto& tet = mesh.elements()[static_cast<std::size_t>(e)];
        double gx = 0.0, gy = 0.0, gz = 0.0;
        for (int i = 0; i < 4; ++i) {
            const double Ti = T[static_cast<std::size_t>(tet.nodes[i])];
            gx += Ti * bb[i];
            gy += Ti * cc[i];
            gz += Ti * dd[i];
        }
        const double inv6V = 1.0 / V6;
        flux_per_element[static_cast<std::size_t>(3 * e + 0)] = -lambda * gx * inv6V;
        flux_per_element[static_cast<std::size_t>(3 * e + 1)] = -lambda * gy * inv6V;
        flux_per_element[static_cast<std::size_t>(3 * e + 2)] = -lambda * gz * inv6V;
    }
}

// -----------------------------------------------------------------------------
// Узловой поток — взвешенное среднее по примыкающим элементам.
// Веса — объёмы элементов (для гладкого, физически согласованного поля).
// -----------------------------------------------------------------------------
void average_fluxes_to_nodes(const Mesh& mesh,
                             const std::vector<double>& flux_per_element,
                             std::vector<double>& flux_per_node) {
    const std::int32_t N  = mesh.n_nodes();
    const std::int32_t Ne = mesh.n_elements();
    flux_per_node.assign(static_cast<std::size_t>(3 * N), 0.0);

    std::vector<double> weight(static_cast<std::size_t>(N), 0.0);

    for (std::int32_t e = 0; e < Ne; ++e) {
        const double Ve = mesh.element_volume(e);
        if (Ve <= 0.0) continue;
        const auto& tet = mesh.elements()[static_cast<std::size_t>(e)];
        const double qx = flux_per_element[static_cast<std::size_t>(3 * e + 0)];
        const double qy = flux_per_element[static_cast<std::size_t>(3 * e + 1)];
        const double qz = flux_per_element[static_cast<std::size_t>(3 * e + 2)];
        for (int i = 0; i < 4; ++i) {
            const std::int32_t gi = tet.nodes[i];
            flux_per_node[static_cast<std::size_t>(3 * gi + 0)] += qx * Ve;
            flux_per_node[static_cast<std::size_t>(3 * gi + 1)] += qy * Ve;
            flux_per_node[static_cast<std::size_t>(3 * gi + 2)] += qz * Ve;
            weight[static_cast<std::size_t>(gi)] += Ve;
        }
    }

    for (std::int32_t i = 0; i < N; ++i) {
        const double w = weight[static_cast<std::size_t>(i)];
        if (w > 0.0) {
            const double inv = 1.0 / w;
            flux_per_node[static_cast<std::size_t>(3 * i + 0)] *= inv;
            flux_per_node[static_cast<std::size_t>(3 * i + 1)] *= inv;
            flux_per_node[static_cast<std::size_t>(3 * i + 2)] *= inv;
        }
    }
}

} // namespace fem
