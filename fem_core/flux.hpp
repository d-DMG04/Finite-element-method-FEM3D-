// =============================================================================
// flux.hpp
// -----------------------------------------------------------------------------
// Вычисление тепловых потоков q = -lambda * grad(T) на каждом элементе
// по найденному узловому решению T. Дополнительно — усреднение в узлы
// для визуализации.
//
// Формулы 1.34, 1.35 пояснительной записки.
// Соответствует разделу 3.1.5.
// =============================================================================

#ifndef FEM_CORE_FLUX_HPP
#define FEM_CORE_FLUX_HPP

#include "mesh.hpp"

#include <cstdint>
#include <vector>

namespace fem {

// Forward declaration ProblemSpec — определена в assembly.hpp.
struct ProblemSpec;

// -----------------------------------------------------------------------------
// Поток на каждом элементе: вектор (qx, qy, qz) длины 3 * Ne.
// lambda берётся из spec в соответствии с material_id каждого тетраэдра.
// -----------------------------------------------------------------------------
void compute_element_fluxes(const Mesh& mesh, const ProblemSpec& spec,
                            const std::vector<double>& T,
                            std::vector<double>& flux_per_element);

// -----------------------------------------------------------------------------
// Усреднение поэлементных потоков в узлы (для отображения векторного поля).
// На выходе: вектор длины 3 * N_nodes.
// -----------------------------------------------------------------------------
void average_fluxes_to_nodes(const Mesh& mesh,
                             const std::vector<double>& flux_per_element,
                             std::vector<double>& flux_per_node);

} // namespace fem

#endif // FEM_CORE_FLUX_HPP
