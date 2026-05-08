// =============================================================================
// mesh.hpp
// -----------------------------------------------------------------------------
// Структуры данных тетраэдральной сетки и алгоритм её построения для
// параллелепипедной области.
//
// Соответствует подсистеме «Построение сетки» (раздел 2.4.4) и формулам
// раздела 2.3.2 (тетраэдр P1).
// =============================================================================

#ifndef FEM_CORE_MESH_HPP
#define FEM_CORE_MESH_HPP

#include <array>
#include <cstdint>
#include <vector>

namespace fem {

// -----------------------------------------------------------------------------
// Идентификаторы граней внешнего параллелепипеда.
// Используются для привязки граничных условий к граням тела.
// -----------------------------------------------------------------------------
enum BoundaryFace : std::int32_t {
    FACE_X_MINUS = 0, // x = x_min
    FACE_X_PLUS  = 1, // x = x_max
    FACE_Y_MINUS = 2, // y = y_min
    FACE_Y_PLUS  = 3, // y = y_max
    FACE_Z_MINUS = 4, // z = z_min
    FACE_Z_PLUS  = 5, // z = z_max
    FACE_COUNT   = 6
};

// -----------------------------------------------------------------------------
// Узел сетки — точка в 3D-пространстве.
// -----------------------------------------------------------------------------
struct Node {
    double x;
    double y;
    double z;
};

// -----------------------------------------------------------------------------
// Тетраэдральный элемент P1 — четыре глобальных номера узлов.
// -----------------------------------------------------------------------------
struct Tetrahedron {
    std::array<std::int32_t, 4> nodes; // глобальные индексы вершин
};

// -----------------------------------------------------------------------------
// Граничная треугольная грань — три глобальных номера узлов и
// идентификатор группы (одна из FACE_*).
// -----------------------------------------------------------------------------
struct BoundaryFaceTri {
    std::array<std::int32_t, 3> nodes; // глобальные индексы вершин грани
    std::int32_t face_id;              // BoundaryFace
};

// -----------------------------------------------------------------------------
// Сетка целиком: массивы узлов, элементов и граничных граней.
// -----------------------------------------------------------------------------
class Mesh {
public:
    // Генерация структурированной тетраэдральной сетки на параллелепипеде
    // [x_min, x_max] x [y_min, y_max] x [z_min, z_max] с числом разбиений
    // (nx, ny, nz). Каждая прямоугольная ячейка разбивается на 6 тетраэдров.
    void generate_box(double x_min, double x_max,
                      double y_min, double y_max,
                      double z_min, double z_max,
                      std::int32_t nx, std::int32_t ny, std::int32_t nz);

    // Загрузка готовой сетки из внешнего источника (Python-стороны через ctypes).
    void load(const double* nodes_xyz, std::int32_t n_nodes,
              const std::int32_t* elements, std::int32_t n_elements,
              const std::int32_t* boundary_nodes,
              const std::int32_t* boundary_face_ids,
              std::int32_t n_boundary_faces);

    // --- Геометрические вычисления ------------------------------------------
    // Объём элемента e (формула 2.17).
    double element_volume(std::int32_t e) const;

    // Коэффициенты b_i, c_i, d_i для базисных функций (формула 1.19).
    // На выходе: b[4], c[4], d[4] — каждый из массивов длины 4.
    // Возвращает 6*V (для удобства последующего деления).
    double element_gradients(std::int32_t e,
                             std::array<double, 4>& b,
                             std::array<double, 4>& c,
                             std::array<double, 4>& d) const;

    // Площадь треугольной граничной грани (через векторное произведение).
    double face_area(std::int32_t face_idx) const;

    // --- Геттеры -------------------------------------------------------------
    std::int32_t n_nodes()    const noexcept { return static_cast<std::int32_t>(nodes_.size()); }
    std::int32_t n_elements() const noexcept { return static_cast<std::int32_t>(elements_.size()); }
    std::int32_t n_boundary_faces() const noexcept {
        return static_cast<std::int32_t>(boundary_faces_.size());
    }

    const std::vector<Node>&            nodes()    const noexcept { return nodes_; }
    const std::vector<Tetrahedron>&     elements() const noexcept { return elements_; }
    const std::vector<BoundaryFaceTri>& boundary_faces() const noexcept { return boundary_faces_; }

private:
    std::vector<Node>            nodes_;
    std::vector<Tetrahedron>     elements_;
    std::vector<BoundaryFaceTri> boundary_faces_;
};

} // namespace fem

#endif // FEM_CORE_MESH_HPP
