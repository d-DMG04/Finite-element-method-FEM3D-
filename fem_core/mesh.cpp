// =============================================================================
// mesh.cpp
// -----------------------------------------------------------------------------
// Реализация генерации структурированной тетраэдральной сетки на
// параллелепипеде. Каждая прямоугольная ячейка nx*ny*nz регулярной решётки
// разбивается на 6 тетраэдров (стандартная декомпозиция Куна).
//
// Соответствует разделу 3.1.2 пояснительной записки.
// =============================================================================

#include "mesh.hpp"

#include <cmath>
#include <stdexcept>

namespace fem {

namespace {

// -----------------------------------------------------------------------------
// Стандартная декомпозиция куба на 6 тетраэдров (декомпозиция Куна).
// Локальные индексы вершин куба:
//   0: (0,0,0)  1: (1,0,0)  2: (1,1,0)  3: (0,1,0)
//   4: (0,0,1)  5: (1,0,1)  6: (1,1,1)  7: (0,1,1)
//
// Все шесть тетраэдров содержат главную диагональ 0-6.
// -----------------------------------------------------------------------------
constexpr int kKuhnTetrahedra[6][4] = {
    {0, 1, 2, 6},
    {0, 2, 3, 6},
    {0, 3, 7, 6},
    {0, 7, 4, 6},
    {0, 4, 5, 6},
    {0, 5, 1, 6}
};

// -----------------------------------------------------------------------------
// Преобразование индексов узла регулярной решётки (i, j, k) в линейный.
// -----------------------------------------------------------------------------
inline std::int32_t node_id(std::int32_t i, std::int32_t j, std::int32_t k,
                            std::int32_t nx, std::int32_t ny) {
    return i + (nx + 1) * (j + (ny + 1) * k);
}

} // namespace

// -----------------------------------------------------------------------------
// Генерация сетки на параллелепипеде.
// -----------------------------------------------------------------------------
void Mesh::generate_box(double x_min, double x_max,
                        double y_min, double y_max,
                        double z_min, double z_max,
                        std::int32_t nx, std::int32_t ny, std::int32_t nz) {
    if (nx < 1 || ny < 1 || nz < 1) {
        throw std::invalid_argument("Mesh::generate_box: nx, ny, nz must be >= 1");
    }
    if (x_max <= x_min || y_max <= y_min || z_max <= z_min) {
        throw std::invalid_argument("Mesh::generate_box: invalid box dimensions");
    }

    const std::int32_t nnx = nx + 1;
    const std::int32_t nny = ny + 1;
    const std::int32_t nnz = nz + 1;

    // --- Узлы ----------------------------------------------------------------
    nodes_.clear();
    nodes_.reserve(static_cast<std::size_t>(nnx) *
                   static_cast<std::size_t>(nny) *
                   static_cast<std::size_t>(nnz));

    const double hx = (x_max - x_min) / static_cast<double>(nx);
    const double hy = (y_max - y_min) / static_cast<double>(ny);
    const double hz = (z_max - z_min) / static_cast<double>(nz);

    for (std::int32_t k = 0; k < nnz; ++k) {
        for (std::int32_t j = 0; j < nny; ++j) {
            for (std::int32_t i = 0; i < nnx; ++i) {
                Node n;
                n.x = x_min + static_cast<double>(i) * hx;
                n.y = y_min + static_cast<double>(j) * hy;
                n.z = z_min + static_cast<double>(k) * hz;
                nodes_.push_back(n);
            }
        }
    }

    // --- Элементы (тетраэдры по 6 на ячейку) ---------------------------------
    elements_.clear();
    elements_.reserve(static_cast<std::size_t>(nx) *
                      static_cast<std::size_t>(ny) *
                      static_cast<std::size_t>(nz) * 6);

    for (std::int32_t k = 0; k < nz; ++k) {
        for (std::int32_t j = 0; j < ny; ++j) {
            for (std::int32_t i = 0; i < nx; ++i) {
                std::int32_t v[8];
                v[0] = node_id(i,     j,     k,     nx, ny);
                v[1] = node_id(i + 1, j,     k,     nx, ny);
                v[2] = node_id(i + 1, j + 1, k,     nx, ny);
                v[3] = node_id(i,     j + 1, k,     nx, ny);
                v[4] = node_id(i,     j,     k + 1, nx, ny);
                v[5] = node_id(i + 1, j,     k + 1, nx, ny);
                v[6] = node_id(i + 1, j + 1, k + 1, nx, ny);
                v[7] = node_id(i,     j + 1, k + 1, nx, ny);

                for (int t = 0; t < 6; ++t) {
                    Tetrahedron tet;
                    tet.nodes[0] = v[kKuhnTetrahedra[t][0]];
                    tet.nodes[1] = v[kKuhnTetrahedra[t][1]];
                    tet.nodes[2] = v[kKuhnTetrahedra[t][2]];
                    tet.nodes[3] = v[kKuhnTetrahedra[t][3]];

                    // Гарантируем положительный объём: при необходимости —
                    // меняем местами две вершины.
                    elements_.push_back(tet);
                    if (element_volume(static_cast<std::int32_t>(elements_.size()) - 1) < 0.0) {
                        std::swap(elements_.back().nodes[1], elements_.back().nodes[2]);
                    }
                }
            }
        }
    }

    // --- Граничные грани -----------------------------------------------------
    // Обходим все грани всех тетраэдров и оставляем те, у которых все три
    // вершины лежат на одной из шести граней внешнего параллелепипеда.
    // Локальные грани тетраэдра (4 шт., противоположные вершине i):
    constexpr int kTetFaceLocal[4][3] = {
        {1, 2, 3}, // напротив 0
        {0, 3, 2}, // напротив 1
        {0, 1, 3}, // напротив 2
        {0, 2, 1}  // напротив 3
    };

    auto on_face = [&](std::int32_t node_idx, BoundaryFace face) -> bool {
        const Node& n = nodes_[static_cast<std::size_t>(node_idx)];
        const double tol = 1e-12;
        switch (face) {
            case FACE_X_MINUS: return std::fabs(n.x - x_min) < tol;
            case FACE_X_PLUS:  return std::fabs(n.x - x_max) < tol;
            case FACE_Y_MINUS: return std::fabs(n.y - y_min) < tol;
            case FACE_Y_PLUS:  return std::fabs(n.y - y_max) < tol;
            case FACE_Z_MINUS: return std::fabs(n.z - z_min) < tol;
            case FACE_Z_PLUS:  return std::fabs(n.z - z_max) < tol;
            default: return false;
        }
    };

    boundary_faces_.clear();
    for (std::size_t e = 0; e < elements_.size(); ++e) {
        const auto& tet = elements_[e];
        for (int lf = 0; lf < 4; ++lf) {
            const std::int32_t a = tet.nodes[kTetFaceLocal[lf][0]];
            const std::int32_t b = tet.nodes[kTetFaceLocal[lf][1]];
            const std::int32_t c = tet.nodes[kTetFaceLocal[lf][2]];
            for (int f = 0; f < FACE_COUNT; ++f) {
                const auto face = static_cast<BoundaryFace>(f);
                if (on_face(a, face) && on_face(b, face) && on_face(c, face)) {
                    BoundaryFaceTri bf;
                    bf.nodes[0] = a;
                    bf.nodes[1] = b;
                    bf.nodes[2] = c;
                    bf.face_id  = f;
                    boundary_faces_.push_back(bf);
                    break; // грань может принадлежать только одной плоскости
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Загрузка готовой сетки из массивов, переданных Python-стороной.
// -----------------------------------------------------------------------------
void Mesh::load(const double* nodes_xyz, std::int32_t n_nodes,
                const std::int32_t* elements, std::int32_t n_elements,
                const std::int32_t* boundary_nodes,
                const std::int32_t* boundary_face_ids,
                std::int32_t n_boundary_faces) {
    if (n_nodes <= 0 || n_elements <= 0) {
        throw std::invalid_argument("Mesh::load: empty mesh");
    }
    nodes_.resize(static_cast<std::size_t>(n_nodes));
    for (std::int32_t i = 0; i < n_nodes; ++i) {
        nodes_[static_cast<std::size_t>(i)].x = nodes_xyz[3 * i + 0];
        nodes_[static_cast<std::size_t>(i)].y = nodes_xyz[3 * i + 1];
        nodes_[static_cast<std::size_t>(i)].z = nodes_xyz[3 * i + 2];
    }
    elements_.resize(static_cast<std::size_t>(n_elements));
    for (std::int32_t e = 0; e < n_elements; ++e) {
        elements_[static_cast<std::size_t>(e)].nodes[0] = elements[4 * e + 0];
        elements_[static_cast<std::size_t>(e)].nodes[1] = elements[4 * e + 1];
        elements_[static_cast<std::size_t>(e)].nodes[2] = elements[4 * e + 2];
        elements_[static_cast<std::size_t>(e)].nodes[3] = elements[4 * e + 3];
        if (element_volume(e) < 0.0) {
            std::swap(elements_[static_cast<std::size_t>(e)].nodes[1],
                      elements_[static_cast<std::size_t>(e)].nodes[2]);
        }
    }
    boundary_faces_.resize(static_cast<std::size_t>(n_boundary_faces));
    for (std::int32_t f = 0; f < n_boundary_faces; ++f) {
        boundary_faces_[static_cast<std::size_t>(f)].nodes[0] = boundary_nodes[3 * f + 0];
        boundary_faces_[static_cast<std::size_t>(f)].nodes[1] = boundary_nodes[3 * f + 1];
        boundary_faces_[static_cast<std::size_t>(f)].nodes[2] = boundary_nodes[3 * f + 2];
        boundary_faces_[static_cast<std::size_t>(f)].face_id  = boundary_face_ids[f];
    }
}

// -----------------------------------------------------------------------------
// Объём тетраэдра (формула 2.17 пояснительной записки):
//     V = (1/6) det[ (x2 - x1) (x3 - x1) (x4 - x1) ]
// -----------------------------------------------------------------------------
double Mesh::element_volume(std::int32_t e) const {
    const auto& tet = elements_[static_cast<std::size_t>(e)];
    const Node& n0 = nodes_[static_cast<std::size_t>(tet.nodes[0])];
    const Node& n1 = nodes_[static_cast<std::size_t>(tet.nodes[1])];
    const Node& n2 = nodes_[static_cast<std::size_t>(tet.nodes[2])];
    const Node& n3 = nodes_[static_cast<std::size_t>(tet.nodes[3])];

    const double a1 = n1.x - n0.x, a2 = n1.y - n0.y, a3 = n1.z - n0.z;
    const double b1 = n2.x - n0.x, b2 = n2.y - n0.y, b3 = n2.z - n0.z;
    const double c1 = n3.x - n0.x, c2 = n3.y - n0.y, c3 = n3.z - n0.z;

    const double det = a1 * (b2 * c3 - b3 * c2)
                     - a2 * (b1 * c3 - b3 * c1)
                     + a3 * (b1 * c2 - b2 * c1);
    return det / 6.0;
}

// -----------------------------------------------------------------------------
// Градиенты базисных функций φ_i = (a_i + b_i*x + c_i*y + d_i*z) / (6V).
// На выходе массивы b[4], c[4], d[4]; возвращает 6V (без знака — гарантируем
// положительность при сборке).
//
// Коэффициенты b, c, d — алгебраические дополнения матрицы координат вершин.
// -----------------------------------------------------------------------------
double Mesh::element_gradients(std::int32_t e,
                               std::array<double, 4>& b,
                               std::array<double, 4>& c,
                               std::array<double, 4>& d) const {
    const auto& tet = elements_[static_cast<std::size_t>(e)];
    const Node& n0 = nodes_[static_cast<std::size_t>(tet.nodes[0])];
    const Node& n1 = nodes_[static_cast<std::size_t>(tet.nodes[1])];
    const Node& n2 = nodes_[static_cast<std::size_t>(tet.nodes[2])];
    const Node& n3 = nodes_[static_cast<std::size_t>(tet.nodes[3])];

    const double x1 = n0.x, y1 = n0.y, z1 = n0.z;
    const double x2 = n1.x, y2 = n1.y, z2 = n1.z;
    const double x3 = n2.x, y3 = n2.y, z3 = n2.z;
    const double x4 = n3.x, y4 = n3.y, z4 = n3.z;

    // b_i — алгебраические дополнения по столбцу x;
    // c_i — по y; d_i — по z.
    // Вершины пронумерованы 1..4 (в коде 0..3) — формулы стандартные.
    b[0] = -((y3 - y2)*(z4 - z2) - (z3 - z2)*(y4 - y2));
    c[0] =  ((x3 - x2)*(z4 - z2) - (z3 - z2)*(x4 - x2));
    d[0] = -((x3 - x2)*(y4 - y2) - (y3 - y2)*(x4 - x2));

    b[1] =  ((y3 - y1)*(z4 - z1) - (z3 - z1)*(y4 - y1));
    c[1] = -((x3 - x1)*(z4 - z1) - (z3 - z1)*(x4 - x1));
    d[1] =  ((x3 - x1)*(y4 - y1) - (y3 - y1)*(x4 - x1));

    b[2] = -((y2 - y1)*(z4 - z1) - (z2 - z1)*(y4 - y1));
    c[2] =  ((x2 - x1)*(z4 - z1) - (z2 - z1)*(x4 - x1));
    d[2] = -((x2 - x1)*(y4 - y1) - (y2 - y1)*(x4 - x1));

    b[3] =  ((y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1));
    c[3] = -((x2 - x1)*(z3 - z1) - (z2 - z1)*(x3 - x1));
    d[3] =  ((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1));

    // 6V — определитель той же матрицы; вычисляем через element_volume.
    const double V6 = 6.0 * element_volume(e);
    return V6;
}

// -----------------------------------------------------------------------------
// Площадь треугольной граничной грани через векторное произведение.
// -----------------------------------------------------------------------------
double Mesh::face_area(std::int32_t face_idx) const {
    const auto& f = boundary_faces_[static_cast<std::size_t>(face_idx)];
    const Node& a = nodes_[static_cast<std::size_t>(f.nodes[0])];
    const Node& b = nodes_[static_cast<std::size_t>(f.nodes[1])];
    const Node& c = nodes_[static_cast<std::size_t>(f.nodes[2])];

    const double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
    const double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;

    const double cx = uy * vz - uz * vy;
    const double cy = uz * vx - ux * vz;
    const double cz = ux * vy - uy * vx;

    return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
}

} // namespace fem
