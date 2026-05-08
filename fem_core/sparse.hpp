// =============================================================================
// sparse.hpp
// -----------------------------------------------------------------------------
// Разреженная матрица в формате CSR (Compressed Sparse Row).
// Внутренняя структура данных вычислительного ядра.
//
// Формат CSR:
//   values      — вектор всех ненулевых элементов, упорядоченных по строкам;
//   col_indices — номер столбца для каждого ненулевого элемента;
//   row_ptr     — длиной N+1, индексы в values, с которых начинается строка.
//
// Соответствует подсистеме «Внутренняя структура данных» (раздел 2.4)
// и формуле раздела 2.2.5 пояснительной записки.
// =============================================================================

#ifndef FEM_CORE_SPARSE_HPP
#define FEM_CORE_SPARSE_HPP

#include <cstddef>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace fem {

// -----------------------------------------------------------------------------
// Разреженная матрица в формате CSR.
// Матрица квадратная (N x N), симметричная, положительно определённая.
// -----------------------------------------------------------------------------
class CSRMatrix {
public:
    using Index = std::int32_t;
    using Value = double;

    CSRMatrix() = default;
    explicit CSRMatrix(Index n) : n_(n), row_ptr_(static_cast<std::size_t>(n) + 1, 0) {}

    // --- Размерности ---------------------------------------------------------
    Index size()  const noexcept { return n_; }
    Index nnz()   const noexcept { return static_cast<Index>(values_.size()); }

    // --- Доступ к внутренним массивам ---------------------------------------
    const std::vector<Value>& values()      const noexcept { return values_; }
    const std::vector<Index>& col_indices() const noexcept { return col_indices_; }
    const std::vector<Index>& row_ptr()     const noexcept { return row_ptr_; }

    std::vector<Value>& values()      noexcept { return values_; }
    std::vector<Index>& col_indices() noexcept { return col_indices_; }
    std::vector<Index>& row_ptr()     noexcept { return row_ptr_; }

    // -------------------------------------------------------------------------
    // Установка шаблона разреженности (первый проход сборки).
    // На вход — для каждой строки множество столбцов (отсортированное,
    // без дублей). Память под values и col_indices выделяется здесь.
    // -------------------------------------------------------------------------
    void set_pattern(const std::vector<std::vector<Index>>& adjacency) {
        if (static_cast<Index>(adjacency.size()) != n_) {
            throw std::invalid_argument("CSRMatrix::set_pattern: size mismatch");
        }
        row_ptr_.assign(static_cast<std::size_t>(n_) + 1, 0);
        for (Index i = 0; i < n_; ++i) {
            row_ptr_[static_cast<std::size_t>(i) + 1] =
                row_ptr_[static_cast<std::size_t>(i)] +
                static_cast<Index>(adjacency[static_cast<std::size_t>(i)].size());
        }
        const Index total = row_ptr_.back();
        values_.assign(static_cast<std::size_t>(total), 0.0);
        col_indices_.resize(static_cast<std::size_t>(total));
        for (Index i = 0; i < n_; ++i) {
            const auto& row = adjacency[static_cast<std::size_t>(i)];
            const Index off = row_ptr_[static_cast<std::size_t>(i)];
            for (std::size_t k = 0; k < row.size(); ++k) {
                col_indices_[static_cast<std::size_t>(off) + k] = row[k];
            }
        }
    }

    // -------------------------------------------------------------------------
    // Поиск позиции элемента (i, j) в массиве values.
    // Возвращает -1, если шаблон не содержит этой позиции (что является
    // диагностической ошибкой при сборке).
    // -------------------------------------------------------------------------
    Index find_position(Index i, Index j) const {
        const Index beg = row_ptr_[static_cast<std::size_t>(i)];
        const Index end = row_ptr_[static_cast<std::size_t>(i) + 1];
        // Бинарный поиск, т.к. col_indices в строке отсортирован.
        const auto first = col_indices_.begin() + beg;
        const auto last  = col_indices_.begin() + end;
        const auto it = std::lower_bound(first, last, j);
        if (it == last || *it != j) return -1;
        return static_cast<Index>(it - col_indices_.begin());
    }

    // -------------------------------------------------------------------------
    // Атомарное добавление к (i, j) в стиле OpenMP atomic.
    // Используется в параллельной сборке.
    // -------------------------------------------------------------------------
    inline void add_atomic(Index i, Index j, Value v) {
        const Index pos = find_position(i, j);
        if (pos < 0) {
            // В корректно построенном шаблоне такого быть не должно.
            // Тихий выход — иначе была бы гонка из-за исключения из OpenMP-цикла.
            return;
        }
        #pragma omp atomic
        values_[static_cast<std::size_t>(pos)] += v;
    }

    // -------------------------------------------------------------------------
    // Прямое добавление к (i, j) — для последовательной сборки.
    // -------------------------------------------------------------------------
    inline void add(Index i, Index j, Value v) {
        const Index pos = find_position(i, j);
        if (pos < 0) return;
        values_[static_cast<std::size_t>(pos)] += v;
    }

    // -------------------------------------------------------------------------
    // Произведение разреженной матрицы на вектор: y = A * x.
    // Параллелизовано по строкам через OpenMP — это горячий цикл CG.
    // -------------------------------------------------------------------------
    void multiply(const std::vector<Value>& x, std::vector<Value>& y) const {
        #pragma omp parallel for schedule(static)
        for (Index i = 0; i < n_; ++i) {
            const Index beg = row_ptr_[static_cast<std::size_t>(i)];
            const Index end = row_ptr_[static_cast<std::size_t>(i) + 1];
            Value s = 0.0;
            for (Index k = beg; k < end; ++k) {
                s += values_[static_cast<std::size_t>(k)] *
                     x[static_cast<std::size_t>(col_indices_[static_cast<std::size_t>(k)])];
            }
            y[static_cast<std::size_t>(i)] = s;
        }
    }

    // -------------------------------------------------------------------------
    // Получение диагонали для построения предобусловливателя Якоби.
    // -------------------------------------------------------------------------
    void diagonal(std::vector<Value>& diag) const {
        diag.assign(static_cast<std::size_t>(n_), 0.0);
        for (Index i = 0; i < n_; ++i) {
            const Index pos = find_position(i, i);
            if (pos >= 0) {
                diag[static_cast<std::size_t>(i)] =
                    values_[static_cast<std::size_t>(pos)];
            }
        }
    }

    // -------------------------------------------------------------------------
    // Обнуление строки и столбца k с установкой 1 на диагональ.
    // Используется при учёте граничных условий Дирихле (раздел 1.2.8).
    // Сохраняет симметрию матрицы.
    // -------------------------------------------------------------------------
    void zero_row_and_column(Index k) {
        // 1) Зануляем строку k.
        const Index beg = row_ptr_[static_cast<std::size_t>(k)];
        const Index end = row_ptr_[static_cast<std::size_t>(k) + 1];
        for (Index p = beg; p < end; ++p) {
            values_[static_cast<std::size_t>(p)] = 0.0;
        }
        // 2) Зануляем столбец k во всех остальных строках.
        for (Index i = 0; i < n_; ++i) {
            if (i == k) continue;
            const Index pos = find_position(i, k);
            if (pos >= 0) {
                values_[static_cast<std::size_t>(pos)] = 0.0;
            }
        }
        // 3) Ставим 1 на диагональ.
        const Index diag = find_position(k, k);
        if (diag >= 0) {
            values_[static_cast<std::size_t>(diag)] = 1.0;
        }
    }

    void resize(Index n) {
        n_ = n;
        row_ptr_.assign(static_cast<std::size_t>(n) + 1, 0);
        values_.clear();
        col_indices_.clear();
    }

private:
    Index n_ = 0;
    std::vector<Value> values_;
    std::vector<Index> col_indices_;
    std::vector<Index> row_ptr_;
};

} // namespace fem

#endif // FEM_CORE_SPARSE_HPP
