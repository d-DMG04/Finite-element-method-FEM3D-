// =============================================================================
// api.hpp
// -----------------------------------------------------------------------------
// C-API вычислительного ядра. 12 функций, экспортируемых под extern "C"
// для вызова из Python-стороны через ctypes (модуль core_bridge.py).
//
// Все функции возвращают код результата:
//   0  — успех;
//  >0  — предупреждение (например, итерации не сошлись);
//  <0  — ошибка (некорректные аргументы, переполнение и т.п.).
//
// Соответствует таблице 3.4 пояснительной записки.
// =============================================================================

#ifndef FEM_CORE_API_HPP
#define FEM_CORE_API_HPP

#include <cstdint>

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef FEM_CORE_BUILD
#    define FEM_API __declspec(dllexport)
#  else
#    define FEM_API __declspec(dllimport)
#  endif
#else
#  define FEM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// 1) Генерация структурированной сетки на параллелепипеде.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_generate_box(
    double x_min, double x_max,
    double y_min, double y_max,
    double z_min, double z_max,
    std::int32_t nx, std::int32_t ny, std::int32_t nz);

// -----------------------------------------------------------------------------
// 2) Загрузка готовой сетки из массивов Python.
//    nodes_xyz       — n_nodes * 3 значений типа double;
//    elements        — n_elements * 4 значений типа int32 (узлы тетраэдров);
//    boundary_nodes  — n_boundary_faces * 3 значений типа int32;
//    boundary_face_ids — n_boundary_faces значений типа int32.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_load_mesh(
    const double* nodes_xyz, std::int32_t n_nodes,
    const std::int32_t* elements, std::int32_t n_elements,
    const std::int32_t* boundary_nodes,
    const std::int32_t* boundary_face_ids,
    std::int32_t n_boundary_faces);

// -----------------------------------------------------------------------------
// 3) Запрос числа узлов сетки.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_get_node_count(void);

// -----------------------------------------------------------------------------
// 4) Передача массива координат узлов в Python.
//    out_xyz — буфер длины 3 * n_nodes (выделяет вызывающий).
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_get_nodes(double* out_xyz);

// -----------------------------------------------------------------------------
// 5) Передача матрицы связности элементов в Python.
//    out_conn — буфер длины 4 * n_elements (выделяет вызывающий).
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_get_elements(std::int32_t* out_conn);

// -----------------------------------------------------------------------------
// 6) Задание коэффициента теплопроводности и плотности источников.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_set_material(double lambda_val, double Q);

// Установка плотности (кг/м³) и удельной теплоёмкости (Дж/(кг·К)) глобального
// материала. Используются в нестационарной задаче (масс-матрица).
FEM_API std::int32_t fem_set_thermal_capacity(double rho, double cp);

// Нестационарная задача: серия снимков T(t).
//   t_end, dt   — финальное время и шаг, сек
//   T_init      — начальная T (одна на всю область)
//   n_save      — число сохранённых снимков (>= 2)
//   out_times   — буфер длины n_save
//   out_T       — буфер длины n_save * n_nodes
//   tol, max_iter — параметры CG на каждом шаге
FEM_API std::int32_t fem_solve_transient(
    double t_end, double dt, double T_init,
    std::int32_t n_save, double* out_times, double* out_T,
    double tol, std::int32_t max_iter);

// -----------------------------------------------------------------------------
// 7) Задание граничного условия на одной из 6 граней параллелепипеда.
//    face_id: 0..5 (X-, X+, Y-, Y+, Z-, Z+);
//    bc_type: 1=Dirichlet, 2=Neumann, 3=Robin.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_set_boundary_condition(
    std::int32_t face_id, std::int32_t bc_type,
    double T0, double q0, double alpha, double T_inf);

// -----------------------------------------------------------------------------
// 8) Сборка матрицы и решение СЛАУ.
//    tol      — относительная норма невязки;
//    max_iter — верхний лимит итераций.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_solve(double tol, std::int32_t max_iter);

// -----------------------------------------------------------------------------
// 9) Получение узлового вектора температур.
//    out_T — буфер длины n_nodes.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_get_temperature(double* out_T);

// -----------------------------------------------------------------------------
// 10) Вычисление и передача тепловых потоков (узловых).
//     out_flux — буфер длины 3 * n_nodes.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_compute_fluxes(double* out_flux);

// -----------------------------------------------------------------------------
// 11) Получение диагностики решателя.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_get_solver_info(
    std::int32_t* out_iterations,
    double*       out_residual,
    double*       out_time_seconds,
    std::int32_t* out_converged);

// -----------------------------------------------------------------------------
// 12) Освобождение всех ресурсов ядра. После вызова все указатели
//     становятся недействительными; для повторного использования
//     требуется вновь сгенерировать или загрузить сетку.
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_free(void);

// -----------------------------------------------------------------------------
// Дополнительная информация (не входит в 12, но полезна для GUI).
// -----------------------------------------------------------------------------
FEM_API std::int32_t fem_get_element_count(void);
FEM_API std::int32_t fem_get_boundary_face_count(void);

// =============================================================================
// Локальные источники тепла (раздел 3.3.11 ПЗ).
// =============================================================================
// Точечный источник: добавляет мощность P (Вт) к правой части в строке node_idx.
//                    Знак P: положительный — нагрев, отрицательный — отвод.
// Объёмный источник: добавляет плотность Q0 (Вт/м³) ко всем элементам, чьи
//                    центры попадают в подобласть.
//   shape: 0 = box (xmin..xmax, ymin..ymax, zmin..zmax),
//          1 = sphere (cx, cy, cz, radius).
// Параметры shape передаются в массиве params[6]:
//   box:    [xmin, ymin, zmin, xmax, ymax, zmax]
//   sphere: [cx,   cy,   cz,   radius, 0, 0]
//
// fem_clear_sources очищает оба списка (вызывайте перед каждой новой задачей).
// =============================================================================

FEM_API std::int32_t fem_clear_sources(void);

FEM_API std::int32_t fem_add_point_source(std::int32_t node_idx, double power);

FEM_API std::int32_t fem_add_volume_source(std::int32_t shape,
                                           const double* params,
                                           double Q0);

// =============================================================================
// Поузельное переопределение значений Дирихле (раздел 3.4.3 — нужно для
// верификационной задачи T3 с гладкой аналитикой; полезно также для случаев,
// когда на грани известен пространственный профиль температуры из эксперимента).
//
// fem_set_node_dirichlet(idx, value) — фиксировать значение T = value в узле idx.
//     Узел не обязан принадлежать грани с BC_DIRICHLET; функция работает
//     независимо. Несколько вызовов на один узел: побеждает последнее значение.
//
// fem_clear_node_dirichlet() — снять все поузельные переопределения.
// =============================================================================
FEM_API std::int32_t fem_set_node_dirichlet(std::int32_t node_idx, double value);
FEM_API std::int32_t fem_clear_node_dirichlet(void);

// =============================================================================
// Прогресс-callback и прерывание расчёта (раздел 3.3.x — отзывчивый GUI).
//
// fem_set_progress_callback(cb) — регистрирует Python-функцию, которая будет
//     вызываться из CG-итерации каждые ~5 итераций. Сигнатура:
//         int32_t cb(int32_t iteration, double residual)
//     Возврат 1 — продолжать, 0 — прервать.
//     Передать NULL — снять callback.
//
// fem_request_cancel() — флаг прерывания; устанавливается из другого потока,
//     CG-цикл его читает и прерывается. Используется для кнопки Cancel в GUI.
//     После прерывания fem_solve возвращает 2.
//
// fem_clear_cancel() — сбросить флаг (новый расчёт сделает это автоматически).
// =============================================================================
FEM_API std::int32_t fem_set_progress_callback(
    std::int32_t (*cb)(std::int32_t, double));
FEM_API std::int32_t fem_request_cancel(void);
FEM_API std::int32_t fem_clear_cancel(void);

// =============================================================================
// Регионы материалов: задание разного λ, Q в разных частях детали.
//
// Базовый материал (id = 0) — это материал, заданный через fem_set_material().
// Дополнительные материалы добавляются через fem_add_material(λ, Q),
// который возвращает их id (1-based).
// Затем материал назначается тетраэдрам, центроиды которых попадают
// в заданный регион (bbox или сфера).
// =============================================================================
FEM_API std::int32_t fem_clear_materials(void);
FEM_API std::int32_t fem_add_material(double lambda_val, double Q_val);
FEM_API std::int32_t fem_assign_material_in_box(
    std::int32_t material_id,
    double x_min, double x_max,
    double y_min, double y_max,
    double z_min, double z_max);
FEM_API std::int32_t fem_assign_material_in_sphere(
    std::int32_t material_id,
    double cx, double cy, double cz, double radius);
FEM_API std::int32_t fem_clear_material_assignments(void);
FEM_API std::int32_t fem_get_material_ids(std::int32_t* out_ids);
FEM_API std::int32_t fem_get_material_count(void);

// =============================================================================
// Анизотропная теплопроводность: задание разных λ по осям X/Y/Z.
//
// fem_set_material_anisotropic — глобальный материал (вместо fem_set_material).
// fem_add_material_anisotropic — региональный материал (вместо fem_add_material).
// =============================================================================
FEM_API std::int32_t fem_set_material_anisotropic(
    double lambda_x, double lambda_y, double lambda_z, double Q_val);
FEM_API std::int32_t fem_add_material_anisotropic(
    double lambda_x, double lambda_y, double lambda_z, double Q_val);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FEM_CORE_API_HPP
