#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke-test вычислительного ядра через ctypes.

Задача T1 (раздел 3.4 ПЗ): куб [0,1]^3, λ=1, Q=0.
Грань Z- : T = 0 (Дирихле)
Грань Z+ : T = 100 (Дирихле)
Прочие грани : Нейман с q=0.

Аналитическое решение: T(x, y, z) = 100 * z (линейный профиль).
Метод P1 на структурированной тетраэдральной сетке должен воспроизвести
это решение с точностью машинного эпсилона.
"""

import ctypes
import os
import sys

LIB_PATH = os.path.join(os.path.dirname(__file__), "fem_core.so")
lib = ctypes.CDLL(LIB_PATH)

# --- Сигнатуры функций -------------------------------------------------------
lib.fem_generate_box.argtypes = [ctypes.c_double] * 6 + [ctypes.c_int32] * 3
lib.fem_generate_box.restype  = ctypes.c_int32

lib.fem_get_node_count.restype = ctypes.c_int32
lib.fem_get_element_count.restype = ctypes.c_int32

lib.fem_get_nodes.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.fem_get_nodes.restype  = ctypes.c_int32

lib.fem_set_material.argtypes = [ctypes.c_double, ctypes.c_double]
lib.fem_set_material.restype  = ctypes.c_int32

lib.fem_set_boundary_condition.argtypes = [
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
lib.fem_set_boundary_condition.restype = ctypes.c_int32

lib.fem_solve.argtypes = [ctypes.c_double, ctypes.c_int32]
lib.fem_solve.restype  = ctypes.c_int32

lib.fem_get_temperature.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.fem_get_temperature.restype  = ctypes.c_int32

lib.fem_get_solver_info.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
]
lib.fem_get_solver_info.restype = ctypes.c_int32

lib.fem_free.restype = ctypes.c_int32

# --- Константы для типов ГУ ---------------------------------------------------
BC_DIRICHLET, BC_NEUMANN, BC_ROBIN = 1, 2, 3

# Идентификаторы граней (см. mesh.hpp::BoundaryFace).
FACE_X_MINUS, FACE_X_PLUS = 0, 1
FACE_Y_MINUS, FACE_Y_PLUS = 2, 3
FACE_Z_MINUS, FACE_Z_PLUS = 4, 5


def main() -> int:
    # 1) Генерация сетки 10x10x10 на единичном кубе.
    nx, ny, nz = 10, 10, 10
    rc = lib.fem_generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
    assert rc == 0, f"fem_generate_box failed: {rc}"

    n_nodes = lib.fem_get_node_count()
    n_elems = lib.fem_get_element_count()
    print(f"Сетка: {n_nodes} узлов, {n_elems} тетраэдров.")
    assert n_nodes == (nx + 1) * (ny + 1) * (nz + 1)
    assert n_elems == 6 * nx * ny * nz

    # 2) Координаты узлов.
    nodes = (ctypes.c_double * (3 * n_nodes))()
    lib.fem_get_nodes(nodes)

    # 3) Материал: λ = 1, Q = 0.
    rc = lib.fem_set_material(1.0, 0.0)
    assert rc == 0

    # 4) Граничные условия:
    #    Z-: T = 0; Z+: T = 100; остальные: Нейман q=0.
    lib.fem_set_boundary_condition(FACE_Z_MINUS, BC_DIRICHLET, 0.0,   0.0, 0.0, 0.0)
    lib.fem_set_boundary_condition(FACE_Z_PLUS,  BC_DIRICHLET, 100.0, 0.0, 0.0, 0.0)
    lib.fem_set_boundary_condition(FACE_X_MINUS, BC_NEUMANN,   0.0,   0.0, 0.0, 0.0)
    lib.fem_set_boundary_condition(FACE_X_PLUS,  BC_NEUMANN,   0.0,   0.0, 0.0, 0.0)
    lib.fem_set_boundary_condition(FACE_Y_MINUS, BC_NEUMANN,   0.0,   0.0, 0.0, 0.0)
    lib.fem_set_boundary_condition(FACE_Y_PLUS,  BC_NEUMANN,   0.0,   0.0, 0.0, 0.0)

    # 5) Решение.
    rc = lib.fem_solve(1e-10, 5000)
    if rc not in (0, 1):
        print(f"fem_solve вернул ошибку: {rc}", file=sys.stderr)
        return 1

    # 6) Диагностика.
    iters    = ctypes.c_int32(0)
    residual = ctypes.c_double(0.0)
    t_solve  = ctypes.c_double(0.0)
    converged = ctypes.c_int32(0)
    lib.fem_get_solver_info(ctypes.byref(iters), ctypes.byref(residual),
                            ctypes.byref(t_solve), ctypes.byref(converged))
    print(f"CG: итераций {iters.value}, невязка {residual.value:.3e}, "
          f"время {t_solve.value*1000:.2f} мс, сошёлся: {bool(converged.value)}")

    # 7) Получение решения.
    T = (ctypes.c_double * n_nodes)()
    lib.fem_get_temperature(T)

    # 8) Сравнение с аналитическим решением T(x,y,z) = 100 * z.
    max_err = 0.0
    for i in range(n_nodes):
        z = nodes[3 * i + 2]
        T_exact = 100.0 * z
        err = abs(T[i] - T_exact)
        if err > max_err:
            max_err = err
    print(f"Максимальное отклонение от аналитики: {max_err:.3e}")

    # Для линейного решения метод P1 точен до округления.
    ok = max_err < 1e-6
    print("РЕЗУЛЬТАТ: PASS" if ok else "РЕЗУЛЬТАТ: FAIL")

    lib.fem_free()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
