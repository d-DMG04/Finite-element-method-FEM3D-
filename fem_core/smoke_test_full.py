#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Расширенный smoke-test ядра.

Тест 1 (T1 ПЗ): чистый Дирихле, линейное аналитическое решение T(z)=100z.
Тест 2: Робен на верхней грани + Дирихле снизу.
         Аналитика для 1D-задачи между Z=0 (T=T0) и Z=L (Робен с T_inf, alpha):
             T(z) = T0 + (T_inf - T0) * z / (L + lambda/alpha)
Тест 3: проверка теплового потока для T1 — должен быть q = (0, 0, -lambda*100/L).
"""

import ctypes
import os
import sys

LIB_PATH = os.path.join(os.path.dirname(__file__), "fem_core.so")
lib = ctypes.CDLL(LIB_PATH)

# --- Сигнатуры ---------------------------------------------------------------
lib.fem_generate_box.argtypes = [ctypes.c_double] * 6 + [ctypes.c_int32] * 3
lib.fem_generate_box.restype  = ctypes.c_int32
lib.fem_get_node_count.restype    = ctypes.c_int32
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
lib.fem_compute_fluxes.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.fem_compute_fluxes.restype  = ctypes.c_int32
lib.fem_get_solver_info.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int32),
]
lib.fem_get_solver_info.restype = ctypes.c_int32
lib.fem_free.restype = ctypes.c_int32

BC_DIRICHLET, BC_NEUMANN, BC_ROBIN = 1, 2, 3
FACES = {"X-": 0, "X+": 1, "Y-": 2, "Y+": 3, "Z-": 4, "Z+": 5}


def diag():
    iters    = ctypes.c_int32(0)
    residual = ctypes.c_double(0.0)
    t        = ctypes.c_double(0.0)
    conv     = ctypes.c_int32(0)
    lib.fem_get_solver_info(ctypes.byref(iters), ctypes.byref(residual),
                            ctypes.byref(t), ctypes.byref(conv))
    return iters.value, residual.value, t.value, bool(conv.value)


def test_dirichlet_linear() -> bool:
    print("--- Тест 1: чистый Дирихле, T(z) = 100*z ---")
    lib.fem_free()
    lib.fem_generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 12, 12, 12)
    n = lib.fem_get_node_count()
    nodes = (ctypes.c_double * (3 * n))()
    lib.fem_get_nodes(nodes)
    lib.fem_set_material(1.0, 0.0)
    lib.fem_set_boundary_condition(FACES["Z-"], BC_DIRICHLET, 0.0,   0.0, 0.0, 0.0)
    lib.fem_set_boundary_condition(FACES["Z+"], BC_DIRICHLET, 100.0, 0.0, 0.0, 0.0)
    for f in ("X-", "X+", "Y-", "Y+"):
        lib.fem_set_boundary_condition(FACES[f], BC_NEUMANN, 0.0, 0.0, 0.0, 0.0)
    lib.fem_solve(1e-10, 5000)
    it, r, t, c = diag()
    print(f"  CG: {it} итераций, невязка {r:.2e}, {t*1000:.2f} мс, conv={c}")
    T = (ctypes.c_double * n)()
    lib.fem_get_temperature(T)
    err = max(abs(T[i] - 100.0 * nodes[3 * i + 2]) for i in range(n))
    print(f"  max|T - T_exact| = {err:.3e}")
    return err < 1e-6


def test_robin_1d() -> bool:
    print("--- Тест 2: Дирихле + Робен, аналитика 1D ---")
    lib.fem_free()
    L      = 0.1            # длина по z, м
    lam    = 50.0           # Вт/(м·К)
    T0     = 100.0          # температура снизу (Дирихле)
    alpha  = 25.0           # коэф. теплоотдачи
    T_inf  = 20.0           # температура среды
    # Аналитическое решение 1D: T(z) = T0 + (T_inf - T0) * z / (L + lam/alpha)
    denom = L + lam / alpha

    lib.fem_generate_box(0.0, L, 0.0, L, 0.0, L, 16, 16, 16)
    n = lib.fem_get_node_count()
    nodes = (ctypes.c_double * (3 * n))()
    lib.fem_get_nodes(nodes)
    lib.fem_set_material(lam, 0.0)
    lib.fem_set_boundary_condition(FACES["Z-"], BC_DIRICHLET, T0, 0.0, 0.0, 0.0)
    lib.fem_set_boundary_condition(FACES["Z+"], BC_ROBIN,     0.0, 0.0, alpha, T_inf)
    for f in ("X-", "X+", "Y-", "Y+"):
        lib.fem_set_boundary_condition(FACES[f], BC_NEUMANN, 0.0, 0.0, 0.0, 0.0)
    lib.fem_solve(1e-10, 5000)
    it, r, t, c = diag()
    print(f"  CG: {it} итераций, невязка {r:.2e}, {t*1000:.2f} мс, conv={c}")
    T = (ctypes.c_double * n)()
    lib.fem_get_temperature(T)
    err = 0.0
    for i in range(n):
        z = nodes[3 * i + 2]
        T_exact = T0 + (T_inf - T0) * z / denom
        e = abs(T[i] - T_exact)
        if e > err: err = e
    print(f"  max|T - T_exact| = {err:.3e}  (1D-аналитика, ожидается ~машинный 0)")
    return err < 1e-6


def test_flux() -> bool:
    print("--- Тест 3: тепловой поток для линейного решения T=100z ---")
    lib.fem_free()
    lam = 1.0
    L   = 1.0
    lib.fem_generate_box(0.0, L, 0.0, L, 0.0, L, 8, 8, 8)
    n = lib.fem_get_node_count()
    lib.fem_set_material(lam, 0.0)
    lib.fem_set_boundary_condition(FACES["Z-"], BC_DIRICHLET, 0.0,   0.0, 0.0, 0.0)
    lib.fem_set_boundary_condition(FACES["Z+"], BC_DIRICHLET, 100.0, 0.0, 0.0, 0.0)
    for f in ("X-", "X+", "Y-", "Y+"):
        lib.fem_set_boundary_condition(FACES[f], BC_NEUMANN, 0.0, 0.0, 0.0, 0.0)
    lib.fem_solve(1e-10, 5000)
    flux = (ctypes.c_double * (3 * n))()
    lib.fem_compute_fluxes(flux)
    # Ожидаем q = (0, 0, -lambda*dT/dz) = (0, 0, -100)
    err_qx = max(abs(flux[3 * i + 0]) for i in range(n))
    err_qy = max(abs(flux[3 * i + 1]) for i in range(n))
    err_qz = max(abs(flux[3 * i + 2] + 100.0) for i in range(n))
    print(f"  max|qx| = {err_qx:.3e}, max|qy| = {err_qy:.3e}, "
          f"max|qz - (-100)| = {err_qz:.3e}")
    return err_qx < 1e-6 and err_qy < 1e-6 and err_qz < 1e-6


def main() -> int:
    ok = True
    ok &= test_dirichlet_linear()
    ok &= test_robin_1d()
    ok &= test_flux()
    lib.fem_free()
    print()
    print("ИТОГ:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
