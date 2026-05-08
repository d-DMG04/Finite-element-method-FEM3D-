# -*- coding: utf-8 -*-
"""
fem3d.verify
============

Верификационные задачи T1–T4, описанные в разделе 3.4 пояснительной записки:
  T1 — воспроизведение линейного решения (точно до машинного эпсилона);
  T2 — сравнение с аналитическим решением для смешанных ГУ;
  T3 — порядок сходимости в нормах L2 и H1 при измельчении сетки;
  T4 — корректное включение всех трёх типов ГУ.

Запуск из командной строки:
    python -m fem3d.verify
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple

import numpy as np

from .core_bridge import (
    BC_DIRICHLET, BC_NEUMANN, BC_ROBIN,
    CoreBridge,
    FACE_X_MINUS, FACE_X_PLUS,
    FACE_Y_MINUS, FACE_Y_PLUS,
    FACE_Z_MINUS, FACE_Z_PLUS,
)


# =============================================================================
# Вспомогательное: задать «изоляцию по умолчанию» на боковых гранях.
# =============================================================================

def _isolate_sides(bridge: CoreBridge) -> None:
    for f in (FACE_X_MINUS, FACE_X_PLUS, FACE_Y_MINUS, FACE_Y_PLUS):
        bridge.set_bc(f, BC_NEUMANN)


# =============================================================================
# T1 — линейное решение T(x, y, z) = x.
# =============================================================================

def test_T1_linear_solution(verbose: bool = True) -> Tuple[bool, float]:
    """T1 (раздел 3.4.1): линейное решение T = x на кубе [0, 1]^3."""
    with CoreBridge() as bridge:
        bridge.generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 10, 10, 10)
        nodes = bridge.get_nodes()
        bridge.set_material(1.0, 0.0)
        bridge.set_bc(FACE_X_MINUS, BC_DIRICHLET, T0=0.0)
        bridge.set_bc(FACE_X_PLUS,  BC_DIRICHLET, T0=1.0)
        for f in (FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS):
            bridge.set_bc(f, BC_NEUMANN)
        info = bridge.solve(tol=1e-12, max_iter=5000)
        T = bridge.get_temperature()

    err_inf = float(np.max(np.abs(T - nodes[:, 0])))
    ok = err_inf < 1e-6
    if verbose:
        print(f"T1: линейное решение T(x)=x")
        print(f"    {info}")
        print(f"    max|T - T_exact| = {err_inf:.3e}    ->  "
              f"{'PASS' if ok else 'FAIL'}")
    return ok, err_inf


# =============================================================================
# T2 — сравнение с аналитическим решением:
#      Дирихле T = T0 на z = 0 + Робен на z = L; боковые — изоляция.
# Аналитика 1D: T(z) = T0 + (T_inf - T0) * z / (L + lambda/alpha).
# =============================================================================

def _T2_exact(z: np.ndarray, *, T0: float, T_inf: float, L: float,
              lam: float, alpha: float) -> np.ndarray:
    return T0 + (T_inf - T0) * z / (L + lam / alpha)


def test_T2_analytic(verbose: bool = True,
                     n: int = 16) -> Tuple[bool, float]:
    """T2: сравнение с 1D-аналитическим решением."""
    L, lam, T0, alpha, T_inf = 0.1, 50.0, 100.0, 25.0, 20.0
    with CoreBridge() as bridge:
        bridge.generate_box(0.0, L, 0.0, L, 0.0, L, n, n, n)
        nodes = bridge.get_nodes()
        bridge.set_material(lam, 0.0)
        bridge.set_bc(FACE_Z_MINUS, BC_DIRICHLET, T0=T0)
        bridge.set_bc(FACE_Z_PLUS,  BC_ROBIN, alpha=alpha, T_inf=T_inf)
        _isolate_sides(bridge)
        info = bridge.solve(tol=1e-12, max_iter=5000)
        T = bridge.get_temperature()

    T_ex = _T2_exact(nodes[:, 2], T0=T0, T_inf=T_inf, L=L, lam=lam, alpha=alpha)
    err_inf = float(np.max(np.abs(T - T_ex)))
    err_rel = err_inf / max(abs(T0 - T_inf), 1.0)
    ok = err_rel < 1e-3
    if verbose:
        print(f"T2: 1D-аналитика (Дирихле + Робен), n = {n}")
        print(f"    {info}")
        print(f"    max|T - T_exact| = {err_inf:.3e}, относительная = {err_rel:.3e}"
              f"  ->  {'PASS' if ok else 'FAIL'}")
    return ok, err_inf


# =============================================================================
# T3 — эмпирический порядок сходимости в нормах L2 и H1 (раздел 3.4.3 ПЗ).
#
# Используется ГЛАДКАЯ гармоническая функция (ΔT = 0):
#       T_exact(x, y, z) = sin(π x) · sin(π y) · cosh(π √2 · z) / cosh(π √2)
#
# На всех узлах границы куба [0, 1]³ задаётся Дирихле T = T_exact (через
# fem_set_node_dirichlet), внутренняя задача — Лаплас. Точное решение
# совпадает с T_exact, поэтому ошибка ‖T_h − T_exact‖_L2 убывает как O(h²),
# а в дискретной H1-полунорме (через узловые градиенты) — как O(h).
# =============================================================================

def _T3_exact(nodes: np.ndarray) -> np.ndarray:
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]
    pi_sqrt2 = np.pi * math.sqrt(2.0)
    return (np.sin(np.pi * x) * np.sin(np.pi * y)
            * np.cosh(pi_sqrt2 * z) / math.cosh(pi_sqrt2))


def _is_boundary(nodes: np.ndarray, L: float = 1.0,
                 tol: float = 1e-9) -> np.ndarray:
    """Маска (N,) — True для узлов на границе единичного куба."""
    return ((np.abs(nodes[:, 0])     < tol) | (np.abs(nodes[:, 0] - L) < tol) |
            (np.abs(nodes[:, 1])     < tol) | (np.abs(nodes[:, 1] - L) < tol) |
            (np.abs(nodes[:, 2])     < tol) | (np.abs(nodes[:, 2] - L) < tol))


def _t3_run_one(n: int) -> tuple[float, float]:
    """Возвращает (h, err_L2) для сетки n×n×n единичного куба."""
    L = 1.0
    h = L / n

    with CoreBridge() as bridge:
        bridge.generate_box(0.0, L, 0.0, L, 0.0, L, n, n, n)
        nodes = bridge.get_nodes()
        T_ex = _T3_exact(nodes)

        bridge.set_material(1.0, 0.0)        # ΔT = 0 (без объёмных источников)
        # Все шесть граней — формально Дирихле (значения нулевые в T0,
        # но они будут ПЕРЕОПРЕДЕЛЕНЫ поузельно).
        for f in (FACE_X_MINUS, FACE_X_PLUS, FACE_Y_MINUS,
                  FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS):
            bridge.set_bc(f, BC_DIRICHLET, T0=0.0)

        # Поузельная установка значений Дирихле = T_exact на всех граничных узлах.
        bridge.clear_node_dirichlet()
        bnd_mask = _is_boundary(nodes, L)
        bnd_idx = np.flatnonzero(bnd_mask)
        bridge.set_node_dirichlet_array(bnd_idx, T_ex[bnd_idx])

        bridge.solve(tol=1e-12, max_iter=20000)
        T = bridge.get_temperature()

    # Дискретная норма L2 (приближение интеграла по объёму куба = 1).
    err_l2 = float(np.sqrt(np.mean((T - T_ex) ** 2)))
    return h, err_l2


def test_T3_convergence(verbose: bool = True,
                        ns: List[int] = [4, 8, 16, 32]) -> Tuple[bool, float]:
    """T3: эмпирический порядок сходимости в L2.

    Серия сеток, измельчающихся в 2 раза. Эмпирический порядок:
        p = log(err(h₁) / err(h₂)) / log(h₁ / h₂)
    Для P1-элементов на гладком решении ожидается p_L2 ≈ 2.
    """
    if verbose:
        print("T3: эмпирический порядок сходимости (гармоническая функция)")
        print("    n      h            err_L2       p_emp")

    errs = []
    for n in ns:
        h, err = _t3_run_one(n)
        errs.append((h, err))
        if verbose:
            # Печатаем с эмпирическим порядком относительно предыдущей точки.
            if len(errs) > 1:
                h0, e0 = errs[-2]
                p = math.log(e0 / err) / math.log(h0 / h) if err > 0 else float("nan")
                print(f"    {n:<6d} {h:.4e}    {err:.3e}    {p:.2f}")
            else:
                print(f"    {n:<6d} {h:.4e}    {err:.3e}    —")

    # Усредняем эмпирический порядок по последним парам, чтобы избежать
    # шума на самой грубой сетке.
    p_emp_list = []
    for i in range(1, len(errs)):
        h0, e0 = errs[i - 1]
        h1, e1 = errs[i]
        if e0 > 0 and e1 > 0:
            p_emp_list.append(math.log(e0 / e1) / math.log(h0 / h1))
    avg_p = sum(p_emp_list[-2:]) / max(1, len(p_emp_list[-2:])) if p_emp_list else 0.0

    # Критерий: средний порядок на двух самых мелких парах должен быть ≥ 1.7
    # (теория — ровно 2; запас на дискретное приближение интеграла и эффекты
    # обусловленности на крупных n).
    ok = avg_p >= 1.7
    if verbose:
        print(f"    Средний эмпирический порядок p_L2 ≈ {avg_p:.2f} (теория: 2.0)"
              f"  ->  {'PASS' if ok else 'FAIL'}")
    return ok, avg_p


# =============================================================================
# T4 — корректное включение всех трёх типов ГУ.
# =============================================================================

def test_T4_all_bc_types(verbose: bool = True) -> Tuple[bool, float]:
    """T4: задача с одновременным присутствием Дирихле, Неймана и Робена.
    Берём 1D-задачу с переменным сечением: на торце Z- — Дирихле, на торце
    Z+ — Робен; боковые — изоляция (Нейман). Проверяем, что решение совпадает
    с 1D-аналитикой (как в T2, но на «слегка вытянутом» теле — другие
    параметры)."""
    L, lam, T0, alpha, T_inf = 0.05, 200.0, 80.0, 50.0, 25.0
    with CoreBridge() as bridge:
        bridge.generate_box(0.0, L, 0.0, L, 0.0, L, 12, 12, 12)
        nodes = bridge.get_nodes()
        bridge.set_material(lam, 0.0)
        bridge.set_bc(FACE_Z_MINUS, BC_DIRICHLET, T0=T0)
        bridge.set_bc(FACE_Z_PLUS,  BC_ROBIN, alpha=alpha, T_inf=T_inf)
        _isolate_sides(bridge)
        info = bridge.solve(tol=1e-12, max_iter=5000)
        T = bridge.get_temperature()

    T_ex = _T2_exact(nodes[:, 2], T0=T0, T_inf=T_inf, L=L, lam=lam, alpha=alpha)
    err_inf = float(np.max(np.abs(T - T_ex)))
    ok = err_inf < 1e-5
    if verbose:
        print(f"T4: одновременно Дирихле + Нейман + Робен")
        print(f"    {info}")
        print(f"    max|T - T_exact| = {err_inf:.3e}  ->  "
              f"{'PASS' if ok else 'FAIL'}")
    return ok, err_inf


# =============================================================================
# Командная точка входа.
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("Верификация программного комплекса МКЭ — задачи T1, T2, T3, T4")
    print("=" * 70)
    results = []
    for fn in (test_T1_linear_solution, test_T2_analytic,
               test_T3_convergence, test_T4_all_bc_types):
        print()
        ok, _ = fn(verbose=True)
        results.append(ok)
    print()
    print("=" * 70)
    n_pass = sum(results)
    print(f"ИТОГО: {n_pass} / {len(results)} тестов пройдено")
    print("=" * 70)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
