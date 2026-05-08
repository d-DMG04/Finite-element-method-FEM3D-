# -*- coding: utf-8 -*-
"""
fem3d.postprocess
=================

Экспорт результатов расчёта во внешние форматы:
  - VTU/VTK (через meshio, опционально)
  - CSV — узлы и температуры
  - текстовый отчёт о расчёте

Покрывает требования Ф6.1, Ф6.2, Ф6.3 ТЗ.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np

from .core_bridge import (
    BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN, FACE_NAMES, SolverInfo,
)
from .problem import Problem


# =============================================================================
# VTU (через meshio) — для ParaView / VisIt.
# =============================================================================

def export_vtu(problem: Problem, path: str) -> None:
    """Экспорт сетки и результатов в формат VTK Unstructured Grid (.vtu)."""
    if problem.T is None:
        raise RuntimeError("Нечего экспортировать: расчёт ещё не выполнен")
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError(
            "Экспорт .vtu требует библиотеку meshio. "
            "Установите её: pip install meshio"
        ) from exc

    point_data = {
        "T": problem.T,
    }
    if problem.flux is not None:
        point_data["heat_flux"] = problem.flux

    cells = [("tetra", problem.elements.astype(np.int32))]
    m = meshio.Mesh(
        points=problem.nodes,
        cells=cells,
        point_data=point_data,
    )
    m.write(path)


# =============================================================================
# CSV — узлы и температуры.
# =============================================================================

def export_csv(problem: Problem, path: str) -> None:
    """Экспорт таблицы (x, y, z, T, qx, qy, qz) в CSV."""
    if problem.T is None or problem.nodes is None:
        raise RuntimeError("Нечего экспортировать: расчёт ещё не выполнен")

    have_flux = problem.flux is not None
    n = problem.nodes.shape[0]

    with open(path, "w", encoding="utf-8") as fout:
        if have_flux:
            fout.write("x,y,z,T,qx,qy,qz\n")
        else:
            fout.write("x,y,z,T\n")
        for i in range(n):
            x, y, z = problem.nodes[i]
            T = problem.T[i]
            if have_flux:
                qx, qy, qz = problem.flux[i]
                fout.write(f"{x:.6e},{y:.6e},{z:.6e},{T:.6e},"
                           f"{qx:.6e},{qy:.6e},{qz:.6e}\n")
            else:
                fout.write(f"{x:.6e},{y:.6e},{z:.6e},{T:.6e}\n")


# =============================================================================
# Текстовый отчёт.
# =============================================================================

def _bc_describe(bc) -> str:
    if bc.type == BC_DIRICHLET:
        return f"Дирихле, T = {bc.T0:g} °C"
    if bc.type == BC_NEUMANN:
        return f"Нейман, q = {bc.q0:g} Вт/м²"
    if bc.type == BC_ROBIN:
        return f"Робен, α = {bc.alpha:g} Вт/(м²·К), T∞ = {bc.T_inf:g} °C"
    return "не задано"


def export_report(problem: Problem, path: str) -> None:
    """Сформировать человекочитаемый отчёт о расчёте."""
    if problem.T is None:
        raise RuntimeError("Нечего экспортировать: расчёт ещё не выполнен")

    g = problem.geometry
    Tmin, Tmax = problem.temperature_range()
    hs = problem.hot_spot()
    info = problem.info

    lines = []
    lines.append("=" * 70)
    lines.append("Отчёт о расчёте — программный комплекс МКЭ для теплопроводности")
    lines.append(f"Сформирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("ГЕОМЕТРИЯ")
    lines.append(f"  Тип: параллелепипед")
    lines.append(f"  Размеры:    Lx = {g.Lx:g} м, Ly = {g.Ly:g} м, Lz = {g.Lz:g} м")
    lines.append(f"  Сетка:      nx = {g.nx}, ny = {g.ny}, nz = {g.nz}")
    if problem.nodes is not None:
        lines.append(f"  Узлов:      {problem.nodes.shape[0]}")
    if problem.elements is not None:
        lines.append(f"  Элементов:  {problem.elements.shape[0]}")
    lines.append("")
    lines.append("МАТЕРИАЛ")
    lines.append(f"  λ = {problem.lambda_:g} Вт/(м·К)")
    lines.append(f"  Q = {problem.Q:g} Вт/м³")
    lines.append("")
    lines.append("ГРАНИЧНЫЕ УСЛОВИЯ")
    for face_id in range(6):
        bc = problem.bcs[face_id]
        lines.append(f"  {FACE_NAMES[face_id]}: {_bc_describe(bc)}")
    lines.append("")
    if info is not None:
        lines.append("РЕШАТЕЛЬ (CG с предобусловливателем Якоби)")
        lines.append(f"  Итераций:           {info.iterations}")
        lines.append(f"  Норма невязки:      {info.residual:.3e}")
        lines.append(f"  Сошёлся:            {'да' if info.converged else 'нет'}")
        lines.append(f"  Время решения, с:   {info.time_seconds:.3f}")
    lines.append("")
    lines.append("РЕЗУЛЬТАТЫ")
    lines.append(f"  Tmin = {Tmin:.4f} °C")
    lines.append(f"  Tmax = {Tmax:.4f} °C")
    if hs is not None:
        idx, x, y, z = hs
        lines.append(f"  Горячая точка: узел #{idx} в ({x:g}, {y:g}, {z:g})")
    lines.append("")
    lines.append("=" * 70)

    with open(path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines))


# =============================================================================
# Сечение по плоскости — для отображения внутренних температур (Ф5.2).
# =============================================================================

def slice_by_plane(problem: Problem, axis: str, value: float,
                   tol_rel: float = 0.02) -> Optional[tuple]:
    """
    Возвращает (xs, ys, Ts) — узлы, попавшие в окрестность плоскости axis = value.
    axis: 'x', 'y' или 'z'.
    tol_rel — толщина среза относительно габарита по этой оси.
    """
    if problem.nodes is None or problem.T is None:
        return None
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
    if axis_idx is None:
        raise ValueError("axis должен быть 'x', 'y' или 'z'")
    coords = problem.nodes[:, axis_idx]
    span = float(coords.max() - coords.min())
    tol = max(span * tol_rel, 1e-9)
    mask = np.abs(coords - value) < tol
    if not np.any(mask):
        return None
    other = [i for i in (0, 1, 2) if i != axis_idx]
    return (problem.nodes[mask, other[0]],
            problem.nodes[mask, other[1]],
            problem.T[mask])
