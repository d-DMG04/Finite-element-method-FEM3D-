#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
headless_demo.py
================

Запускает три сценария расчёта без графического интерфейса и сохраняет
результаты (PNG-картинки и VTU/CSV/отчёт) в каталог demo_output/. Полезно
для:
  - проверки полного пайплайна управляющий слой → ядро → постобработка;
  - быстрой демонстрации возможностей в окружении без PyQt5;
  - регрессионного тестирования.

Использует только NumPy и matplotlib.

Запуск:
    python headless_demo.py
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fem3d import (BC_DIRICHLET, BC_NEUMANN, BC_ROBIN, BoundaryCondition,
                   BoxGeometry, CoreBridge, FACE_X_MINUS, FACE_X_PLUS,
                   FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS,
                   PointSource, Problem, VolumeSource, VOLSRC_SPHERE,
                   template_bottom_heat_top_cool)
from fem3d.postprocess import export_csv, export_report, slice_by_plane


def _save_slice_image(problem: Problem, axis: str, axis_pos: float,
                      outpath: str, title: str) -> None:
    """Сохраняет PNG-картинку температурного поля на сечении по выбранной оси."""
    res = slice_by_plane(problem, axis, axis_pos, tol_rel=0.04)
    if res is None:
        print(f"  [предупреждение] на оси {axis} = {axis_pos:.3f} нет узлов")
        return
    xs, ys, Ts = res
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(xs, ys, c=Ts, cmap="inferno", s=24)
    plt.colorbar(sc, ax=ax, label="T, °C")
    other = [a for a in "xyz" if a != axis]
    ax.set_xlabel(other[0])
    ax.set_ylabel(other[1])
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def _save_3d_scatter(problem: Problem, outpath: str, title: str) -> None:
    """Сохраняет PNG: разреженный 3D-облако точек узлов с цветом по T."""
    nodes = problem.nodes
    T = problem.T
    # Упростим: берём подвыборку узлов на поверхности для скорости.
    if nodes is None or T is None:
        return
    # Сэмплируем не более 4000 узлов.
    n = nodes.shape[0]
    if n > 4000:
        idx = np.random.choice(n, size=4000, replace=False)
    else:
        idx = np.arange(n)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(nodes[idx, 0], nodes[idx, 1], nodes[idx, 2],
                    c=T[idx], cmap="inferno", s=4)
    plt.colorbar(sc, ax=ax, label="T, °C")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def _run_scenario(name: str, problem: Problem, outdir: str) -> None:
    """Запуск одного сценария + сохранение всех артефактов."""
    print(f"\n>> Сценарий: {name}")
    with CoreBridge() as bridge:
        problem.build_mesh_in_core(bridge)
        info = problem.solve(bridge, tol=1e-8, max_iter=5000)
    Tmin, Tmax = problem.temperature_range()
    print(f"   Узлов: {problem.nodes.shape[0]}, элементов: {problem.elements.shape[0]}")
    print(f"   {info}")
    print(f"   Tmin = {Tmin:.2f} °C, Tmax = {Tmax:.2f} °C")
    base = os.path.join(outdir, name)
    _save_slice_image(problem, "z", problem.geometry.Lz / 2,
                      base + "_slice_z.png",
                      f"{name}: сечение z = Lz/2")
    _save_slice_image(problem, "x", problem.geometry.Lx / 2,
                      base + "_slice_x.png",
                      f"{name}: сечение x = Lx/2")
    _save_3d_scatter(problem, base + "_3d.png", f"{name}: T(x,y,z)")
    export_csv(problem, base + ".csv")
    export_report(problem, base + ".txt")
    print(f"   Артефакты сохранены: {base}_*.png, {base}.csv, {base}.txt")


def main() -> int:
    outdir = os.path.join(HERE, "demo_output")
    os.makedirs(outdir, exist_ok=True)
    print(f"Каталог результатов: {outdir}")

    # --- Сценарий 1: Нагрев снизу + конвекция сверху, алюминий --------------
    p1 = Problem(
        geometry=BoxGeometry(Lx=0.10, Ly=0.10, Lz=0.10, nx=20, ny=20, nz=20),
        lambda_=237.0, Q=0.0,
        bcs=template_bottom_heat_top_cool(),
    )
    _run_scenario("01_aluminium_box_heat_below", p1, outdir)

    # --- Сценарий 2: тонкая стальная пластина, объёмный источник -----------
    p2 = Problem(
        geometry=BoxGeometry(Lx=0.20, Ly=0.20, Lz=0.01, nx=24, ny=24, nz=4),
        lambda_=55.0, Q=2.0e6,
        bcs={
            FACE_Z_MINUS: BoundaryCondition(type=BC_ROBIN, alpha=15.0, T_inf=20.0),
            FACE_Z_PLUS:  BoundaryCondition(type=BC_ROBIN, alpha=15.0, T_inf=20.0),
            FACE_X_MINUS: BoundaryCondition(type=BC_NEUMANN),
            FACE_X_PLUS:  BoundaryCondition(type=BC_NEUMANN),
            FACE_Y_MINUS: BoundaryCondition(type=BC_NEUMANN),
            FACE_Y_PLUS:  BoundaryCondition(type=BC_NEUMANN),
        },
    )
    _run_scenario("02_steel_plate_volume_source", p2, outdir)

    # --- Сценарий 3: тепловой мост между двумя температурами ---------------
    p3 = Problem(
        geometry=BoxGeometry(Lx=0.05, Ly=0.05, Lz=0.20, nx=10, ny=10, nz=30),
        lambda_=401.0, Q=0.0,    # медь
        bcs={
            FACE_Z_MINUS: BoundaryCondition(type=BC_DIRICHLET, T0=120.0),
            FACE_Z_PLUS:  BoundaryCondition(type=BC_DIRICHLET, T0=20.0),
            FACE_X_MINUS: BoundaryCondition(type=BC_ROBIN, alpha=8.0, T_inf=20.0),
            FACE_X_PLUS:  BoundaryCondition(type=BC_ROBIN, alpha=8.0, T_inf=20.0),
            FACE_Y_MINUS: BoundaryCondition(type=BC_ROBIN, alpha=8.0, T_inf=20.0),
            FACE_Y_PLUS:  BoundaryCondition(type=BC_ROBIN, alpha=8.0, T_inf=20.0),
        },
    )
    _run_scenario("03_copper_bar_thermal_bridge", p3, outdir)

    # --- Сценарий 4: радиатор с локальными источниками тепла --------------
    # Алюминиевая «подложка» с двумя источниками: точечный (электронный
    # компонент) и объёмный шар (область высокого тепловыделения).
    p4 = Problem(
        geometry=BoxGeometry(Lx=0.10, Ly=0.10, Lz=0.02, nx=24, ny=24, nz=6),
        lambda_=237.0, Q=0.0,
        bcs={
            FACE_Z_MINUS: BoundaryCondition(type=BC_ROBIN, alpha=20.0, T_inf=20.0),
            FACE_Z_PLUS:  BoundaryCondition(type=BC_ROBIN, alpha=80.0, T_inf=20.0),
            FACE_X_MINUS: BoundaryCondition(type=BC_NEUMANN),
            FACE_X_PLUS:  BoundaryCondition(type=BC_NEUMANN),
            FACE_Y_MINUS: BoundaryCondition(type=BC_NEUMANN),
            FACE_Y_PLUS:  BoundaryCondition(type=BC_NEUMANN),
        },
    )
    # Сначала строим сетку, чтобы знать координаты узлов и привязать
    # точечный источник к ближайшему узлу.
    with CoreBridge() as br:
        p4.build_mesh_in_core(br)
    # Точка (0.03, 0.03, 0.01) — компонент мощностью 30 Вт.
    diff = p4.nodes - np.array([0.03, 0.03, 0.01])
    idx = int(np.argmin(np.sum(diff * diff, axis=1)))
    p4.point_sources.append(PointSource(node_idx=idx, power=30.0))
    # Объёмный шар r=8 мм с центром (0.07, 0.07, 0.01) и Q = 5e7 Вт/м³.
    p4.volume_sources.append(VolumeSource(
        shape=VOLSRC_SPHERE, params=(0.07, 0.07, 0.01, 0.008), Q0=5.0e7))
    _run_scenario("04_radiator_with_sources", p4, outdir)

    print("\nВсе три сценария рассчитаны и сохранены в demo_output/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
