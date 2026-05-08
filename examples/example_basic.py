#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_basic.py
================

Минимальный пример использования fem3d из пользовательского кода:
параллелепипед 100×100×100 мм, алюминий, нагрев снизу 100 °C,
конвекция сверху (α = 25 Вт/(м²·К), T∞ = 20 °C), боковые грани изолированы.

Решает задачу, выводит сводку и сохраняет CSV.
"""

import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from fem3d import (BC_DIRICHLET, BC_NEUMANN, BC_ROBIN, BoundaryCondition,
                   BoxGeometry, CoreBridge, FACE_X_MINUS, FACE_X_PLUS,
                   FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS,
                   Problem)
from fem3d.postprocess import export_csv, export_report


def main() -> int:
    problem = Problem(
        geometry=BoxGeometry(Lx=0.10, Ly=0.10, Lz=0.10, nx=20, ny=20, nz=20),
        lambda_=237.0,                  # алюминий
        Q=0.0,
        bcs={
            FACE_Z_MINUS: BoundaryCondition(type=BC_DIRICHLET, T0=100.0),
            FACE_Z_PLUS:  BoundaryCondition(type=BC_ROBIN, alpha=25.0, T_inf=20.0),
            FACE_X_MINUS: BoundaryCondition(type=BC_NEUMANN),
            FACE_X_PLUS:  BoundaryCondition(type=BC_NEUMANN),
            FACE_Y_MINUS: BoundaryCondition(type=BC_NEUMANN),
            FACE_Y_PLUS:  BoundaryCondition(type=BC_NEUMANN),
        },
    )

    with CoreBridge() as bridge:
        problem.build_mesh_in_core(bridge)
        print(f"Сетка: {problem.nodes.shape[0]} узлов, "
              f"{problem.elements.shape[0]} тетраэдров")
        info = problem.solve(bridge, tol=1e-8, max_iter=5000)

    print(info)
    Tmin, Tmax = problem.temperature_range()
    print(f"Tmin = {Tmin:.2f} °C, Tmax = {Tmax:.2f} °C")

    out_csv = os.path.join(HERE, "example_basic_result.csv")
    out_txt = os.path.join(HERE, "example_basic_report.txt")
    export_csv(problem, out_csv)
    export_report(problem, out_txt)
    print(f"Сохранено: {out_csv}\nСохранено: {out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
