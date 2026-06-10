# -*- coding: utf-8 -*-
"""
Пример: алюминиевый швеллер, частично погружённый в воду.

Швеллер длиной 0.40 м стоит вертикально (длина вдоль Z). Нижние 0.05 м опущены
в кипящую воду — смоченный поясок (торец + нижние полоски стенок) фиксируется
при 100 °C (Дирихле). Открытая часть теряет тепло свободной конвекцией в воздух.
Это создаёт измеримый градиент по высоте, сопоставимый с показаниями датчиков
DS18B20 в физическом эксперименте.

Запуск:
    FEM_CORE_LIB=fem_core/fem_core.so python3 examples/example_immersion_uchannel.py
"""

import numpy as np

from fem3d import shapes
from fem3d.core_bridge import CoreBridge, BC_DIRICHLET, BC_ROBIN
from fem3d.problem import Problem, BoundaryCondition, Immersion

# --- Геометрия швеллера: длина вдоль Z = 0.40 м ----------------------------
LENGTH = 0.40          # полная длина детали, м
WATER_LEVEL = 0.05     # глубина погружения (линия воды от низа), м

nodes, tets, bnd, fids = shapes.make_u_channel(
    outer_width=0.08, outer_height=0.06, thickness=0.012,
    length=LENGTH, n_thickness=3, n_length=40)

problem = Problem(lambda_=237.0, Q=0.0)        # алюминий, λ = 237 Вт/(м·К)
problem.material_name = "Алюминий"
problem.external_nodes = nodes
problem.external_elements = tets
problem.external_bnd_nodes = bnd
problem.external_bnd_face_ids = fids

# Открытые поверхности: свободная конвекция в воздух (на всех гранях).
for fid in range(6):
    problem.bcs[fid] = BoundaryCondition(type=BC_ROBIN, alpha=8.0, T_inf=20.0)

# Погружение: нижний конец по Z в кипящей воде, Дирихле 100 °C на смоченном пояске.
problem.immersion = Immersion(
    enabled=True, axis=2, level=WATER_LEVEL, side=0,
    wetted_bc=BoundaryCondition(type=BC_DIRICHLET, T0=100.0),
)
# Альтернатива (некипящая ванна): Робин с h и T воды:
#   wetted_bc=BoundaryCondition(type=BC_ROBIN, alpha=600.0, T_inf=80.0)

bridge = CoreBridge()
problem.build_mesh_in_core(bridge)
info = problem.solve(bridge, tol=1e-9, max_iter=8000)

z = problem.nodes[:, 2]
T = problem.T
print(f"Сошлось за {info.iterations} итераций, невязка {info.residual:.1e}")
print(f"Смоченная грань id = {problem._immersion_wetted_id}")
print("Профиль температуры по высоте (среднее по сечению):")
for zc in np.linspace(0.0, LENGTH, 9):
    m = np.abs(z - zc) < (LENGTH / 80.0)
    if m.any():
        print(f"  z = {zc:5.3f} м :  T = {T[m].mean():6.2f} °C")
print(f"Диапазон: {T.min():.2f} … {T.max():.2f} °C")
