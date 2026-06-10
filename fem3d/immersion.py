# -*- coding: utf-8 -*-
"""
fem3d.immersion
===============

Частичное погружение детали в жидкость (раздел экспериментальной методики ПЗ).

Идея
----
Деталь стоит вертикально, нижний конец опущен в воду. Линия воды режет боковые
стенки, поэтому одной грани (id 0..5) нельзя назначить два разных ГУ. Решение —
перед расчётом пройтись по всем граничным фасеткам и у каждой, чей центроид ниже
уровня воды, **перебить face_id на id погружённого торца** (для вертикали с нижним
концом в воде это id = 4, «Z−»). Тогда весь смоченный поясок — реальное дно плюс
нижние полоски боковых стенок — становится одной «гранью», и на неё вешается одно
ГУ (Дирихле T_воды или Робин h+T_воды). Сухие части стенок сохраняют свои id и
своё воздушное ГУ.

Подход не требует перекомпиляции ядра: ядро по-прежнему работает с 6 гранями,
а вся логика погружения выполняется на стороне Python над массивом
boundary_face_ids перед вызовом load_mesh.

Соглашение face_id (см. shapes._classify_face_by_normal):
    0: X−   1: X+   2: Y−   3: Y+   4: Z− (низ)   5: Z+ (верх)
=> id торца = 2 * axis + side, где axis ∈ {0:x, 1:y, 2:z}, side ∈ {0:min, 1:max}.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Имена осей для GUI/отчётов.
AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}
SIDE_NAMES = {0: "минимальный конец", 1: "максимальный конец"}


def end_face_id(axis: int, side: int) -> int:
    """face_id торца, погружённого в воду.

    axis: 0=x, 1=y, 2=z;  side: 0=min (нижний конец), 1=max (верхний конец).
    """
    return 2 * int(axis) + int(side)


def split_boundary_by_waterline(
    nodes: np.ndarray,
    bnd_nodes: np.ndarray,
    bnd_face_ids: np.ndarray,
    axis: int,
    level: float,
    side: int = 0,
) -> Tuple[np.ndarray, int, int]:
    """Перебить face_id смоченных фасеток на id погружённого торца.

    Parameters
    ----------
    nodes        : (N, 3)  координаты узлов, м.
    bnd_nodes    : (Nf, 3) индексы узлов каждой граничной фасетки.
    bnd_face_ids : (Nf,)   исходные face_id (0..5).
    axis         : ось погружения (0=x, 1=y, 2=z).
    level        : координата линии воды вдоль axis, м.
    side         : 0 — в воде нижний конец (центроид < level);
                   1 — в воде верхний конец (центроид > level).

    Returns
    -------
    new_face_ids : (Nf,) int32  — копия с перебитыми смоченными фасетками.
    wetted_id    : int          — face_id, под которым теперь весь смоченный поясок.
    n_wetted     : int          — сколько фасеток оказалось в воде.
    """
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    bnd_nodes = np.ascontiguousarray(bnd_nodes, dtype=np.int64)
    centroid_coord = nodes[bnd_nodes][:, :, int(axis)].mean(axis=1)  # (Nf,)

    if int(side) == 0:
        wet = centroid_coord < float(level)
    else:
        wet = centroid_coord > float(level)

    wetted_id = end_face_id(axis, side)
    new_ids = np.array(bnd_face_ids, dtype=np.int32, copy=True)
    new_ids[wet] = wetted_id
    return new_ids, int(wetted_id), int(np.count_nonzero(wet))


# =============================================================================
# Структурированная тетраэдральная сетка параллелепипеда (для балки-коробки).
#
# Ядро строит box через generate_box на стороне C++, и его фасетки в Python не
# видны. Чтобы перебивать face_id при погружении, коробку нужно построить в
# Python. Используется 6-тетраэдральное разбиение каждой ячейки (Kuhn), surface
# и классификация по нормалям — через shapes._finalize.
# =============================================================================

# Шаблон 6 тетраэдров на ячейку (общая диагональ 0–6), индексы локальных вершин.
_HEX_TETS = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


def make_box_mesh(Lx: float, Ly: float, Lz: float,
                  nx: int, ny: int, nz: int):
    """Тетраэдральная сетка параллелепипеда [0,Lx]×[0,Ly]×[0,Lz].

    Возвращает (nodes, tets, bnd_nodes, bnd_face_ids) — тот же формат, что у
    генераторов shapes.*, пригодный для CoreBridge.load_mesh.
    """
    from fem3d.shapes import _finalize  # ленивый импорт: избегаем циклов

    nx, ny, nz = max(1, int(nx)), max(1, int(ny)), max(1, int(nz))
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    zs = np.linspace(0.0, Lz, nz + 1)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = (
                    nid(i,     j,     k),
                    nid(i + 1, j,     k),
                    nid(i + 1, j + 1, k),
                    nid(i,     j + 1, k),
                    nid(i,     j,     k + 1),
                    nid(i + 1, j,     k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i,     j + 1, k + 1),
                )
                for a, b, c, d in _HEX_TETS:
                    tets.append((v[a], v[b], v[c], v[d]))

    tets = np.asarray(tets, dtype=np.int32)
    return _finalize(nodes, tets)
