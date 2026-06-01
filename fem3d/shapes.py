# -*- coding: utf-8 -*-
"""
fem3d.shapes — генераторы сложных трёхмерных геометрий.

Все функции возвращают четвёрку (nodes, tets, bnd_nodes, bnd_face_ids)
в том же формате, что и core_bridge.CoreBridge.load_mesh.

Сетки строятся структурированно (декомпозиция Куна для каждого
гексаэдрального элемента → 6 тетраэдров) — это даёт хорошее качество
и предсказуемое число элементов без необходимости в gmsh.

Геометрии:
    make_cylinder        — сплошной цилиндр;
    make_hollow_cylinder — труба;
    make_torus           — тор (бублик);
    make_pyramid         — пирамида (4-гранная);
    make_l_beam          — L-образный профиль;
    make_pcb             — модель платы (тонкая пластина + чипы);
    make_fin_heatsink    — радиатор с рёбрами;
    make_plate_with_hole — пластина с прямоугольным отверстием.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# =============================================================================
# Утилиты низкого уровня.
# =============================================================================

# Декомпозиция Куна: каждый кубический ячейка из 8 вершин — на 6 тетраэдров.
_KUHN_TETS = np.array([
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
    [0, 5, 1, 6],
], dtype=np.int32)


def _hexes_to_tets(hex_corner_ids: np.ndarray) -> np.ndarray:
    """Преобразует массив (Nh, 8) гексаэдров (с локальной нумерацией VTK_HEXAHEDRON)
    в массив (Nh*6, 4) тетраэдров через декомпозицию Куна.

    VTK-порядок вершин гекса:
        0: (0,0,0)  1: (1,0,0)  2: (1,1,0)  3: (0,1,0)
        4: (0,0,1)  5: (1,0,1)  6: (1,1,1)  7: (0,1,1)
    """
    nh = hex_corner_ids.shape[0]
    tets = np.empty((nh * 6, 4), dtype=np.int32)
    for i, kt in enumerate(_KUHN_TETS):
        tets[i::6, :] = hex_corner_ids[:, kt]
    return tets


def _extract_surface(tets: np.ndarray) -> np.ndarray:
    """Извлекает поверхностные треугольники тетраэдральной сетки —
    грани, встречающиеся ровно один раз."""
    local_faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
                           dtype=np.int32)
    Ne = tets.shape[0]
    all_faces = np.empty((4 * Ne, 3), dtype=np.int64)
    for k in range(4):
        all_faces[k * Ne:(k + 1) * Ne, :] = tets[:, local_faces[k]]
    sorted_faces = np.sort(all_faces, axis=1)
    keys = sorted_faces.view([("a", np.int64), ("b", np.int64),
                              ("c", np.int64)]).ravel()
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    surf_indices = np.flatnonzero(counts == 1)
    first_occ = np.full(uniq.size, -1, dtype=np.int64)
    for i, idx in enumerate(inv):
        if first_occ[idx] < 0:
            first_occ[idx] = i
    surface_face_orig_idx = first_occ[surf_indices]
    return all_faces[surface_face_orig_idx, :].astype(np.int32)


def _orient_tets_positive(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Гарантирует положительный объём всех тетраэдров (меняет местами две
    вершины при необходимости)."""
    p0 = nodes[tets[:, 0]]
    p1 = nodes[tets[:, 1]]
    p2 = nodes[tets[:, 2]]
    p3 = nodes[tets[:, 3]]
    v01 = p1 - p0; v02 = p2 - p0; v03 = p3 - p0
    vol6 = np.einsum("ij,ij->i", v01, np.cross(v02, v03))
    negative = vol6 < 0.0
    if np.any(negative):
        out = tets.copy()
        out[negative, 1], out[negative, 2] = tets[negative, 2], tets[negative, 1]
        return out
    return tets


def _classify_face_by_normal(nodes: np.ndarray,
                              bnd_faces: np.ndarray) -> np.ndarray:
    """Назначить каждой треугольной грани face_id из 0..5 по доминирующей
    компоненте средней нормали.

    Соглашение face_id:
        0: X-   (нормаль направлена против оси X)
        1: X+   (вдоль +X)
        2: Y-   (против Y)
        3: Y+   (вдоль +Y)
        4: Z-   (против Z, "низ")
        5: Z+   (вдоль Z, "верх")

    Это позволяет применить ГУ к любой поверхности, даже если она кривая:
    для цилиндра/сферы пользователь может задать «Z+ верх», «Z- низ» (плоские
    основания) и одно из X±/Y± — оно покроет всю боковую поверхность.

    Для упрощения интерфейса в случае сложных фигур мы ставим face_id так,
    чтобы пользователь мог использовать все 6 граней в диалоге BC.
    """
    n0 = nodes[bnd_faces[:, 0]]
    n1 = nodes[bnd_faces[:, 1]]
    n2 = nodes[bnd_faces[:, 2]]
    # Внешняя нормаль направлена от центроида тела наружу.
    body_center = nodes.mean(axis=0)
    normals = np.cross(n1 - n0, n2 - n0)
    norm_len = np.linalg.norm(normals, axis=1)
    norm_len[norm_len < 1e-30] = 1.0
    normals = normals / norm_len[:, None]
    centroids = (n0 + n1 + n2) / 3.0
    outward = np.einsum("ij,ij->i", normals, centroids - body_center) > 0
    normals[~outward] = -normals[~outward]
    # Классификация по доминирующей компоненте.
    ax = np.argmax(np.abs(normals), axis=1)  # 0=x, 1=y, 2=z
    sign = np.sign(normals[np.arange(len(normals)), ax])
    # face_id: 0=X-, 1=X+, 2=Y-, 3=Y+, 4=Z-, 5=Z+
    face_ids = np.zeros(len(bnd_faces), dtype=np.int32)
    for i in range(len(bnd_faces)):
        if ax[i] == 0:
            face_ids[i] = 1 if sign[i] > 0 else 0
        elif ax[i] == 1:
            face_ids[i] = 3 if sign[i] > 0 else 2
        else:
            face_ids[i] = 5 if sign[i] > 0 else 4
    return face_ids


def _finalize(nodes: np.ndarray, tets: np.ndarray,
              face_id_default: int = 0) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Завершающие операции: дедупликация узлов, отбраковка вырожденных
    тетраэдров, ориентация + извлечение поверхности + классификация
    по нормалям."""
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    tets = np.ascontiguousarray(tets, dtype=np.int32)

    # 1. Дедупликация совпадающих узлов (например, вершина пирамиды).
    # Округляем до 1e-10 для надёжного сравнения чисел с плавающей точкой.
    rounded = np.round(nodes, decimals=10)
    _, unique_idx, inverse = np.unique(rounded, axis=0, return_index=True,
                                         return_inverse=True)
    if unique_idx.size < nodes.shape[0]:
        nodes = nodes[unique_idx]
        tets = inverse[tets].astype(np.int32)

    # 2. Отбраковка вырожденных тетраэдров (объём ниже порога).
    a = nodes[tets[:, 0]]; b = nodes[tets[:, 1]]
    c = nodes[tets[:, 2]]; d = nodes[tets[:, 3]]
    vol6 = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))
    bbox = nodes.max(axis=0) - nodes.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    # Порог: тет с объёмом меньше 1e-14 · (диагональ bbox)^3 считается вырожденным.
    eps_vol = 1e-14 * diag * diag * diag
    keep = np.abs(vol6) / 6.0 > eps_vol
    if not keep.all():
        tets = tets[keep]

    # 3. Также убрать тетры с повторяющимися индексами узлов.
    deduped = (tets[:, 0] != tets[:, 1]) & (tets[:, 0] != tets[:, 2]) & \
              (tets[:, 0] != tets[:, 3]) & (tets[:, 1] != tets[:, 2]) & \
              (tets[:, 1] != tets[:, 3]) & (tets[:, 2] != tets[:, 3])
    if not deduped.all():
        tets = tets[deduped]

    # 4. Ориентация тетраэдров (положительный объём).
    tets = _orient_tets_positive(nodes, tets)

    # 5. Поверхностные треугольники + face_id по нормалям.
    bnd_nodes = _extract_surface(tets)
    bnd_face_ids = _classify_face_by_normal(nodes, bnd_nodes)
    return nodes, tets, bnd_nodes, bnd_face_ids


# =============================================================================
# Цилиндр.
# =============================================================================

def make_cylinder(radius: float = 0.05, height: float = 0.10,
                  n_radial: int = 8, n_angular: int = 24,
                  n_axial: int = 12) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """Сплошной цилиндр.

    Ось цилиндра — Z. Центр основания — в точке (0, 0, 0), верх — z = height.

    Сетка строится по полярной координатной системе (n_angular секторов,
    n_radial колец), плюс центральная вершина — итого структурированная
    «пирожок» в плоскости xy.
    """
    if n_angular < 3 or n_radial < 1 or n_axial < 1:
        raise ValueError("n_angular >= 3, n_radial >= 1, n_axial >= 1")

    # Шаг по углу и радиусу.
    angles = np.linspace(0.0, 2.0 * np.pi, n_angular, endpoint=False)
    radii = np.linspace(0.0, radius, n_radial + 1)
    z_levels = np.linspace(0.0, height, n_axial + 1)

    # Узлы: на каждом z-уровне — центральная + (n_radial)*(n_angular) колец.
    # Точки центральной оси индексируются отдельно (по одной на уровень).
    # Точки на радиальном ринге: для r > 0 берём (n_angular) точек на круге.
    nodes_per_level = 1 + n_radial * n_angular

    coords = np.empty((nodes_per_level * (n_axial + 1), 3), dtype=np.float64)
    for kz, z in enumerate(z_levels):
        base = kz * nodes_per_level
        coords[base] = (0.0, 0.0, z)
        for ir in range(1, n_radial + 1):
            r = radii[ir]
            for ia in range(n_angular):
                a = angles[ia]
                idx = base + 1 + (ir - 1) * n_angular + ia
                coords[idx] = (r * np.cos(a), r * np.sin(a), z)

    def nid(kz, ir, ia):
        """Индекс узла на уровне kz, радиальной позиции ir (0=центр), угле ia."""
        base = kz * nodes_per_level
        if ir == 0:
            return base
        return base + 1 + (ir - 1) * n_angular + (ia % n_angular)

    # Гексаэдральная сетка: между уровнями kz и kz+1, между ir и ir+1,
    # между ia и ia+1. Для ir == 0 — вырожденный гекс (фактически 5-узловой),
    # который мы превращаем в 3 тетраэдра вручную.
    tets_list = []

    for kz in range(n_axial):
        # Центральная клиновидная зона (ir = 0 → 1): 6 тетраэдров на сектор
        # через триангуляцию вырожденного гекса с двумя совпадающими рёбрами.
        for ia in range(n_angular):
            ia2 = (ia + 1) % n_angular
            # 6 углов клина: нижняя центральная, нижние два внешних,
            # верхняя центральная, верхние два внешних.
            c0 = nid(kz,     0,  0)
            c1 = nid(kz,     1,  ia)
            c2 = nid(kz,     1,  ia2)
            c3 = nid(kz + 1, 0,  0)
            c4 = nid(kz + 1, 1,  ia)
            c5 = nid(kz + 1, 1,  ia2)
            # Призма (c0,c1,c2,c3,c4,c5) → 3 тетраэдра.
            tets_list.append((c0, c1, c2, c5))
            tets_list.append((c0, c1, c5, c4))
            tets_list.append((c0, c4, c5, c3))

        # Кольцевая зона (ir >= 1): обычные гексы.
        for ir in range(1, n_radial):
            for ia in range(n_angular):
                ia2 = (ia + 1) % n_angular
                v = [
                    nid(kz,     ir,     ia),
                    nid(kz,     ir + 1, ia),
                    nid(kz,     ir + 1, ia2),
                    nid(kz,     ir,     ia2),
                    nid(kz + 1, ir,     ia),
                    nid(kz + 1, ir + 1, ia),
                    nid(kz + 1, ir + 1, ia2),
                    nid(kz + 1, ir,     ia2),
                ]
                # Декомпозиция Куна на 6 тетраэдров.
                for kt in _KUHN_TETS:
                    tets_list.append((v[kt[0]], v[kt[1]], v[kt[2]], v[kt[3]]))

    tets = np.asarray(tets_list, dtype=np.int32)
    return _finalize(coords, tets)


# =============================================================================
# Полый цилиндр (труба).
# =============================================================================

def make_hollow_cylinder(r_inner: float = 0.03, r_outer: float = 0.05,
                          height: float = 0.10,
                          n_radial: int = 4, n_angular: int = 24,
                          n_axial: int = 12) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """Полый цилиндр (труба).
    Аналог make_cylinder, но без центральной оси: только кольцо."""
    if r_inner <= 0.0 or r_outer <= r_inner:
        raise ValueError("Должно быть 0 < r_inner < r_outer")
    if n_angular < 3 or n_radial < 1 or n_axial < 1:
        raise ValueError("n_angular >= 3, n_radial >= 1, n_axial >= 1")

    angles = np.linspace(0.0, 2.0 * np.pi, n_angular, endpoint=False)
    radii = np.linspace(r_inner, r_outer, n_radial + 1)
    z_levels = np.linspace(0.0, height, n_axial + 1)

    nodes_per_level = (n_radial + 1) * n_angular
    coords = np.empty((nodes_per_level * (n_axial + 1), 3), dtype=np.float64)
    for kz, z in enumerate(z_levels):
        for ir in range(n_radial + 1):
            r = radii[ir]
            for ia in range(n_angular):
                a = angles[ia]
                idx = (kz * nodes_per_level + ir * n_angular + ia)
                coords[idx] = (r * np.cos(a), r * np.sin(a), z)

    def nid(kz, ir, ia):
        return kz * nodes_per_level + ir * n_angular + (ia % n_angular)

    tets_list = []
    for kz in range(n_axial):
        for ir in range(n_radial):
            for ia in range(n_angular):
                ia2 = (ia + 1) % n_angular
                v = [
                    nid(kz,     ir,     ia),
                    nid(kz,     ir + 1, ia),
                    nid(kz,     ir + 1, ia2),
                    nid(kz,     ir,     ia2),
                    nid(kz + 1, ir,     ia),
                    nid(kz + 1, ir + 1, ia),
                    nid(kz + 1, ir + 1, ia2),
                    nid(kz + 1, ir,     ia2),
                ]
                for kt in _KUHN_TETS:
                    tets_list.append((v[kt[0]], v[kt[1]], v[kt[2]], v[kt[3]]))

    tets = np.asarray(tets_list, dtype=np.int32)
    return _finalize(coords, tets)


# =============================================================================
# Тор (бублик).
# =============================================================================

def make_torus(R_major: float = 0.05, r_minor: float = 0.015,
                n_major: int = 32, n_minor: int = 12,
                n_radial: int = 3) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    """Сплошной тор (бублик).

    R_major — большой радиус (от центра тора до центра трубки);
    r_minor — малый радиус (радиус трубки);
    n_major — число секций вокруг главной оси (Z);
    n_minor — число секций вокруг малой оси;
    n_radial — число радиальных слоёв в трубке.

    Тор центрирован в (0, 0, 0), главная ось — Z.
    """
    if R_major <= r_minor:
        raise ValueError("R_major должен быть больше r_minor")

    phi = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)  # вокруг Z
    theta = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)  # в сечении
    rho = np.linspace(0.0, r_minor, n_radial + 1)

    # Структура узлов: для каждой пары (i_phi, i_theta) — n_radial+1 узлов
    # вдоль радиуса трубки. Центр трубки (rho=0) — отдельный узел на каждой
    # i_phi, который разделяется всеми i_theta.
    nodes_per_section = 1 + n_radial * n_minor
    n_total = nodes_per_section * n_major
    coords = np.empty((n_total, 3), dtype=np.float64)

    for ip in range(n_major):
        p = phi[ip]
        cosp, sinp = np.cos(p), np.sin(p)
        base = ip * nodes_per_section
        # Центральная точка трубки на этом сечении (rho = 0).
        coords[base] = (R_major * cosp, R_major * sinp, 0.0)
        for ir in range(1, n_radial + 1):
            r = rho[ir]
            for it in range(n_minor):
                t = theta[it]
                # Координаты в локальной системе сечения: в плоскости
                # (радиальная_к_оси, Z), радиус r, угол t.
                dr = r * np.cos(t)
                dz = r * np.sin(t)
                R = R_major + dr
                idx = base + 1 + (ir - 1) * n_minor + it
                coords[idx] = (R * cosp, R * sinp, dz)

    def nid(ip, ir, it):
        ip = ip % n_major
        base = ip * nodes_per_section
        if ir == 0:
            return base
        return base + 1 + (ir - 1) * n_minor + (it % n_minor)

    tets_list = []
    for ip in range(n_major):
        ip2 = (ip + 1) % n_major
        # Между сечениями ip и ip2 — кольцевые элементы.
        # Центральный «клин» (ir = 0 → 1) для каждого угла it.
        for it in range(n_minor):
            it2 = (it + 1) % n_minor
            # Призма: 6 углов.
            c0 = nid(ip,  0,  0)
            c1 = nid(ip,  1,  it)
            c2 = nid(ip,  1,  it2)
            c3 = nid(ip2, 0,  0)
            c4 = nid(ip2, 1,  it)
            c5 = nid(ip2, 1,  it2)
            tets_list.append((c0, c1, c2, c5))
            tets_list.append((c0, c1, c5, c4))
            tets_list.append((c0, c4, c5, c3))

        # Кольцевые гексы (ir >= 1).
        for ir in range(1, n_radial):
            for it in range(n_minor):
                it2 = (it + 1) % n_minor
                v = [
                    nid(ip,  ir,     it),
                    nid(ip,  ir + 1, it),
                    nid(ip,  ir + 1, it2),
                    nid(ip,  ir,     it2),
                    nid(ip2, ir,     it),
                    nid(ip2, ir + 1, it),
                    nid(ip2, ir + 1, it2),
                    nid(ip2, ir,     it2),
                ]
                for kt in _KUHN_TETS:
                    tets_list.append((v[kt[0]], v[kt[1]], v[kt[2]], v[kt[3]]))

    tets = np.asarray(tets_list, dtype=np.int32)
    return _finalize(coords, tets)


# =============================================================================
# Пирамида.
# =============================================================================

def make_pyramid(base_side: float = 0.10, height: float = 0.08,
                  n_base: int = 16,
                  n_axial: int = 12) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """Четырёхгранная пирамида с квадратным основанием.

    Основание — квадрат со стороной base_side в плоскости z=0, центр в (0,0,0).
    Вершина — в (0, 0, height).

    Сетка строится как стопка квадратных пластин, у которых сторона
    линейно уменьшается с z.
    """
    z_levels = np.linspace(0.0, height, n_axial + 1)
    # На каждом z — (n_base+1) × (n_base+1) точек, но размер меняется.
    nodes_per_level = (n_base + 1) ** 2

    coords = np.empty((nodes_per_level * (n_axial + 1), 3), dtype=np.float64)
    for kz, z in enumerate(z_levels):
        # Линейное масштабирование основания: при z=0 — base_side, при
        # z=height — 0. Чтобы избежать вырожденности на самой вершине,
        # сохраняем минимальный размер.
        scale = 1.0 - z / height
        side = base_side * max(scale, 1e-6)
        if scale < 1e-6:
            # Все точки на этом уровне совпадают — но мы их всё равно
            # сохраняем (вершина).
            for j in range(n_base + 1):
                for i in range(n_base + 1):
                    idx = kz * nodes_per_level + j * (n_base + 1) + i
                    coords[idx] = (0.0, 0.0, z)
        else:
            xs = np.linspace(-side / 2, side / 2, n_base + 1)
            ys = np.linspace(-side / 2, side / 2, n_base + 1)
            for j in range(n_base + 1):
                for i in range(n_base + 1):
                    idx = kz * nodes_per_level + j * (n_base + 1) + i
                    coords[idx] = (xs[i], ys[j], z)

    def nid(kz, i, j):
        return kz * nodes_per_level + j * (n_base + 1) + i

    tets_list = []
    # Последний уровень — это вершина (все узлы совпадают), поэтому
    # элементы строим только для kz = 0..n_axial - 1.
    for kz in range(n_axial):
        for j in range(n_base):
            for i in range(n_base):
                v = [
                    nid(kz,     i,     j),
                    nid(kz,     i + 1, j),
                    nid(kz,     i + 1, j + 1),
                    nid(kz,     i,     j + 1),
                    nid(kz + 1, i,     j),
                    nid(kz + 1, i + 1, j),
                    nid(kz + 1, i + 1, j + 1),
                    nid(kz + 1, i,     j + 1),
                ]
                for kt in _KUHN_TETS:
                    tets_list.append((v[kt[0]], v[kt[1]], v[kt[2]], v[kt[3]]))

    tets = np.asarray(tets_list, dtype=np.int32)
    nodes, tets, bnd_nodes, bnd_face_ids = _finalize(coords, tets)
    return nodes, tets, bnd_nodes, bnd_face_ids


# =============================================================================
# L-образный профиль.
# =============================================================================

def make_l_beam(arm_length: float = 0.10, thickness: float = 0.02,
                depth: float = 0.05,
                n_arm: int = 16, n_thick: int = 4,
                n_depth: int = 8) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """L-образная балка.

    Два прямоугольных «плеча» сложены в L-форме. Длина каждого плеча
    arm_length, толщина thickness, глубина (по Y) — depth.
    Внутренний угол в (0, 0, 0). Плечи лежат вдоль +X и +Z.
    """
    # Стратегия: построим L-фигуру в плоскости (X, Z) как объединение
    # двух прямоугольников, рассечь её на структурированную сетку.
    # Сетка строится так: проходим все ячейки регулярной сетки в bbox
    # и оставляем только те, что попадают в L-форму.
    bbox_xmax = arm_length + thickness
    bbox_zmax = arm_length + thickness

    nx_total = int(round(bbox_xmax / arm_length * n_arm))
    nz_total = int(round(bbox_zmax / arm_length * n_arm))

    xs = np.linspace(0.0, bbox_xmax, nx_total + 1)
    zs = np.linspace(0.0, bbox_zmax, nz_total + 1)
    ys = np.linspace(0.0, depth, n_depth + 1)

    # Узлы — все точки решётки, индексация (i, j, k) → линейный.
    nx, ny, nz = len(xs), len(ys), len(zs)
    coords = np.empty((nx * ny * nz, 3), dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                coords[k * nx * ny + j * nx + i] = (xs[i], ys[j], zs[k])

    def nid(i, j, k):
        return k * nx * ny + j * nx + i

    # L-форма: точка (x, z) в L, если (x < thickness) или (z < thickness).
    def in_L(x, z):
        return (x <= thickness + 1e-12) or (z <= thickness + 1e-12)

    tets_list = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Центр ячейки.
                cx = 0.5 * (xs[i] + xs[i + 1])
                cz = 0.5 * (zs[k] + zs[k + 1])
                if not in_L(cx, cz):
                    continue
                v = [
                    nid(i,     j,     k),
                    nid(i + 1, j,     k),
                    nid(i + 1, j + 1, k),
                    nid(i,     j + 1, k),
                    nid(i,     j,     k + 1),
                    nid(i + 1, j,     k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i,     j + 1, k + 1),
                ]
                for kt in _KUHN_TETS:
                    tets_list.append((v[kt[0]], v[kt[1]], v[kt[2]], v[kt[3]]))

    tets = np.asarray(tets_list, dtype=np.int32)
    # Узлы, не использованные ни одним тетраэдром, оставляем — это нестрашно
    # для решателя, но непрактично. Чистим их.
    nodes, tets, bnd_nodes, bnd_face_ids = _compact_unused_nodes(coords, tets)
    return nodes, tets, bnd_nodes, bnd_face_ids


# =============================================================================
# Плата с прямоугольным отверстием.
# =============================================================================

def make_plate_with_hole(Lx: float = 0.20, Ly: float = 0.10,
                          thickness: float = 0.005,
                          hole_x: float = 0.06, hole_y: float = 0.04,
                          hole_w: float = 0.04, hole_h: float = 0.02,
                          nx: int = 28, ny: int = 14, nz: int = 3
                          ) -> Tuple[np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    """Прямоугольная плата с отверстием.

    Размеры пластины (Lx, Ly), толщина по Z = thickness.
    Прямоугольное отверстие задано параметрами:
        hole_x, hole_y — нижний-левый угол;
        hole_w, hole_h — ширина и высота.
    """
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    zs = np.linspace(0.0, thickness, nz + 1)

    coords = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=np.float64)
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                idx = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
                coords[idx] = (xs[i], ys[j], zs[k])

    def nid(i, j, k):
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i

    def in_hole(x, y):
        return (hole_x <= x <= hole_x + hole_w) and (hole_y <= y <= hole_y + hole_h)

    tets_list = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cx = 0.5 * (xs[i] + xs[i + 1])
                cy = 0.5 * (ys[j] + ys[j + 1])
                if in_hole(cx, cy):
                    continue
                v = [
                    nid(i,     j,     k),
                    nid(i + 1, j,     k),
                    nid(i + 1, j + 1, k),
                    nid(i,     j + 1, k),
                    nid(i,     j,     k + 1),
                    nid(i + 1, j,     k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i,     j + 1, k + 1),
                ]
                for kt in _KUHN_TETS:
                    tets_list.append((v[kt[0]], v[kt[1]], v[kt[2]], v[kt[3]]))

    tets = np.asarray(tets_list, dtype=np.int32)
    return _compact_unused_nodes(coords, tets)


# =============================================================================
# Радиатор с рёбрами.
# =============================================================================

def make_fin_heatsink(base_Lx: float = 0.10, base_Ly: float = 0.06,
                       base_thickness: float = 0.005,
                       fin_height: float = 0.025, fin_thickness: float = 0.003,
                       n_fins: int = 6, gap: float = 0.008,
                       n_base_x: int = 32, n_base_y: int = 16, n_base_z: int = 2,
                       n_fin_y: int = 10, n_fin_z: int = 8
                       ) -> Tuple[np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]:
    """Радиатор: тонкая прямоугольная подложка + ряд параллельных рёбер.

    Подложка по X-Y, основание в плоскости z=0. Рёбра идут вдоль Y, расположены
    регулярно по X.
    """
    # Строим основание как обычную box-сетку.
    xs_base = np.linspace(0.0, base_Lx, n_base_x + 1)
    ys_base = np.linspace(0.0, base_Ly, n_base_y + 1)
    zs_base = np.linspace(0.0, base_thickness, n_base_z + 1)

    coords_base = []
    base_node_idx = {}  # (i, j, k) → global idx
    for k, z in enumerate(zs_base):
        for j, y in enumerate(ys_base):
            for i, x in enumerate(xs_base):
                base_node_idx[(i, j, k)] = len(coords_base)
                coords_base.append((x, y, z))

    def nid_base(i, j, k):
        return base_node_idx[(i, j, k)]

    tets_list = []
    for k in range(n_base_z):
        for j in range(n_base_y):
            for i in range(n_base_x):
                v = [
                    nid_base(i,     j,     k),
                    nid_base(i + 1, j,     k),
                    nid_base(i + 1, j + 1, k),
                    nid_base(i,     j + 1, k),
                    nid_base(i,     j,     k + 1),
                    nid_base(i + 1, j,     k + 1),
                    nid_base(i + 1, j + 1, k + 1),
                    nid_base(i,     j + 1, k + 1),
                ]
                for kt in _KUHN_TETS:
                    tets_list.append((v[kt[0]], v[kt[1]], v[kt[2]], v[kt[3]]))

    # Рёбра.
    # Распределим n_fins рёбер равномерно по X.
    stride = fin_thickness + gap
    total_w = n_fins * fin_thickness + (n_fins - 1) * gap
    x0 = (base_Lx - total_w) / 2

    coords_all = list(coords_base)
    nid_fin_lookup = {}  # (fin_index, i_local, j, k) → global idx (для уникальности)

    for f in range(n_fins):
        fin_x_min = x0 + f * stride
        fin_x_max = fin_x_min + fin_thickness
        # Внутри ребра: 1 ячейка по X (грубо), плюс соединение с основанием
        # через сопряжённую плоскость z = base_thickness.
        # Чтобы избежать «висячих узлов», берём общие узлы основания на
        # верхней грани в области ребра. То есть на z = base_thickness под
        # ребром у нас уже есть точки от основания.

        # Найдём в основании индексы i_base, ближайшие к fin_x_min и fin_x_max.
        i_min = int(np.argmin(np.abs(xs_base - fin_x_min)))
        i_max = int(np.argmin(np.abs(xs_base - fin_x_max)))
        if i_max <= i_min:
            i_max = i_min + 1
        # Берём индексы [i_min .. i_max] основания как нижнюю поверхность ребра.

        zs_fin = np.linspace(base_thickness, base_thickness + fin_height,
                              n_fin_z + 1)
        ys_fin = ys_base  # та же сетка по Y, что и в основании
        xs_fin = xs_base[i_min:i_max + 1]
        n_fin_x = len(xs_fin) - 1

        fin_node_idx = {}
        for k_loc, z in enumerate(zs_fin):
            for j_loc, y in enumerate(ys_fin):
                for i_loc, x in enumerate(xs_fin):
                    key = (f, i_loc, j_loc, k_loc)
                    if k_loc == 0:
                        # Это узел верхней грани основания — переиспользуем.
                        fin_node_idx[key] = nid_base(i_min + i_loc, j_loc,
                                                      n_base_z)
                    else:
                        fin_node_idx[key] = len(coords_all)
                        coords_all.append((x, y, z))

        def nid_fin(i_loc, j_loc, k_loc):
            return fin_node_idx[(f, i_loc, j_loc, k_loc)]

        for k_loc in range(n_fin_z):
            for j_loc in range(len(ys_fin) - 1):
                for i_loc in range(n_fin_x):
                    v = [
                        nid_fin(i_loc,     j_loc,     k_loc),
                        nid_fin(i_loc + 1, j_loc,     k_loc),
                        nid_fin(i_loc + 1, j_loc + 1, k_loc),
                        nid_fin(i_loc,     j_loc + 1, k_loc),
                        nid_fin(i_loc,     j_loc,     k_loc + 1),
                        nid_fin(i_loc + 1, j_loc,     k_loc + 1),
                        nid_fin(i_loc + 1, j_loc + 1, k_loc + 1),
                        nid_fin(i_loc,     j_loc + 1, k_loc + 1),
                    ]
                    for kt in _KUHN_TETS:
                        tets_list.append((v[kt[0]], v[kt[1]],
                                          v[kt[2]], v[kt[3]]))

    coords = np.asarray(coords_all, dtype=np.float64)
    tets = np.asarray(tets_list, dtype=np.int32)
    return _compact_unused_nodes(coords, tets)


# =============================================================================
# Утилита: убрать узлы, не использованные ни одним элементом.
# =============================================================================

def _compact_unused_nodes(nodes: np.ndarray,
                           tets: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """Удаляет из nodes узлы, не упомянутые в tets, и перенумеровывает tets."""
    used = np.zeros(nodes.shape[0], dtype=bool)
    used[tets.ravel()] = True
    if used.all():
        return _finalize(nodes, tets)
    new_idx = np.full(nodes.shape[0], -1, dtype=np.int32)
    new_idx[used] = np.arange(used.sum(), dtype=np.int32)
    new_nodes = nodes[used]
    new_tets = new_idx[tets]
    return _finalize(new_nodes, new_tets)


# =============================================================================
# Сфера.
# =============================================================================

def make_sphere(radius: float = 0.05, n_phi: int = 16, n_theta: int = 16,
                 n_radial: int = 8):
    """Сплошная сфера. Сферические координаты: радиус разбит на слои."""
    coords = []
    # Центр.
    coords.append([0.0, 0.0, 0.0])
    # Сетка по сферическим оболочкам.
    for ir in range(1, n_radial + 1):
        r = radius * ir / n_radial
        for it in range(n_theta + 1):
            theta = np.pi * it / n_theta  # 0..pi
            for ip in range(n_phi):
                phi = 2 * np.pi * ip / n_phi
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                coords.append([x, y, z])
    coords = np.array(coords, dtype=np.float64)
    # Соединение в гексаэдры с центром, потом декомпозиция Куна.
    # Упрощённый подход: триангуляция через scipy.spatial.Delaunay.
    try:
        from scipy.spatial import Delaunay
        delaunay = Delaunay(coords)
        tets = delaunay.simplices
    except ImportError:
        # Fallback: сразу куб обёрнут в сферу.
        return make_cylinder(radius=radius, height=2*radius,
                              n_radial=n_radial, n_circ=n_phi,
                              n_height=n_theta // 2 + 1)
    return _finalize(coords, tets.astype(np.int32))


# =============================================================================
# T-образный профиль.
# =============================================================================

def make_t_profile(width: float = 0.10, height: float = 0.10,
                    thickness: float = 0.02,
                    length: float = 0.05,
                    n_thickness: int = 4,
                    n_length: int = 6):
    """T-образный профиль: горизонтальная полка + вертикальная стенка."""
    coords = []; tets = []
    nx_top = max(8, int(width / thickness * 2))
    nz_top = n_thickness
    nx_stem = n_thickness
    nz_stem = max(6, int((height - thickness) / thickness * 1.5))
    # Полка (верх): x ∈ [0, width], z ∈ [height - thickness, height]
    nodes_top = _grid3d(0, width, height - thickness, height,
                         0, length, nx_top + 1, nz_top + 1, n_length + 1)
    tets_top = _kuhn_tets(nx_top, nz_top, n_length, nodes_top.shape[0])
    base = 0
    coords.append(nodes_top); tets.append(tets_top + base)
    base += nodes_top.shape[0]
    # Стойка (низ): x ∈ [width/2 - thickness/2, width/2 + thickness/2],
    #               z ∈ [0, height - thickness]
    nodes_stem = _grid3d((width - thickness) / 2, (width + thickness) / 2,
                          0, height - thickness,
                          0, length, nx_stem + 1, nz_stem + 1, n_length + 1)
    tets_stem = _kuhn_tets(nx_stem, nz_stem, n_length, nodes_stem.shape[0])
    coords.append(nodes_stem); tets.append(tets_stem + base)
    nodes = np.vstack(coords); tets = np.vstack(tets)
    return _compact_unused_nodes(nodes, tets.astype(np.int32))


# =============================================================================
# U-образный канал.
# =============================================================================

def make_u_channel(outer_width: float = 0.08, outer_height: float = 0.06,
                    thickness: float = 0.012,
                    length: float = 0.10,
                    n_thickness: int = 3,
                    n_length: int = 10):
    """П-образный (U) канал: дно + две стенки."""
    coords = []; tets = []
    base = 0
    # Дно: полная ширина, нижние n_thickness слоёв.
    nx = max(8, int(outer_width / thickness * 1.5))
    nz_bottom = n_thickness
    nodes_b = _grid3d(0, outer_width, 0, thickness, 0, length,
                       nx + 1, nz_bottom + 1, n_length + 1)
    tets_b = _kuhn_tets(nx, nz_bottom, n_length, nodes_b.shape[0])
    coords.append(nodes_b); tets.append(tets_b + base)
    base += nodes_b.shape[0]
    # Левая стенка.
    nx_w = n_thickness
    nz_w = max(4, int((outer_height - thickness) / thickness * 1.5))
    nodes_l = _grid3d(0, thickness, thickness, outer_height, 0, length,
                       nx_w + 1, nz_w + 1, n_length + 1)
    tets_l = _kuhn_tets(nx_w, nz_w, n_length, nodes_l.shape[0])
    coords.append(nodes_l); tets.append(tets_l + base)
    base += nodes_l.shape[0]
    # Правая стенка.
    nodes_r = _grid3d(outer_width - thickness, outer_width,
                       thickness, outer_height, 0, length,
                       nx_w + 1, nz_w + 1, n_length + 1)
    tets_r = _kuhn_tets(nx_w, nz_w, n_length, nodes_r.shape[0])
    coords.append(nodes_r); tets.append(tets_r + base)
    nodes = np.vstack(coords); tets = np.vstack(tets)
    return _compact_unused_nodes(nodes, tets.astype(np.int32))


# =============================================================================
# Диск с центральным отверстием (шайба).
# =============================================================================

def make_disk_with_hole(R_outer: float = 0.05, R_inner: float = 0.015,
                         thickness: float = 0.01,
                         n_circ: int = 32, n_radial: int = 6,
                         n_thickness: int = 3):
    """Тонкая шайба: внешний R, внутреннее отверстие R_inner, толщина."""
    coords = []
    # Параметризация: r ∈ [R_inner, R_outer], θ ∈ [0, 2π), z ∈ [0, thickness].
    for iz in range(n_thickness + 1):
        z = thickness * iz / n_thickness
        for ir in range(n_radial + 1):
            r = R_inner + (R_outer - R_inner) * ir / n_radial
            for it in range(n_circ):
                theta = 2 * np.pi * it / n_circ
                coords.append([r * np.cos(theta), r * np.sin(theta), z])
    coords = np.array(coords, dtype=np.float64)
    # Соединение в hexes (n_radial × n_circ × n_thickness ячеек, замкнутых по θ).
    tets = []
    layer = (n_radial + 1) * n_circ
    for iz in range(n_thickness):
        for ir in range(n_radial):
            for it in range(n_circ):
                it_n = (it + 1) % n_circ
                n0 = iz * layer + ir * n_circ + it
                n1 = iz * layer + ir * n_circ + it_n
                n2 = iz * layer + (ir + 1) * n_circ + it_n
                n3 = iz * layer + (ir + 1) * n_circ + it
                n4 = (iz + 1) * layer + ir * n_circ + it
                n5 = (iz + 1) * layer + ir * n_circ + it_n
                n6 = (iz + 1) * layer + (ir + 1) * n_circ + it_n
                n7 = (iz + 1) * layer + (ir + 1) * n_circ + it
                hex8 = np.array([n0, n1, n2, n3, n4, n5, n6, n7])
                for kt in _KUHN_TETS:
                    tets.append(hex8[kt])
    tets = np.array(tets, dtype=np.int32)
    return _finalize(coords, tets)


# =============================================================================
# Конус.
# =============================================================================

def make_cone(R_bottom: float = 0.05, R_top: float = 0.0,
               height: float = 0.10,
               n_circ: int = 24, n_radial: int = 5, n_height: int = 8):
    """Усечённый конус. R_top=0 даёт обычный конус, R_top=R_bottom даёт цилиндр."""
    coords = []
    for iz in range(n_height + 1):
        z = height * iz / n_height
        r_at_z = R_bottom + (R_top - R_bottom) * iz / n_height
        coords.append([0.0, 0.0, z])  # ось
        for ir in range(1, n_radial + 1):
            r = r_at_z * ir / n_radial
            if r < 1e-9:
                continue
            for it in range(n_circ):
                theta = 2 * np.pi * it / n_circ
                coords.append([r * np.cos(theta), r * np.sin(theta), z])
    coords = np.array(coords, dtype=np.float64)
    try:
        from scipy.spatial import Delaunay
        tets = Delaunay(coords).simplices.astype(np.int32)
    except ImportError:
        return make_cylinder(radius=R_bottom, height=height,
                              n_radial=n_radial, n_circ=n_circ,
                              n_height=n_height)
    return _finalize(coords, tets)


# =============================================================================
# Утилиты для регулярных сеток (общие для t-profile, u-channel).
# =============================================================================

def _grid3d(xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz):
    """Регулярная сетка узлов в параллелепипеде. Возвращает (nx*ny*nz, 3)."""
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def _kuhn_tets(nx_cells, ny_cells, nz_cells, total_nodes):
    """Декомпозиция Куна на регулярной сетке (nx+1)×(ny+1)×(nz+1) узлов."""
    nx = nx_cells + 1; ny = ny_cells + 1; nz = nz_cells + 1
    tets = []
    for i in range(nx_cells):
        for j in range(ny_cells):
            for k in range(nz_cells):
                n0 = (i  ) * ny * nz + (j  ) * nz + (k  )
                n1 = (i+1) * ny * nz + (j  ) * nz + (k  )
                n2 = (i+1) * ny * nz + (j+1) * nz + (k  )
                n3 = (i  ) * ny * nz + (j+1) * nz + (k  )
                n4 = (i  ) * ny * nz + (j  ) * nz + (k+1)
                n5 = (i+1) * ny * nz + (j  ) * nz + (k+1)
                n6 = (i+1) * ny * nz + (j+1) * nz + (k+1)
                n7 = (i  ) * ny * nz + (j+1) * nz + (k+1)
                hex8 = np.array([n0, n1, n2, n3, n4, n5, n6, n7])
                for kt in _KUHN_TETS:
                    tets.append(hex8[kt])
    return np.array(tets, dtype=np.int32)


# =============================================================================
# Спираль (виток теплообменника / индукционная катушка).
# =============================================================================

def make_helix(R_major: float = 0.05, r_section: float = 0.008,
                pitch: float = 0.015, n_turns: float = 2.0,
                n_phi: int = 8, n_radial: int = 3,
                n_along: int = None):
    """Спиральная трубка (виток теплообменника или индукционная катушка).

    R_major — радиус спирали (расстояние от оси Z до центра сечения).
    r_section — радиус сечения трубки.
    pitch — шаг спирали по оси Z (м/виток).
    n_turns — число витков (можно дробное).
    n_phi — узлов по углу сечения трубки.
    n_radial — узлов по радиусу сечения.
    n_along — узлов вдоль траектории (если None, выбирается автоматически).
    """
    if n_along is None:
        n_along = max(24, int(40 * n_turns))
    coords = []
    s_vals = np.linspace(0.0, n_turns * 2 * np.pi, n_along)
    # Параметризация: центральная линия (x(s), y(s), z(s)).
    for s in s_vals:
        cx = R_major * np.cos(s)
        cy = R_major * np.sin(s)
        cz = pitch * s / (2 * np.pi)
        # Локальный базис в плоскости сечения трубки.
        # Касательная к спирали:
        tx = -R_major * np.sin(s)
        ty =  R_major * np.cos(s)
        tz =  pitch / (2 * np.pi)
        T = np.array([tx, ty, tz])
        T = T / np.linalg.norm(T)
        # Нормаль (к оси Z).
        N = np.cross(T, np.array([0.0, 0.0, 1.0]))
        nl = np.linalg.norm(N)
        if nl < 1e-12:
            N = np.array([1.0, 0.0, 0.0])
        else:
            N = N / nl
        B = np.cross(T, N)
        # Узлы сечения.
        coords.append([cx, cy, cz])  # центр сечения
        for ir in range(1, n_radial + 1):
            r = r_section * ir / n_radial
            for ip in range(n_phi):
                phi = 2 * np.pi * ip / n_phi
                p = np.array([cx, cy, cz]) + r * (np.cos(phi) * N + np.sin(phi) * B)
                coords.append(p.tolist())
    coords = np.array(coords, dtype=np.float64)
    # Триангуляция через Delaunay (надёжно для свободных форм).
    try:
        from scipy.spatial import Delaunay
        tets = Delaunay(coords).simplices.astype(np.int32)
    except ImportError:
        # Без scipy не получится — fallback на сферу.
        return make_sphere(radius=R_major + r_section, n_phi=12,
                             n_theta=12, n_radial=4)
    return _finalize(coords, tets)


# =============================================================================
# Сферический купол (полусфера).
# =============================================================================

def make_hemisphere(radius: float = 0.05,
                     n_phi: int = 18, n_theta: int = 10, n_radial: int = 6):
    """Полусфера (верхняя половина): z ∈ [0, R]."""
    coords = [[0.0, 0.0, 0.0]]  # центр основания
    for ir in range(1, n_radial + 1):
        r = radius * ir / n_radial
        for it in range(n_theta + 1):
            theta = (np.pi / 2) * it / n_theta  # 0..pi/2 (только верх)
            for ip in range(n_phi):
                phi = 2 * np.pi * ip / n_phi
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                coords.append([x, y, z])
    coords = np.array(coords, dtype=np.float64)
    try:
        from scipy.spatial import Delaunay
        tets = Delaunay(coords).simplices.astype(np.int32)
    except ImportError:
        return make_sphere(radius=radius)
    return _finalize(coords, tets)


# =============================================================================
# Эллипсоид (сплюснутая или вытянутая сфера).
# =============================================================================

def make_ellipsoid(rx: float = 0.05, ry: float = 0.03, rz: float = 0.02,
                    n_phi: int = 16, n_theta: int = 12, n_radial: int = 6):
    """Эллипсоид с полуосями rx, ry, rz."""
    coords = [[0.0, 0.0, 0.0]]
    for ir in range(1, n_radial + 1):
        s = ir / n_radial  # масштабирование радиуса
        for it in range(n_theta + 1):
            theta = np.pi * it / n_theta
            for ip in range(n_phi):
                phi = 2 * np.pi * ip / n_phi
                x = s * rx * np.sin(theta) * np.cos(phi)
                y = s * ry * np.sin(theta) * np.sin(phi)
                z = s * rz * np.cos(theta)
                coords.append([x, y, z])
    coords = np.array(coords, dtype=np.float64)
    try:
        from scipy.spatial import Delaunay
        tets = Delaunay(coords).simplices.astype(np.int32)
    except ImportError:
        return make_sphere(radius=max(rx, ry, rz))
    return _finalize(coords, tets)


# =============================================================================
# Контроль сетки: локальное и глобальное измельчение.
# =============================================================================

def refine_mesh_uniform(nodes: np.ndarray, tets: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
    """Равномерное измельчение тетраэдральной сетки делением 1→8.

    Каждый тетраэдр разбивается на 8 меньших через добавление узлов на
    серединах рёбер (Bey's subdivision). Размер ячейки уменьшается ровно
    в 2 раза, число элементов растёт в 8 раз.

    Возвращает (new_nodes, new_tets) — новые массивы. Поверхностные грани
    нужно пересчитать через _finalize().
    """
    # 1. Собираем все уникальные рёбра.
    Ne = tets.shape[0]
    edges_set = {}  # (a, b) → новый узел
    midpoints = []
    # Для нумерации: сначала старые узлы (0..N-1), затем середины (N..N+M-1).
    N0 = nodes.shape[0]
    new_nodes = [nodes]
    edge_indices = np.empty((Ne, 6), dtype=np.int32)
    # Каноническая нумерация рёбер тетраэдра: 0-1, 0-2, 0-3, 1-2, 1-3, 2-3.
    EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    next_id = N0
    for e in range(Ne):
        tet = tets[e]
        for ei, (a, b) in enumerate(EDGE_PAIRS):
            i, j = int(tet[a]), int(tet[b])
            key = (min(i, j), max(i, j))
            if key not in edges_set:
                edges_set[key] = next_id
                midpoints.append(0.5 * (nodes[i] + nodes[j]))
                next_id += 1
            edge_indices[e, ei] = edges_set[key]
    if midpoints:
        new_nodes.append(np.array(midpoints, dtype=np.float64))
    out_nodes = np.vstack(new_nodes)

    # 2. Для каждого тет — 8 новых.
    # Стандартная схема Bey: 4 уголных + 4 внутренних октаэдра.
    new_tets = []
    for e in range(Ne):
        v = tets[e]   # 4 угла
        m = edge_indices[e]  # 6 серединных
        # 4 уголных тета (с одним из старых углов).
        new_tets.append([v[0], m[0], m[1], m[2]])  # near v0
        new_tets.append([v[1], m[0], m[3], m[4]])  # near v1
        new_tets.append([v[2], m[1], m[3], m[5]])  # near v2
        new_tets.append([v[3], m[2], m[4], m[5]])  # near v3
        # 4 внутренних октаэдральных (декомпозиция октаэдра m0,m1,m2,m3,m4,m5
        # по самой короткой диагонали).
        new_tets.append([m[0], m[1], m[2], m[4]])
        new_tets.append([m[0], m[1], m[4], m[3]])
        new_tets.append([m[1], m[2], m[4], m[5]])
        new_tets.append([m[1], m[3], m[4], m[5]])
    out_tets = np.array(new_tets, dtype=np.int32)
    return out_nodes, out_tets


def refine_mesh_in_region(nodes: np.ndarray, tets: np.ndarray,
                            center: np.ndarray, radius: float) -> Tuple[
        np.ndarray, np.ndarray]:
    """Адаптивное измельчение: делятся только тетры, у которых центроид
    попадает в сферу (center, radius). Остальные оставляются как есть.

    Это создаст «висячие узлы» на стыках, что для P1-МКЭ некорректно строго —
    но для большинства задач даёт приемлемый результат. Для строгого согласования
    нужны переходные элементы или ремеширование.

    В контексте дипломной программы это полезный демонстрационный инструмент.
    """
    # Центроиды.
    cents = 0.25 * (nodes[tets[:, 0]] + nodes[tets[:, 1]]
                     + nodes[tets[:, 2]] + nodes[tets[:, 3]])
    d2 = np.sum((cents - center) ** 2, axis=1)
    mask_refine = d2 < radius * radius
    if not np.any(mask_refine):
        return nodes, tets  # ничего не делим
    # Делим только подмножество.
    tets_to_refine = tets[mask_refine]
    tets_kept = tets[~mask_refine]
    refined_nodes, refined_tets = refine_mesh_uniform(nodes, tets_to_refine)
    # Обновлённые tets_kept ссылаются на старые узлы 0..N-1 (которые в начале
    # refined_nodes). Объединяем.
    out_tets = np.vstack([tets_kept, refined_tets])
    return refined_nodes, out_tets
