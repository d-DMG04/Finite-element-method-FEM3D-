# -*- coding: utf-8 -*-
"""
fem3d.mesh
==========

Работа с геометрией и сетками: пресеты, обёртки над импортом из meshio,
вспомогательные структуры данных.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Пресеты типовых геометрий (Ф1.3 ТЗ).
# =============================================================================

@dataclass(frozen=True)
class BoxPreset:
    """Описание параллелепипедного пресета."""
    label: str
    Lx: float    # размер по X, м
    Ly: float    # размер по Y, м
    Lz: float    # размер по Z, м
    nx: int      # число разбиений по X
    ny: int      # число разбиений по Y
    nz: int      # число разбиений по Z
    description: str = ""


# Список пресетов, отображаемых в выпадающем списке левой панели.
# Размеры подобраны как «учебно-инженерные»: 5–10 см куб, тонкая пластина,
# длинная балка.
PRESETS = [
    BoxPreset(
        label="Куб 100×100×100 мм",
        Lx=0.10, Ly=0.10, Lz=0.10, nx=15, ny=15, nz=15,
        description="Стандартный тестовый куб, 100×100×100 мм.",
    ),
    BoxPreset(
        label="Пластина 200×200×10 мм",
        Lx=0.20, Ly=0.20, Lz=0.01, nx=20, ny=20, nz=4,
        description="Тонкая пластина — типичная плата или диск.",
    ),
    BoxPreset(
        label="Балка 200×40×40 мм",
        Lx=0.20, Ly=0.04, Lz=0.04, nx=30, ny=8, nz=8,
        description="Длинная балка — конструкционная деталь.",
    ),
    BoxPreset(
        label="Микрокуб 20×20×20 мм",
        Lx=0.020, Ly=0.020, Lz=0.020, nx=10, ny=10, nz=10,
        description="Маленький куб — корпус электронного компонента.",
    ),
]


# =============================================================================
# Справочник материалов (значения λ из таблицы 1.1 ПЗ при T = 20 °C).
# =============================================================================

@dataclass(frozen=True)
class Material:
    name: str
    lambda_: float          # Вт/(м·К)


MATERIALS = [
    Material("Серебро",                429.0),
    Material("Медь",                   401.0),
    Material("Алюминий",               237.0),
    Material("Латунь",                 110.0),
    Material("Сталь углеродистая",      55.0),
    Material("Сталь нержавеющая",       18.0),
    Material("Стекло",                   1.0),
    Material("Бетон",                    1.5),
    Material("Кирпич",                   0.65),
    Material("Дерево (попер. волокон)",  0.15),
    Material("Минеральная вата",         0.045),
    Material("Воздух (неподвижный)",     0.026),
]


def material_by_name(name: str) -> Optional[Material]:
    for m in MATERIALS:
        if m.name == name:
            return m
    return None


# =============================================================================
# Информация о сетке (для отображения в левой панели).
# =============================================================================

@dataclass
class MeshInfo:
    n_nodes: int
    n_elements: int
    n_boundary_faces: int
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]

    @property
    def memory_mb(self) -> float:
        # Грубая оценка: узлы — 24 байта, элементы — 16 байт, грани — 16 байт.
        bytes_total = 24 * self.n_nodes + 16 * self.n_elements + 16 * self.n_boundary_faces
        return bytes_total / (1024 * 1024)


def compute_mesh_info(nodes: np.ndarray, n_elements: int,
                      n_boundary_faces: int) -> MeshInfo:
    if nodes.size == 0:
        return MeshInfo(0, n_elements, n_boundary_faces,
                        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return MeshInfo(
        n_nodes=int(nodes.shape[0]),
        n_elements=int(n_elements),
        n_boundary_faces=int(n_boundary_faces),
        bbox_min=tuple(nodes.min(axis=0).tolist()),
        bbox_max=tuple(nodes.max(axis=0).tolist()),
    )


# =============================================================================
# Импорт из внешних форматов — единая точка входа.
# =============================================================================
# Поддерживаются:
#   .msh (Gmsh)       — через meshio. Требует объёмные тетраэдры в файле.
#   .vtu / .vtk       — через meshio. Аналогично, нужны тетраэдры.
#   .stl              — через meshio (только поверхностная сетка) + gmsh для
#                       автоматической объёмной тетраэдризации (если установлен).
#   .step / .stp      — через gmsh: чтение CAD-геометрии и тетраэдризация.
#                       Требует установленного gmsh (pip install gmsh).
#
# Все импорты сторонних библиотек — ленивые: модуль fem3d.mesh загружается
# и без них; пользователь увидит понятную ошибку с инструкцией установки.
#
# Возвращаемый формат во всех функциях единый:
#   nodes:             (N, 3)  float64  координаты узлов
#   tets:              (Ne, 4) int32    связности тетраэдров
#   bnd_nodes:         (Nf, 3) int32    индексы узлов поверхностных треугольников
#   bnd_face_ids:      (Nf,)   int32    идентификатор группы поверхности
# =============================================================================

SUPPORTED_IMPORT_EXTENSIONS = (".msh", ".vtu", ".vtk", ".stl", ".step", ".stp")


def import_mesh_file(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Универсальная точка входа: определяет формат по расширению и вызывает
    соответствующую функцию импорта. Возвращает кортеж в едином формате."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".msh":
        return import_msh(path)
    if ext in (".vtu", ".vtk"):
        return import_vtu(path)
    if ext == ".stl":
        return import_stl(path)
    if ext in (".step", ".stp"):
        return import_step(path)
    raise RuntimeError(
        f"Неподдерживаемое расширение файла: {ext}\n"
        f"Поддерживаются: {', '.join(SUPPORTED_IMPORT_EXTENSIONS)}"
    )


def import_msh(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Импорт объёмной тетраэдральной сетки Gmsh (.msh)."""
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError(
            "Импорт .msh требует библиотеку meshio.\n"
            "Установите её: pip install meshio"
        ) from exc

    m = meshio.read(path)
    return _meshio_to_arrays(m, path)


def import_vtu(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Импорт сетки в формате VTK Unstructured Grid (.vtu/.vtk)."""
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError(
            "Импорт .vtu/.vtk требует библиотеку meshio.\n"
            "Установите её: pip install meshio"
        ) from exc

    m = meshio.read(path)
    return _meshio_to_arrays(m, path)


def import_stl(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Импорт STL: поверхностная сетка → объёмная тетраэдризация через gmsh.

    STL содержит только поверхностные треугольники, поэтому необходимо
    построить объёмную тетраэдральную сетку. Это делает gmsh: сначала
    создаётся «оболочка», потом запускается алгоритм тетраэдризации.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "Импорт STL требует библиотеку gmsh для объёмной тетраэдризации.\n"
            "Установите её: pip install gmsh"
        ) from exc

    if not gmsh.isInitialized():
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)  # глушим вывод gmsh
        gmsh.clear()
        gmsh.model.add("imported_stl")
        gmsh.merge(path)
        # Создаём объёмное тело из поверхности.
        gmsh.model.mesh.classifySurfaces(angle=40.0 * np.pi / 180.0,
                                          boundary=True, forReparametrization=True)
        gmsh.model.mesh.createGeometry()
        surfaces = gmsh.model.getEntities(dim=2)
        sl = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
        gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        # Сохраняем во временный файл и читаем через meshio (надёжнее, чем
        # прямой обмен через gmsh API).
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
        tmp.close()
        gmsh.write(tmp.name)
        try:
            return import_msh(tmp.name)
        finally:
            os.unlink(tmp.name)
    finally:
        gmsh.finalize()


def import_step(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Импорт CAD-геометрии STEP с тетраэдризацией через gmsh."""
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "Импорт STEP требует библиотеку gmsh.\n"
            "Установите её: pip install gmsh"
        ) from exc

    if not gmsh.isInitialized():
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("imported_step")
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        # Адаптивный размер сетки: 5 % от характерного размера габарита.
        bbox = gmsh.model.getBoundingBox(-1, -1)
        diag = np.sqrt(sum((bbox[i + 3] - bbox[i]) ** 2 for i in range(3)))
        h = max(diag * 0.03, 1e-6)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h * 0.3)
        gmsh.model.mesh.generate(3)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
        tmp.close()
        gmsh.write(tmp.name)
        try:
            return import_msh(tmp.name)
        finally:
            os.unlink(tmp.name)
    finally:
        gmsh.finalize()


def _meshio_to_arrays(m, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Извлекает узлы, тетраэдры и поверхностные треугольники из meshio.Mesh."""
    nodes = np.asarray(m.points, dtype=np.float64)
    if nodes.shape[1] == 2:
        nodes = np.hstack([nodes, np.zeros((nodes.shape[0], 1))])

    tets_blocks = [c for c in m.cells if c.type == "tetra"]
    if not tets_blocks:
        raise RuntimeError(
            f"В файле {path} не найдено тетраэдральных элементов.\n"
            "Если это поверхностная сетка (например, STL), используйте "
            "import_stl() для автоматической тетраэдризации."
        )
    tets = np.vstack([blk.data for blk in tets_blocks]).astype(np.int32)

    faces = _extract_surface_faces(tets)
    face_ids = np.zeros(faces.shape[0], dtype=np.int32)
    return nodes, tets, faces.astype(np.int32), face_ids


def _extract_surface_faces(tets: np.ndarray) -> np.ndarray:
    """Возвращает (Nf, 3) — поверхностные треугольники тетраэдральной сетки."""
    # Локальные грани тетраэдра.
    local_faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int32)
    Ne = tets.shape[0]
    all_faces = np.empty((4 * Ne, 3), dtype=np.int64)
    for k in range(4):
        all_faces[k * Ne:(k + 1) * Ne, :] = tets[:, local_faces[k]]

    # Сортируем узлы каждой грани, чтобы одинаковые грани имели одинаковый ключ.
    sorted_faces = np.sort(all_faces, axis=1)
    # Уникальные грани и количества вхождений.
    keys = sorted_faces.view([("a", np.int64), ("b", np.int64), ("c", np.int64)]).ravel()
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    surf_mask = counts == 1
    surf_indices = np.flatnonzero(surf_mask)
    # Восстановим оригинальную ориентированную грань для каждого уникального ключа.
    # Берём первое попавшееся вхождение.
    first_occurrence = np.full(uniq.size, -1, dtype=np.int64)
    for i, idx in enumerate(inv):
        if first_occurrence[idx] < 0:
            first_occurrence[idx] = i
    surface_face_orig_idx = first_occurrence[surf_indices]
    return all_faces[surface_face_orig_idx, :]
