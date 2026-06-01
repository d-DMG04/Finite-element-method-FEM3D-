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


@dataclass(frozen=True)
class ShapePreset:
    """Описание пресета сложной геометрии.

    factory — функция, принимающая опциональный density: float (множитель
    плотности сетки, по умолчанию 1.0) и возвращающая
    (nodes, tets, bnd_nodes, bnd_face_ids).
    Если factory не принимает density, вызывается без аргументов.
    """
    label: str
    factory: object
    description: str = ""
    supports_density: bool = True

    def build(self, density: float = 1.0):
        """Построить геометрию. density управляет плотностью сетки:
        1.0 — стандартная, 2.0 — вдвое мельче (×8 элементов), 0.5 — грубее."""
        import inspect
        try:
            sig = inspect.signature(self.factory)
            if "density" in sig.parameters:
                return self.factory(density=density)
        except (ValueError, TypeError):
            pass
        return self.factory()


# Список пресетов параллелепипеда.
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


# Пресеты сложных геометрий. Импортируем shapes лениво в фабриках, чтобы
# модуль fem3d.mesh оставался самодостаточным.
# density — множитель плотности сетки: 1.0 стандарт, 2.0 вдвое мельче.
def _sc(n, density):
    """Масштабировать число разбиений с округлением (минимум сохраняем)."""
    return max(2, int(round(n * density)))


def _make_cylinder_preset(density: float = 1.0):
    from .shapes import make_cylinder
    return make_cylinder(radius=0.05, height=0.10,
                          n_radial=_sc(4, density),
                          n_angular=_sc(24, density),
                          n_axial=_sc(10, density))


def _make_hollow_cyl_preset(density: float = 1.0):
    from .shapes import make_hollow_cylinder
    return make_hollow_cylinder(r_inner=0.03, r_outer=0.05, height=0.10,
                                 n_radial=_sc(3, density),
                                 n_angular=_sc(24, density),
                                 n_axial=_sc(10, density))


def _make_torus_preset(density: float = 1.0):
    from .shapes import make_torus
    return make_torus(R_major=0.06, r_minor=0.018,
                      n_major=_sc(32, density), n_minor=_sc(14, density),
                      n_radial=_sc(3, density))


def _make_pyramid_preset(density: float = 1.0):
    from .shapes import make_pyramid
    return make_pyramid(base_side=0.10, height=0.08,
                        n_base=_sc(14, density), n_axial=_sc(10, density))


def _make_l_beam_preset(density: float = 1.0):
    from .shapes import make_l_beam
    return make_l_beam(arm_length=0.10, thickness=0.02, depth=0.05,
                       n_arm=_sc(14, density), n_thick=_sc(4, density),
                       n_depth=_sc(8, density))


def _make_pcb_preset(density: float = 1.0):
    from .shapes import make_plate_with_hole
    return make_plate_with_hole(Lx=0.20, Ly=0.10, thickness=0.005,
                                hole_x=0.06, hole_y=0.04,
                                hole_w=0.04, hole_h=0.02,
                                nx=_sc(28, density), ny=_sc(14, density),
                                nz=_sc(3, density))


def _make_fin_heatsink_preset(density: float = 1.0):
    from .shapes import make_fin_heatsink
    return make_fin_heatsink(base_Lx=0.10, base_Ly=0.06, base_thickness=0.005,
                              fin_height=0.025, fin_thickness=0.003,
                              n_fins=6, gap=0.008,
                              n_base_x=_sc(32, density), n_base_y=_sc(16, density),
                              n_base_z=_sc(2, density),
                              n_fin_y=_sc(8, density), n_fin_z=_sc(6, density))


def _make_sphere_preset(density: float = 1.0):
    from .shapes import make_sphere
    return make_sphere(radius=0.05, n_phi=_sc(14, density),
                        n_theta=_sc(10, density), n_radial=_sc(5, density))


def _make_t_profile_preset(density: float = 1.0):
    from .shapes import make_t_profile
    return make_t_profile(width=0.10, height=0.10, thickness=0.02,
                           length=0.05, n_thickness=_sc(3, density),
                           n_length=_sc(6, density))


def _make_u_channel_preset(density: float = 1.0):
    from .shapes import make_u_channel
    return make_u_channel(outer_width=0.08, outer_height=0.06,
                           thickness=0.012, length=0.10,
                           n_thickness=_sc(3, density), n_length=_sc(10, density))


def _make_disk_with_hole_preset(density: float = 1.0):
    from .shapes import make_disk_with_hole
    return make_disk_with_hole(R_outer=0.05, R_inner=0.015,
                                thickness=0.01,
                                n_circ=_sc(24, density), n_radial=_sc(5, density),
                                n_thickness=_sc(3, density))


def _make_cone_preset(density: float = 1.0):
    from .shapes import make_cone
    return make_cone(R_bottom=0.05, R_top=0.01, height=0.10,
                      n_circ=_sc(20, density), n_radial=_sc(4, density),
                      n_height=_sc(8, density))


def _make_helix_preset(density: float = 1.0):
    from .shapes import make_helix
    return make_helix(R_major=0.04, r_section=0.008,
                       pitch=0.018, n_turns=2.0,
                       n_phi=_sc(8, density), n_radial=_sc(3, density),
                       n_along=_sc(48, density))


def _make_hemisphere_preset(density: float = 1.0):
    from .shapes import make_hemisphere
    return make_hemisphere(radius=0.05, n_phi=_sc(14, density),
                            n_theta=_sc(8, density), n_radial=_sc(5, density))


def _make_ellipsoid_preset(density: float = 1.0):
    from .shapes import make_ellipsoid
    return make_ellipsoid(rx=0.06, ry=0.04, rz=0.025,
                            n_phi=_sc(14, density), n_theta=_sc(10, density),
                            n_radial=_sc(5, density))


SHAPE_PRESETS = [
    ShapePreset(label="Цилиндр Ø100×100 мм",
                factory=_make_cylinder_preset,
                description="Сплошной цилиндр Ø=100, h=100 мм."),
    ShapePreset(label="Труба Ø60–100×100 мм",
                factory=_make_hollow_cyl_preset,
                description="Полый цилиндр, внутр. диаметр 60 мм, внеш. 100 мм."),
    ShapePreset(label="Тор (бублик) R=60, r=18 мм",
                factory=_make_torus_preset,
                description="Тороидальная катушка, R=60, r=18 мм."),
    ShapePreset(label="Пирамида 100×100 мм, h=80 мм",
                factory=_make_pyramid_preset,
                description="Квадратная пирамида: основание 100×100 мм, высота 80 мм."),
    ShapePreset(label="L-балка 100×100×50 мм",
                factory=_make_l_beam_preset,
                description="L-образный профиль: плечи 100 мм, толщина 20 мм."),
    ShapePreset(label="Плата 200×100×5 мм с отверстием",
                factory=_make_pcb_preset,
                description="Тонкая печатная плата с прямоугольным вырезом."),
    ShapePreset(label="Радиатор с 6 рёбрами",
                factory=_make_fin_heatsink_preset,
                description="Подложка 100×60×5 мм с шестью рёбрами 25×3 мм."),
    ShapePreset(label="Сфера Ø100 мм",
                factory=_make_sphere_preset,
                description="Сплошная сфера диаметром 100 мм."),
    ShapePreset(label="T-профиль 100×100×20 мм, L=50 мм",
                factory=_make_t_profile_preset,
                description="T-образный профиль балки: полка + стойка."),
    ShapePreset(label="U-канал 80×60 мм, L=100 мм",
                factory=_make_u_channel_preset,
                description="П-образный канал: дно + две стенки, толщина 12 мм."),
    ShapePreset(label="Диск с отверстием Ø100/Ø30 мм, h=10 мм",
                factory=_make_disk_with_hole_preset,
                description="Шайба: внешний Ø=100, отверстие Ø=30 мм."),
    ShapePreset(label="Конус Ø100→Ø20 мм, h=100 мм",
                factory=_make_cone_preset,
                description="Усечённый конус: основание Ø=100, верх Ø=20 мм."),
    ShapePreset(label="Спираль Ø80×40 мм, 2 витка",
                factory=_make_helix_preset,
                description="Виток теплообменника или индукционная катушка: "
                            "R_major=40 мм, r_section=8 мм, шаг 18 мм, 2 витка."),
    ShapePreset(label="Полусфера Ø100 мм",
                factory=_make_hemisphere_preset,
                description="Сферический купол: радиус 50 мм, верхняя половина."),
    ShapePreset(label="Эллипсоид 120×80×50 мм",
                factory=_make_ellipsoid_preset,
                description="Сплюснутый эллипсоид: полуоси 60, 40, 25 мм."),
]


# =============================================================================
# Справочник материалов (значения λ из таблицы 1.1 ПЗ при T = 20 °C).
# =============================================================================

@dataclass(frozen=True)
class Material:
    """Свойства материала.

    name      — отображаемое имя;
    lambda_   — коэффициент теплопроводности при T = 20 °C, Вт/(м·К);
    rho       — плотность, кг/м³ (для нестационара и тепловой ёмкости);
    cp        — удельная теплоёмкость, Дж/(кг·К);
    emissivity — степень черноты поверхности (0..1) для излучения;
    lambda_temp_coef — температурный коэффициент λ, [1/К].
        Полная формула: λ(T) = lambda_ * (1 + lambda_temp_coef * (T - 20)).
        Для большинства металлов отрицательный (λ уменьшается при росте T).
    category — категория для группировки в GUI ("металлы"/"диэлектрики"/...).
    is_anisotropic — если True, lambda_ игнорируется и используются
        lambda_x/y/z по осям. Подходит для композитов, печатных плат, кристаллов.
    lambda_x, lambda_y, lambda_z — анизотропные значения.
    """
    name: str
    lambda_: float
    rho: float = 0.0
    cp: float = 0.0
    emissivity: float = 0.0
    lambda_temp_coef: float = 0.0
    category: str = "прочее"
    is_anisotropic: bool = False
    lambda_x: float = 0.0
    lambda_y: float = 0.0
    lambda_z: float = 0.0

    def lambda_at(self, T_celsius: float) -> float:
        """Эффективный λ при заданной температуре."""
        return self.lambda_ * (1.0 + self.lambda_temp_coef * (T_celsius - 20.0))

    def effective_lambdas(self) -> tuple:
        """Возвращает (λ_x, λ_y, λ_z)."""
        if self.is_anisotropic:
            return self.lambda_x, self.lambda_y, self.lambda_z
        return self.lambda_, self.lambda_, self.lambda_

    def effective_lambda(self) -> float:
        """Скалярный λ для отображения/изотропного расчёта.
        Для анизотропного — среднее по осям."""
        if self.is_anisotropic:
            return (self.lambda_x + self.lambda_y + self.lambda_z) / 3.0
        return self.lambda_


# Справочник материалов с реальными физическими свойствами при T=20°C.
# Источники: NIST, ASM Materials Handbook.
#   ρ — кг/м³, cp — Дж/(кг·К), ε — степень черноты (полированная/окисленная),
#   lambda_temp_coef — линейный коэффициент для λ(T).
MATERIALS = [
    # --- Металлы (высокая λ, низкая ε для полированных) ---
    Material("Серебро",       429.0, 10490, 235, 0.02, -1.2e-4, "Металлы"),
    Material("Медь",          401.0,  8960, 385, 0.05, -1.5e-4, "Металлы"),
    Material("Золото",        317.0, 19300, 129, 0.02, -1.4e-4, "Металлы"),
    Material("Алюминий",      237.0,  2702, 903, 0.09, -3.0e-4, "Металлы"),
    Material("Латунь",        110.0,  8530, 380, 0.40, -1.0e-4, "Металлы"),
    Material("Железо чистое",  80.4,  7870, 449, 0.21, -3.5e-4, "Металлы"),
    Material("Сталь углеродистая", 55.0, 7850, 490, 0.80, -1.5e-4, "Металлы"),
    Material("Сталь нержавеющая 304", 16.2, 7900, 500, 0.50, +1.0e-4, "Металлы"),
    Material("Титан",          21.9,  4500, 522, 0.30, +2.5e-4, "Металлы"),
    Material("Свинец",         35.3, 11340, 130, 0.28, -2.0e-4, "Металлы"),

    # --- Полупроводники ---
    Material("Кремний",       148.0,  2330, 712, 0.60, -1.0e-3, "Полупроводники"),
    Material("Германий",       60.0,  5323, 320, 0.50, -5.0e-4, "Полупроводники"),

    # --- Диэлектрики и стёкла ---
    Material("Стекло оконное",  1.05, 2500, 840, 0.95, 0.0, "Диэлектрики"),
    Material("Кварцевое стекло", 1.4, 2200, 745, 0.93, 0.0, "Диэлектрики"),
    Material("Керамика (Al2O3)", 30.0, 3970, 880, 0.50, 0.0, "Диэлектрики"),
    Material("PCB FR-4",          0.3, 1850, 1300, 0.90, 0.0, "Диэлектрики"),

    # --- Строительные ---
    Material("Бетон",            1.5,  2400, 880, 0.90, 0.0, "Строительные"),
    Material("Кирпич",           0.65, 1800, 840, 0.93, 0.0, "Строительные"),
    Material("Гипсокартон",      0.21,  800, 1090, 0.90, 0.0, "Строительные"),

    # --- Древесина ---
    Material("Дуб (поперёк)",    0.17,  720, 2400, 0.90, 0.0, "Дерево"),
    Material("Сосна (поперёк)",  0.12,  450, 2300, 0.90, 0.0, "Дерево"),

    # --- Утеплители ---
    Material("Минвата",          0.045, 100, 840, 0.90, 0.0, "Утеплители"),
    Material("Пенополистирол",   0.038,  35, 1340, 0.90, 0.0, "Утеплители"),
    Material("Пенополиуретан",   0.025,  35, 1470, 0.90, 0.0, "Утеплители"),

    # --- Жидкости и газы ---
    Material("Вода",             0.60, 998, 4186, 0.96, +1.5e-3, "Жидкости"),
    Material("Воздух",           0.026, 1.2, 1005, 0.0, +3.0e-3, "Газы"),

    # --- Анизотропные материалы (композиты, кристаллы) ---
    # λ задаётся через lambda_x/y/z; lambda_ — для совместимости (среднее).
    Material("Графит (вдоль слоёв)", 200.0, 2250, 710, 0.85, 0.0,
             "Анизотропные", is_anisotropic=True,
             lambda_x=200.0, lambda_y=200.0, lambda_z=10.0),
    Material("Углепластик (CFRP)", 50.0, 1600, 1050, 0.85, 0.0,
             "Анизотропные", is_anisotropic=True,
             lambda_x=10.0, lambda_y=0.8, lambda_z=0.8),
    Material("Дерево (вдоль волокон)", 0.3, 720, 2400, 0.90, 0.0,
             "Анизотропные", is_anisotropic=True,
             lambda_x=0.40, lambda_y=0.17, lambda_z=0.17),
    Material("PCB FR-4 (анизотроп.)", 0.5, 1850, 1300, 0.90, 0.0,
             "Анизотропные", is_anisotropic=True,
             lambda_x=0.8, lambda_y=0.8, lambda_z=0.3),
]


# Файл пользовательских материалов. Подгружается при импорте модуля,
# дополняет MATERIALS (но в код не сохраняется).
_USER_MATERIALS_FILE = os.path.expanduser("~/.fem_heat3d_user_materials.json")


def load_user_materials() -> list:
    """Загрузить пользовательские материалы из ~/.fem_heat3d_user_materials.json."""
    import json
    if not os.path.isfile(_USER_MATERIALS_FILE):
        return []
    try:
        with open(_USER_MATERIALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = []
        for d in data:
            result.append(Material(
                name=d.get("name", "Unnamed"),
                lambda_=float(d.get("lambda_", 1.0)),
                rho=float(d.get("rho", 0.0)),
                cp=float(d.get("cp", 0.0)),
                emissivity=float(d.get("emissivity", 0.0)),
                lambda_temp_coef=float(d.get("lambda_temp_coef", 0.0)),
                category=d.get("category", "Пользовательские"),
            ))
        return result
    except Exception:
        return []


def save_user_materials(materials: list) -> None:
    """Сохранить пользовательские материалы в файл."""
    import json
    data = []
    for m in materials:
        data.append({
            "name": m.name, "lambda_": m.lambda_, "rho": m.rho, "cp": m.cp,
            "emissivity": m.emissivity, "lambda_temp_coef": m.lambda_temp_coef,
            "category": m.category,
        })
    with open(_USER_MATERIALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def all_materials() -> list:
    """Возвращает встроенные + пользовательские материалы."""
    return MATERIALS + load_user_materials()


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


def _extract_surface_faces_with_opposite(tets: np.ndarray):
    """Граничные грани + индекс противоположной вершины породившего тетраэдра.

    Возвращает (faces (Nf,3), opposite (Nf,)). Противоположная вершина —
    это четвёртый узел тетраэдра, не входящий в грань. Используется для
    надёжного построения внешней нормали (направлена от opp к грани) даже
    на невыпуклых телах.
    """
    # Локальные грани и противоположная вершина для каждой.
    # Грань k образована тремя узлами, opp[k] — оставшийся узел.
    local_faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
                            dtype=np.int32)
    local_opp = np.array([0, 1, 2, 3], dtype=np.int32)
    Ne = tets.shape[0]
    all_faces = np.empty((4 * Ne, 3), dtype=np.int64)
    all_opp = np.empty(4 * Ne, dtype=np.int64)
    for k in range(4):
        all_faces[k * Ne:(k + 1) * Ne, :] = tets[:, local_faces[k]]
        all_opp[k * Ne:(k + 1) * Ne] = tets[:, local_opp[k]]
    sorted_faces = np.sort(all_faces, axis=1)
    keys = sorted_faces.view([("a", np.int64), ("b", np.int64),
                               ("c", np.int64)]).ravel()
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    surf_mask = counts == 1
    surf_indices = np.flatnonzero(surf_mask)
    first_occurrence = np.full(uniq.size, -1, dtype=np.int64)
    for i, idx in enumerate(inv):
        if first_occurrence[idx] < 0:
            first_occurrence[idx] = i
    orig_idx = first_occurrence[surf_indices]
    return all_faces[orig_idx, :], all_opp[orig_idx]


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
