# -*- coding: utf-8 -*-
"""
fem3d.problem
=============

Высокоуровневое описание задачи теплопроводности: геометрия, материал,
шесть граничных условий, источники. Оркестрирует обращение к ядру через
CoreBridge и хранит результат расчёта.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .core_bridge import (
    BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN, BC_RADIATION,
    CoreBridge, FACE_NAMES, SolverInfo,
    VOLSRC_BOX, VOLSRC_SPHERE,
)


# =============================================================================
# Описание граничного условия.
# =============================================================================

@dataclass
class BoundaryCondition:
    """Граничное условие на одной грани.

    Конвенция знаков для пользователя (физически интуитивная):
      - q0 > 0  →  тепло ВХОДИТ в тело (нагрев);
      - q0 < 0  →  тепло ВЫХОДИТ из тела (отвод).
    Внутри ядро использует обратную математическую конвенцию
    (-λ∂T/∂n = -q0), это преобразование делается в push_to_core.
    """
    type: int = BC_NONE         # один из BC_*
    T0: float = 20.0            # Дирихле: заданная температура, °C
    q0: float = 0.0             # Нейман: поток ВНУТРЬ тела, Вт/м²
    alpha: float = 0.0          # Робен: коэффициент теплоотдачи, Вт/(м²·К)
    T_inf: float = 20.0         # Робен/радиация: температура среды, °C
    emissivity: float = 0.85    # Радиация: степень черноты (0..1)

    def description(self) -> str:
        """Понятное описание для GUI — физическими терминами."""
        if self.type == BC_DIRICHLET:
            return f"Заданная T = {self.T0:g} °C"
        if self.type == BC_NEUMANN:
            if abs(self.q0) < 1e-15:
                return "Изолировано (тепло не уходит)"
            if self.q0 > 0:
                return f"Нагрев потоком q = {self.q0:g} Вт/м²"
            return f"Отвод потоком |q| = {abs(self.q0):g} Вт/м²"
        if self.type == BC_ROBIN:
            return f"Конвекция: α = {self.alpha:g}, T_среды = {self.T_inf:g} °C"
        if self.type == BC_RADIATION:
            return f"Излучение: ε = {self.emissivity:g}, T_окр = {self.T_inf:g} °C"
        return "Не задано"

    def physical_name(self) -> str:
        """Краткое имя типа для тулбара/чипа."""
        names = {BC_NONE: "—",
                 BC_DIRICHLET: "Заданная T",
                 BC_NEUMANN: ("Изоляция" if abs(self.q0) < 1e-15
                              else ("Нагрев" if self.q0 > 0 else "Отвод")),
                 BC_ROBIN: "Конвекция",
                 BC_RADIATION: "Излучение"}
        return names.get(self.type, "—")


# Пресеты конвекции — типичные α для разных условий охлаждения.
# Источник: Incropera, Fundamentals of Heat and Mass Transfer, Table 1.1.
CONVECTION_PRESETS = [
    ("— Произвольно —",                None, None),
    ("Воздух (свободная конвекция)",    8.0,  20.0),
    ("Воздух с вентилятором",          50.0,  25.0),
    ("Воздух мощный поток",           150.0,  25.0),
    ("Вода (свободная конвекция)",    500.0,  20.0),
    ("Вода с насосом",               3000.0,  20.0),
    ("Кипящая вода",                10000.0, 100.0),
    ("Масло (свободная конвекция)",    50.0,  20.0),
    ("Конденсация пара",            20000.0, 100.0),
]


# =============================================================================
# Локальные источники тепла (раздел 3.3.11 ПЗ).
# =============================================================================

@dataclass
class PointSource:
    """Точечный источник в узле сетки."""
    node_idx: int
    power: float           # Вт; знак: + нагрев, − отвод
    active: bool = True

    def description(self) -> str:
        sign = "нагрев" if self.power >= 0 else "отвод"
        return f"Точечный (узел {self.node_idx}): P = {self.power:g} Вт ({sign})"


@dataclass
class VolumeSource:
    """Объёмный источник в подобласти box или sphere."""
    shape: int             # VOLSRC_BOX | VOLSRC_SPHERE
    params: tuple          # box: (xmin,ymin,zmin,xmax,ymax,zmax); sphere: (cx,cy,cz,r)
    Q0: float              # Вт/м³
    active: bool = True

    def description(self) -> str:
        if self.shape == VOLSRC_BOX:
            xmin, ymin, zmin, xmax, ymax, zmax = self.params[:6]
            return (f"Объёмный (box): {xmin:g}..{xmax:g} × "
                    f"{ymin:g}..{ymax:g} × {zmin:g}..{zmax:g}, "
                    f"Q = {self.Q0:g} Вт/м³")
        cx, cy, cz, r = self.params[:4]
        return (f"Объёмный (sphere): центр ({cx:g}, {cy:g}, {cz:g}), "
                f"R = {r:g}, Q = {self.Q0:g} Вт/м³")


# =============================================================================
# Параметры геометрии (только параллелепипед в текущей версии GUI;
# импорт сетки управляется отдельным путём через mesh.import_msh).
# =============================================================================

@dataclass
class BoxGeometry:
    Lx: float = 0.10
    Ly: float = 0.10
    Lz: float = 0.10
    nx: int = 15
    ny: int = 15
    nz: int = 15


# =============================================================================
# Регион материала: подобласть, в которой используется свой λ и Q.
# =============================================================================

# Типы геометрии региона.
REGION_BOX = "box"
REGION_SPHERE = "sphere"


@dataclass
class MaterialRegion:
    """Подобласть с собственным материалом.

    name      — отображаемое имя (для GUI);
    lambda_   — коэффициент теплопроводности в этой области, Вт/(м·К);
    Q         — объёмная плотность источников в этой области, Вт/м³;
    shape     — REGION_BOX или REGION_SPHERE;
    params    — для REGION_BOX: (x_min, x_max, y_min, y_max, z_min, z_max);
                для REGION_SPHERE: (cx, cy, cz, radius);
    color     — цвет для подсветки в 3D (hex-строка);
    is_anisotropic — если True, lambda_ игнорируется (используется λ_x/y/z);
    lambda_x, lambda_y, lambda_z — анизотропные значения теплопроводности.
    """
    name: str
    lambda_: float
    Q: float = 0.0
    shape: str = REGION_BOX
    params: tuple = (0.0, 0.1, 0.0, 0.1, 0.0, 0.1)
    color: str = "#f0a030"
    is_anisotropic: bool = False
    lambda_x: float = 0.0
    lambda_y: float = 0.0
    lambda_z: float = 0.0

    def description(self) -> str:
        if self.shape == REGION_BOX:
            xmin, xmax, ymin, ymax, zmin, zmax = self.params
            shape_str = (f"box ({xmin:.3g}..{xmax:.3g}, "
                         f"{ymin:.3g}..{ymax:.3g}, {zmin:.3g}..{zmax:.3g})")
        elif self.shape == REGION_SPHERE:
            cx, cy, cz, r = self.params
            shape_str = f"sphere c=({cx:.3g}, {cy:.3g}, {cz:.3g}) r={r:.3g}"
        else:
            shape_str = self.shape
        if self.is_anisotropic:
            lam_str = (f"λ=({self.lambda_x:g}, {self.lambda_y:g}, "
                       f"{self.lambda_z:g}) [анизо]")
        else:
            lam_str = f"λ={self.lambda_:g}"
        return f"{self.name}: {lam_str}, Q={self.Q:g}, {shape_str}"


# =============================================================================
# Шаблоны граничных условий (Ф3.4 ТЗ).
# =============================================================================
# Все шаблоны — функции без аргументов, возвращающие dict {face_id: BC}.
# Они применимы к параллелепипедной геометрии (6 граней). Для импортированных
# сеток шаблоны тоже работают: ГУ задаются на единственную группу 0 в виде
# усреднения «по верхней грани».
# =============================================================================


def template_bottom_heat_top_cool() -> Dict[int, BoundaryCondition]:
    """Нагрев снизу 100 °C + конвекция сверху + изоляция боков."""
    return {
        4: BoundaryCondition(type=BC_DIRICHLET, T0=100.0),       # Z−
        5: BoundaryCondition(type=BC_ROBIN, alpha=25.0, T_inf=20.0),  # Z+
        0: BoundaryCondition(type=BC_NEUMANN),
        1: BoundaryCondition(type=BC_NEUMANN),
        2: BoundaryCondition(type=BC_NEUMANN),
        3: BoundaryCondition(type=BC_NEUMANN),
    }


def template_top_heat_bottom_cool() -> Dict[int, BoundaryCondition]:
    """Нагрев сверху 80 °C + охлаждение снизу 10 °C + изоляция боков."""
    return {
        5: BoundaryCondition(type=BC_DIRICHLET, T0=80.0),
        4: BoundaryCondition(type=BC_DIRICHLET, T0=10.0),
        0: BoundaryCondition(type=BC_NEUMANN),
        1: BoundaryCondition(type=BC_NEUMANN),
        2: BoundaryCondition(type=BC_NEUMANN),
        3: BoundaryCondition(type=BC_NEUMANN),
    }


def template_all_convection() -> Dict[int, BoundaryCondition]:
    """Все шесть граней — конвекция с воздухом 20 °C."""
    return {
        f: BoundaryCondition(type=BC_ROBIN, alpha=10.0, T_inf=20.0)
        for f in range(6)
    }


def template_water_cooled() -> Dict[int, BoundaryCondition]:
    """Интенсивное водяное охлаждение со всех сторон (α ≈ 500)."""
    return {
        f: BoundaryCondition(type=BC_ROBIN, alpha=500.0, T_inf=15.0)
        for f in range(6)
    }


def template_one_side_furnace() -> Dict[int, BoundaryCondition]:
    """«Печь» — горячая грань X+ при 500 °C, остальные — конвекция 20 °C."""
    bcs = {f: BoundaryCondition(type=BC_ROBIN, alpha=15.0, T_inf=20.0)
           for f in range(6)}
    bcs[1] = BoundaryCondition(type=BC_DIRICHLET, T0=500.0)  # X+
    return bcs


def template_hot_cold_walls() -> Dict[int, BoundaryCondition]:
    """Две противоположные грани X- (горячая) и X+ (холодная), остальные изолированы."""
    return {
        0: BoundaryCondition(type=BC_DIRICHLET, T0=120.0),  # X−
        1: BoundaryCondition(type=BC_DIRICHLET, T0=20.0),   # X+
        2: BoundaryCondition(type=BC_NEUMANN),
        3: BoundaryCondition(type=BC_NEUMANN),
        4: BoundaryCondition(type=BC_NEUMANN),
        5: BoundaryCondition(type=BC_NEUMANN),
    }


def template_heat_flux_bottom() -> Dict[int, BoundaryCondition]:
    """Заданный тепловой поток снизу (1 кВт/м²), конвекция сверху, изоляция боков."""
    return {
        4: BoundaryCondition(type=BC_NEUMANN, q0=+1000.0),   # положительный = нагрев (тепло внутрь)
        5: BoundaryCondition(type=BC_ROBIN, alpha=15.0, T_inf=20.0),
        0: BoundaryCondition(type=BC_NEUMANN),
        1: BoundaryCondition(type=BC_NEUMANN),
        2: BoundaryCondition(type=BC_NEUMANN),
        3: BoundaryCondition(type=BC_NEUMANN),
    }


def template_solar_panel() -> Dict[int, BoundaryCondition]:
    """Солнечная панель: интенсивный поток сверху (≈800 Вт/м²),
    свободная конвекция снизу, изоляция боков."""
    return {
        5: BoundaryCondition(type=BC_NEUMANN, q0=+800.0),    # солнечный поток (тепло внутрь)
        4: BoundaryCondition(type=BC_ROBIN, alpha=8.0, T_inf=25.0),
        0: BoundaryCondition(type=BC_NEUMANN),
        1: BoundaryCondition(type=BC_NEUMANN),
        2: BoundaryCondition(type=BC_NEUMANN),
        3: BoundaryCondition(type=BC_NEUMANN),
    }


def template_cpu_cooler() -> Dict[int, BoundaryCondition]:
    """Процессорный кулер: горячая зона снизу (центр компонента, 90 °C),
    интенсивная конвекция (α ≈ 80) на остальных гранях."""
    return {
        4: BoundaryCondition(type=BC_DIRICHLET, T0=90.0),   # горячий контакт
        5: BoundaryCondition(type=BC_ROBIN, alpha=80.0, T_inf=25.0),
        0: BoundaryCondition(type=BC_ROBIN, alpha=40.0, T_inf=25.0),
        1: BoundaryCondition(type=BC_ROBIN, alpha=40.0, T_inf=25.0),
        2: BoundaryCondition(type=BC_ROBIN, alpha=40.0, T_inf=25.0),
        3: BoundaryCondition(type=BC_ROBIN, alpha=40.0, T_inf=25.0),
    }


def template_insulated_box() -> Dict[int, BoundaryCondition]:
    """Все грани изолированы (для проверки задач с объёмным источником)."""
    # Эта конфигурация требует ненулевого источника Q или Дирихле где-то,
    # иначе задача математически некорректна. GUI это сообщит пользователю.
    return {f: BoundaryCondition(type=BC_NEUMANN) for f in range(6)}


def template_reset() -> Dict[int, BoundaryCondition]:
    """Сбросить все условия в «не задано»."""
    return {f: BoundaryCondition() for f in range(6)}


def template_building_wall_winter() -> Dict[int, BoundaryCondition]:
    """Стена здания зимой: внутри +20 °C (α=8), снаружи -25 °C (α=25),
    верх/низ/боки изолированы (плоский элемент стены)."""
    return {
        0: BoundaryCondition(type=BC_ROBIN, alpha=8.0,  T_inf=+20.0),  # X- внутри
        1: BoundaryCondition(type=BC_ROBIN, alpha=25.0, T_inf=-25.0),  # X+ снаружи
        2: BoundaryCondition(type=BC_NEUMANN),
        3: BoundaryCondition(type=BC_NEUMANN),
        4: BoundaryCondition(type=BC_NEUMANN),
        5: BoundaryCondition(type=BC_NEUMANN),
    }


def template_heat_exchanger() -> Dict[int, BoundaryCondition]:
    """Теплообменник: горячая жидкость снизу (вода кипение α=10000, T=120 °C),
    холодная сверху (вода α=3000, T=15 °C), боки изолированы."""
    return {
        4: BoundaryCondition(type=BC_ROBIN, alpha=10000.0, T_inf=120.0),  # низ
        5: BoundaryCondition(type=BC_ROBIN, alpha=3000.0,  T_inf=15.0),   # верх
        0: BoundaryCondition(type=BC_NEUMANN),
        1: BoundaryCondition(type=BC_NEUMANN),
        2: BoundaryCondition(type=BC_NEUMANN),
        3: BoundaryCondition(type=BC_NEUMANN),
    }


def template_radiation_cooling() -> Dict[int, BoundaryCondition]:
    """Излучение в открытое пространство: одна горячая грань (X-) 300°C,
    остальные излучают (ε=0.85) в окружение 25°C — типично для космической
    аппаратуры или горячих частей двигателя."""
    bcs = {f: BoundaryCondition(type=BC_RADIATION, emissivity=0.85, T_inf=25.0)
           for f in range(6)}
    bcs[0] = BoundaryCondition(type=BC_DIRICHLET, T0=300.0)
    return bcs


def template_electronics_in_chassis() -> Dict[int, BoundaryCondition]:
    """Электроника в корпусе: компонент закрытый в корпус с вентилятором.
    Низ — посадочное место (Дирихле 60°C), вверх — выдув вентилятора (α=80),
    боковины — стенки корпуса с свободной конвекцией внутри (α=10)."""
    return {
        4: BoundaryCondition(type=BC_DIRICHLET, T0=60.0),               # низ контакт
        5: BoundaryCondition(type=BC_ROBIN, alpha=80.0, T_inf=30.0),    # вверх вентилятор
        0: BoundaryCondition(type=BC_ROBIN, alpha=10.0, T_inf=30.0),
        1: BoundaryCondition(type=BC_ROBIN, alpha=10.0, T_inf=30.0),
        2: BoundaryCondition(type=BC_ROBIN, alpha=10.0, T_inf=30.0),
        3: BoundaryCondition(type=BC_ROBIN, alpha=10.0, T_inf=30.0),
    }


def template_pipe_flow() -> Dict[int, BoundaryCondition]:
    """Труба с теплоносителем: внутренняя поверхность (X-) — горячая вода
    с интенсивной конвекцией (α=3000, 90°C), внешняя (X+) — наружный воздух
    (α=10, 20°C), торцы изолированы."""
    return {
        0: BoundaryCondition(type=BC_ROBIN, alpha=3000.0, T_inf=90.0),
        1: BoundaryCondition(type=BC_ROBIN, alpha=10.0,   T_inf=20.0),
        2: BoundaryCondition(type=BC_NEUMANN),
        3: BoundaryCondition(type=BC_NEUMANN),
        4: BoundaryCondition(type=BC_NEUMANN),
        5: BoundaryCondition(type=BC_NEUMANN),
    }


def template_welding() -> Dict[int, BoundaryCondition]:
    """Сварка: интенсивный точечный нагрев (моделируется как высокий поток
    q=50 кВт/м² через грань X-), остальные — естественное охлаждение
    воздухом (α=15, 25°C)."""
    bcs = {f: BoundaryCondition(type=BC_ROBIN, alpha=15.0, T_inf=25.0)
           for f in range(6)}
    bcs[0] = BoundaryCondition(type=BC_NEUMANN, q0=+50000.0)
    return bcs


# Каталог шаблонов для GUI: (label, factory).
# Каталог шаблонов: (метка, фабрика, описание, категория-иконка).
# Категория используется для выбора схемы-миниатюры в галерее.
HEATING_TEMPLATES_FULL = [
    ("Нагрев снизу + охлаждение сверху", template_bottom_heat_top_cool,
     "Низ нагрет, верх охлаждается. Классический вертикальный градиент — "
     "например, плита на нагревателе.", "bottom_hot"),
    ("Нагрев сверху + охлаждение снизу", template_top_heat_bottom_cool,
     "Верх нагрет, низ охлаждается. Обратный градиент.", "top_hot"),
    ("Конвекция со всех сторон (α=10)", template_all_convection,
     "Свободная конвекция воздуха со всех 6 граней. Тело остывает/нагревается "
     "до температуры окружающей среды.", "all_conv"),
    ("Водяное охлаждение (α=500)", template_water_cooled,
     "Интенсивное жидкостное охлаждение со всех сторон (α=500 Вт/(м²·К)).",
     "all_conv"),
    ("Печь сбоку (X+ = 500 °C)", template_one_side_furnace,
     "Одна грань нагрета до 500 °C (стенка печи), остальные — конвекция.",
     "one_hot"),
    ("Горячая / холодная грани (X)", template_hot_cold_walls,
     "Противоположные грани при разных фиксированных температурах. "
     "Линейный профиль — эталонная задача теплопроводности.", "two_walls"),
    ("Тепловой поток снизу 1 кВт/м²", template_heat_flux_bottom,
     "Заданный тепловой поток через нижнюю грань (нагреватель), "
     "верх — конвекция.", "flux_bottom"),
    ("Солнечная панель", template_solar_panel,
     "Поглощённый солнечный поток сверху (~800 Вт/м²), конвекция и "
     "излучение с остальных граней.", "flux_top"),
    ("Процессорный кулер", template_cpu_cooler,
     "Тепловыделение снизу (контакт с кристаллом), форсированный обдув "
     "сверху (вентилятор α=80).", "cpu"),
    ("Изолированный со всех сторон", template_insulated_box,
     "Адиабатические границы — нет теплообмена. Только для проверки "
     "источников или начального состояния.", "insulated"),
    ("Стена здания зимой", template_building_wall_winter,
     "Внутри +20 °C (α=8), снаружи −25 °C (α=25). Расчёт теплопотерь "
     "через ограждающую конструкцию.", "two_walls"),
    ("Теплообменник вода/вода", template_heat_exchanger,
     "Горячий теплоноситель снизу (120 °C, α=10000), холодный сверху "
     "(15 °C, α=3000). Боковины изолированы.", "exchanger"),
    ("Радиационное охлаждение (космос)", template_radiation_cooling,
     "Горячая грань 300 °C, излучение Стефана-Больцмана (ε=0.85) с "
     "остальных граней в окружение 25 °C.", "radiation"),
    ("Электроника в корпусе", template_electronics_in_chassis,
     "Компонент на посадочном месте (60 °C снизу), обдув сверху (α=80), "
     "свободная конвекция в корпусе по бокам.", "cpu"),
    ("Труба с теплоносителем", template_pipe_flow,
     "Внутренняя поверхность — горячая вода (90 °C, α=3000), внешняя — "
     "наружный воздух (20 °C, α=10).", "two_walls"),
    ("Сварка (поток 50 кВт/м²)", template_welding,
     "Интенсивный локальный нагрев (50 кВт/м² через грань X−), "
     "естественное охлаждение остальных граней.", "flux_bottom"),
    ("— Сбросить все условия —", template_reset,
     "Снять все граничные условия (вернуть «не задано»).", "reset"),
]

# Совместимость со старым кодом: список (метка, фабрика).
HEATING_TEMPLATES = [(t[0], t[1]) for t in HEATING_TEMPLATES_FULL]


# =============================================================================
# Класс задачи: всё, что введено пользователем + результаты.
# =============================================================================

@dataclass
class Problem:
    geometry: BoxGeometry = field(default_factory=BoxGeometry)
    lambda_: float = 237.0       # алюминий по умолчанию
    Q: float = 0.0
    bcs: Dict[int, BoundaryCondition] = field(default_factory=lambda: {
        f: BoundaryCondition() for f in range(6)
    })

    # --- Теплофизические свойства для нестационарной задачи ----------------
    rho: float = 0.0   # плотность кг/м³ (0 = не задано, дефолт 1000)
    cp:  float = 0.0   # удельная теплоёмкость Дж/(кг·К)

    # --- Глобальная анизотропия материала (λ разный по осям X/Y/Z) ---------
    is_anisotropic: bool = False
    lambda_x: float = 0.0
    lambda_y: float = 0.0
    lambda_z: float = 0.0

    # Имя выбранного материала (для отображения в UI и отчётах).
    material_name: str = ""

    # Точки наблюдения (виртуальные термопары): список (x, y, z).
    # Используются для записи T(t) в нестационарном режиме.
    observation_points: list = field(default_factory=list)

    # --- Нелинейная λ(T) ---------------------------------------------------
    # Если задана, λ пересчитывается через Picard-итерации.
    # Сигнатура: lambda_T_func(T_mean: float) -> float (Вт/(м·К))
    lambda_T_func: Optional[callable] = None
    # Допустимая относительная погрешность Picard для λ(T).
    picard_tol: float = 1e-4
    picard_max_iter: int = 20

    # --- Локальные источники (раздел 3.3.11 ПЗ) -----------------------------
    point_sources: list = field(default_factory=list)   # list[PointSource]
    volume_sources: list = field(default_factory=list)  # list[VolumeSource]

    # --- Регионы материалов (раздел 3.3.x ПЗ — несколько материалов в детали).
    material_regions: list = field(default_factory=list)  # list[MaterialRegion]

    # --- Импортированная сетка (если задана — используется вместо generate_box).
    # external_nodes:  (N, 3) float64;
    # external_elements: (Ne, 4) int32;
    # external_bnd_nodes: (Nf, 3) int32; external_bnd_face_ids: (Nf,) int32.
    external_nodes: Optional[np.ndarray] = None
    external_elements: Optional[np.ndarray] = None
    external_bnd_nodes: Optional[np.ndarray] = None
    external_bnd_face_ids: Optional[np.ndarray] = None

    # --- Результаты расчёта (заполняются после solve) -----------------------
    nodes: Optional[np.ndarray] = None        # (N, 3)
    elements: Optional[np.ndarray] = None     # (Ne, 4)
    T: Optional[np.ndarray] = None            # (N,)
    flux: Optional[np.ndarray] = None         # (N, 3)
    info: Optional[SolverInfo] = None

    # =========================================================================
    # Пайплайн.
    # =========================================================================

    def has_external_mesh(self) -> bool:
        return self.external_nodes is not None and self.external_elements is not None

    def build_mesh_in_core(self, bridge: CoreBridge) -> None:
        if self.has_external_mesh():
            bridge.load_mesh(
                self.external_nodes,
                self.external_elements,
                self.external_bnd_nodes
                if self.external_bnd_nodes is not None
                else np.empty((0, 3), dtype=np.int32),
                self.external_bnd_face_ids
                if self.external_bnd_face_ids is not None
                else np.empty((0,), dtype=np.int32),
            )
        else:
            g = self.geometry
            bridge.generate_box(
                0.0, g.Lx, 0.0, g.Ly, 0.0, g.Lz, g.nx, g.ny, g.nz)
        self.nodes = bridge.get_nodes()
        self.elements = bridge.get_elements()

    def push_to_core(self, bridge: CoreBridge,
                      T_estimate: Optional[np.ndarray] = None) -> None:
        """Передать материал, ГУ, источники и регионы материалов в ядро.

        T_estimate — текущая оценка поля T (для линеаризации радиационного ГУ).
        """
        # Глобальный материал: анизотропный или изотропный.
        if self.is_anisotropic:
            bridge.set_material_anisotropic(
                self.lambda_x, self.lambda_y, self.lambda_z, self.Q)
        else:
            bridge.set_material(self.lambda_, self.Q)
        # Плотность и теплоёмкость (для нестационарной задачи).
        if self.rho > 0 and self.cp > 0:
            try:
                bridge.set_thermal_capacity(self.rho, self.cp)
            except Exception:
                pass  # старый бинарник без этой функции — не критично

        # Регионы материалов: каждый регион — это (λ, Q) + bbox/sphere.
        # 1. Очищаем все предыдущие назначения.
        # 2. Создаём материалы по очереди (получая 1-based id).
        # 3. Назначаем каждому соответствующему региону его id.
        bridge.clear_materials()
        for region in self.material_regions:
            if region.is_anisotropic:
                mat_id = bridge.add_material_anisotropic(
                    region.lambda_x, region.lambda_y, region.lambda_z, region.Q)
            else:
                mat_id = bridge.add_material(region.lambda_, region.Q)
            if region.shape == REGION_BOX:
                xmin, xmax, ymin, ymax, zmin, zmax = region.params
                bridge.assign_material_in_box(
                    mat_id, xmin, xmax, ymin, ymax, zmin, zmax)
            elif region.shape == REGION_SPHERE:
                cx, cy, cz, r = region.params
                bridge.assign_material_in_sphere(mat_id, cx, cy, cz, r)

        from fem3d.core_bridge import (BC_DIRICHLET as _BD, BC_NEUMANN as _BN,
                                          BC_ROBIN as _BR, BC_RADIATION as _BRD,
                                          STEFAN_BOLTZMANN)

        # Оценки температуры на гранях (для линеаризации радиации).
        # Работает и для box, и для внешней сетки.
        T_face_avg = {}
        if T_estimate is not None:
            for fid in range(6):
                T_face_avg[fid] = self._mean_T_on_face(fid, T_estimate)

        for face_id in range(6):
            bc = self.bcs[face_id]
            if bc.type == _BRD:
                # Линеаризация радиации: ε σ (T⁴−T_ext⁴) = α_rad·(T_s−T_ext),
                # где α_rad = ε σ (T_s+T_ext)(T_s²+T_ext²).  T в Кельвинах.
                T_ext_K = bc.T_inf + 273.15
                T_s_K = (T_face_avg.get(face_id, bc.T_inf) + 273.15
                         if face_id in T_face_avg else T_ext_K)
                alpha_rad = (bc.emissivity * STEFAN_BOLTZMANN
                             * (T_s_K + T_ext_K) * (T_s_K**2 + T_ext_K**2))
                bridge.set_bc(face_id, _BR,
                              T0=0.0, q0=0.0,
                              alpha=alpha_rad, T_inf=bc.T_inf)
            elif bc.type == _BN:
                # Конвенция: q0>0 = нагрев. В ядре наоборот.
                bridge.set_bc(face_id, _BN,
                              T0=bc.T0, q0=-bc.q0,
                              alpha=bc.alpha, T_inf=bc.T_inf)
            else:
                bridge.set_bc(face_id, bc.type,
                              T0=bc.T0, q0=bc.q0,
                              alpha=bc.alpha, T_inf=bc.T_inf)

        # Источники: всегда сначала очищаем, потом добавляем активные.
        bridge.clear_sources()
        for ps in self.point_sources:
            if ps.active:
                bridge.add_point_source(ps.node_idx, ps.power)
        for vs in self.volume_sources:
            if not vs.active:
                continue
            if vs.shape == VOLSRC_BOX:
                xmin, ymin, zmin, xmax, ymax, zmax = vs.params[:6]
                bridge.add_volume_source_box(
                    xmin, ymin, zmin, xmax, ymax, zmax, vs.Q0)
            elif vs.shape == VOLSRC_SPHERE:
                cx, cy, cz, r = vs.params[:4]
                bridge.add_volume_source_sphere(cx, cy, cz, r, vs.Q0)

    def solve(self, bridge: CoreBridge,
              tol: float = 1e-8, max_iter: int = 5000,
              progress_callback=None) -> SolverInfo:
        """Полный цикл: материал + ГУ → CG → результаты.

        Если есть радиационные ГУ или задана нелинейная λ(T), выполняется
        Picard-итерация по нелинейной части.
        Сходимость по T: max|ΔT| < 0.01 °C между внешними итерациями.
        """
        has_lambda_T = self.lambda_T_func is not None
        has_radiation = self.has_radiation_bc()
        # Линейная задача — один прогон.
        if not has_radiation and not has_lambda_T:
            self.push_to_core(bridge)
            info = bridge.solve(tol=tol, max_iter=max_iter,
                                progress_callback=progress_callback)
            self.T = bridge.get_temperature()
            self.flux = bridge.compute_fluxes()
            self.info = info
            return info

        # Нелинейная — Picard-итерация.
        T_prev = None
        info = None
        original_lambda = self.lambda_
        try:
            for outer in range(self.picard_max_iter):
                # Обновим λ от T_mean текущей оценки.
                if has_lambda_T and T_prev is not None:
                    T_mean = float(np.mean(T_prev))
                    self.lambda_ = float(self.lambda_T_func(T_mean))
                self.push_to_core(bridge, T_estimate=T_prev)
                info = bridge.solve(tol=tol, max_iter=max_iter,
                                    progress_callback=progress_callback)
                T_now = bridge.get_temperature()
                if T_prev is not None:
                    diff = float(np.max(np.abs(T_now - T_prev)))
                    if diff < 0.01:
                        break
                T_prev = T_now.copy()
        finally:
            # Возвращаем оригинальное значение λ (не сохраняем последнее).
            if has_lambda_T:
                self.lambda_ = original_lambda
        self.T = T_prev
        self.flux = bridge.compute_fluxes()
        self.info = info
        return info

    def solve_transient(self, bridge: CoreBridge,
                          t_end: float, dt: float, T_init: float = 0.0,
                          n_save: int = 50,
                          tol: float = 1e-8, max_iter: int = 5000):
        """Нестационарная задача через C++ ядро.

        Возвращает (times, T_history) аналогично CoreBridge.solve_transient(),
        дополнительно сохраняет T = T_history[-1], info = последнюю инфу решателя.
        """
        if self.rho <= 0 or self.cp <= 0:
            raise ValueError("Для нестационарной задачи нужно задать ρ и c_p "
                             "(rho, cp полей Problem)")
        self.push_to_core(bridge)
        bridge.set_thermal_capacity(self.rho, self.cp)
        times, T_hist = bridge.solve_transient(t_end, dt, T_init=T_init,
                                                 n_save=n_save,
                                                 tol=tol, max_iter=max_iter)
        # Финальное состояние — последний снимок.
        self.T = T_hist[-1].copy()
        self.flux = bridge.compute_fluxes()
        self.info = bridge.solver_info()
        return times, T_hist

    def _mean_T_on_face(self, face_id: int, T: np.ndarray) -> float:
        """Средняя T на грани face_id. Работает и для box, и для внешней сетки.

        Для внешней сетки усредняет T по узлам граничных треугольников,
        отнесённых к данному face_id (через external_bnd_*). Для box —
        по узлам, лежащим в соответствующей координатной плоскости.
        """
        if self.nodes is None or T is None:
            return 20.0
        # Внешняя сетка: усреднение по узлам граничных треугольников face_id.
        if (self.has_external_mesh()
                and self.external_bnd_nodes is not None
                and self.external_bnd_face_ids is not None):
            sel = self.external_bnd_face_ids == face_id
            if not np.any(sel):
                return 20.0
            tri_nodes = np.unique(self.external_bnd_nodes[sel].ravel())
            if tri_nodes.size == 0:
                return 20.0
            return float(T[tri_nodes].mean())
        # Box-геометрия: узлы в координатной плоскости грани.
        g = self.geometry
        coord_axis = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}[face_id]
        coord_val = {0: 0.0, 1: g.Lx, 2: 0.0, 3: g.Ly, 4: 0.0, 5: g.Lz}[face_id]
        coords = self.nodes[:, coord_axis]
        tol_d = 1e-6 * max(g.Lx, g.Ly, g.Lz)
        mask = np.abs(coords - coord_val) < tol_d
        if not np.any(mask):
            return 20.0
        return float(T[mask].mean())

    def has_radiation_bc(self) -> bool:
        from fem3d.core_bridge import BC_RADIATION
        return any(bc.type == BC_RADIATION for bc in self.bcs.values())

    # =========================================================================
    # Утилиты для GUI.
    # =========================================================================

    def temperature_range(self) -> tuple[float, float]:
        if self.T is None or self.T.size == 0:
            return (0.0, 0.0)
        return float(self.T.min()), float(self.T.max())

    def _face_id_for_triangles(self, faces: np.ndarray) -> np.ndarray:
        """Сопоставить каждому граничному треугольнику его face_id (0..5).

        Для внешней сетки (сложные фигуры) — через словарь по отсортированным
        узлам из external_bnd_nodes/external_bnd_face_ids.
        Для box — по координатам центроида грани.
        """
        nf = faces.shape[0]
        result = np.full(nf, -1, dtype=np.int32)
        if (self.has_external_mesh()
                and self.external_bnd_nodes is not None
                and self.external_bnd_face_ids is not None):
            # Словарь: отсортированный кортеж узлов → face_id.
            lookup = {}
            ext = self.external_bnd_nodes
            ids = self.external_bnd_face_ids
            for i in range(ext.shape[0]):
                key = tuple(sorted((int(ext[i, 0]), int(ext[i, 1]),
                                     int(ext[i, 2]))))
                lookup[key] = int(ids[i])
            for t in range(nf):
                key = tuple(sorted((int(faces[t, 0]), int(faces[t, 1]),
                                     int(faces[t, 2]))))
                result[t] = lookup.get(key, -1)
            return result
        # Box: по координатам центроида.
        g = self.geometry
        c = (self.nodes[faces[:, 0]] + self.nodes[faces[:, 1]]
             + self.nodes[faces[:, 2]]) / 3.0
        eps = 1e-6 * max(g.Lx, g.Ly, g.Lz)
        result[np.abs(c[:, 0]) < eps] = 0
        result[np.abs(c[:, 0] - g.Lx) < eps] = 1
        result[np.abs(c[:, 1]) < eps] = 2
        result[np.abs(c[:, 1] - g.Ly) < eps] = 3
        result[np.abs(c[:, 2]) < eps] = 4
        result[np.abs(c[:, 2] - g.Lz) < eps] = 5
        return result

    def energy_balance(self) -> Optional[dict]:
        """Проверка энергобаланса для стационарной задачи:
            тепло, выделяемое внутри (объёмные + локальные источники)
            должно равняться суммарному потоку через границу.
        Возвращает dict с полями:
            q_in   — Вт, тепло, входящее в тело через границу;
            q_out  — Вт, выходящее;
            q_gen  — Вт, генерируемое внутри (Q*V + точечные + объёмные шары);
            net    — Вт, q_gen - (q_out - q_in);
            rel_err — относительная ошибка |net| / max(|q_gen|, |q_out|);
        None если расчёт ещё не выполнен.
        """
        if self.T is None or self.nodes is None or self.elements is None:
            return None
        if self.flux is None:
            return None

        # Объём каждого тетраэдра и центроиды (нужны и для генерации).
        tets = self.elements.astype(np.int64)
        p0 = self.nodes[tets[:, 0]]
        p1 = self.nodes[tets[:, 1]]
        p2 = self.nodes[tets[:, 2]]
        p3 = self.nodes[tets[:, 3]]
        vol = np.abs(np.einsum("ij,ij->i",
                                p1 - p0,
                                np.cross(p2 - p0, p3 - p0))) / 6.0
        total_volume = float(vol.sum())
        centroids = 0.25 * (p0 + p1 + p2 + p3)   # (Ne, 3)

        # -------------------------------------------------------------------
        # Генерация внутри тела — считается ПОЭЛЕМЕНТНО, в точности повторяя
        # логику C++ assemble(). Это гарантирует, что проверяемая генерация
        # равна реально собранной в правую часть F, и покрывает: глобальный Q,
        # объёмный Q регионов материалов, объёмные источники (и box, и sphere).
        # -------------------------------------------------------------------
        from fem3d.core_bridge import VOLSRC_BOX, VOLSRC_SPHERE
        from fem3d.problem import REGION_BOX, REGION_SPHERE

        # Q на элемент: по умолчанию глобальный self.Q; перекрывается регионом,
        # если центроид внутри (последний регион выигрывает — как assign в ядре).
        Q_per_elem = np.full(tets.shape[0], float(self.Q), dtype=np.float64)
        for region in self.material_regions:
            rq = float(getattr(region, "Q", 0.0))
            if region.shape == REGION_BOX:
                xmin, xmax, ymin, ymax, zmin, zmax = region.params
                m = ((centroids[:, 0] >= xmin) & (centroids[:, 0] <= xmax) &
                     (centroids[:, 1] >= ymin) & (centroids[:, 1] <= ymax) &
                     (centroids[:, 2] >= zmin) & (centroids[:, 2] <= zmax))
            elif region.shape == REGION_SPHERE:
                cx, cy, cz, r = region.params[:4]
                d2 = ((centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2
                      + (centroids[:, 2] - cz) ** 2)
                m = d2 <= r * r
            else:
                continue
            Q_per_elem[m] = rq
        q_gen_volume = float(np.sum(Q_per_elem * vol))

        # Точечные источники — ровно как в assemble (F[node] += power).
        q_gen_point = float(sum(ps.power for ps in self.point_sources))

        # Объёмные источники: Q_loc·V_e, суммарно по всем элементам, для box и
        # sphere одинаково (центроид внутри источника). Точное совпадение с F.
        q_gen_vol_src = 0.0
        for vs in self.volume_sources:
            if vs.shape == VOLSRC_BOX:
                x0, y0, z0, x1, y1, z1 = vs.params[:6]
                m = ((centroids[:, 0] >= x0) & (centroids[:, 0] <= x1) &
                     (centroids[:, 1] >= y0) & (centroids[:, 1] <= y1) &
                     (centroids[:, 2] >= z0) & (centroids[:, 2] <= z1))
            elif vs.shape == VOLSRC_SPHERE:
                cx, cy, cz, r = vs.params[:4]
                d2 = ((centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2
                      + (centroids[:, 2] - cz) ** 2)
                m = d2 <= r * r
            else:
                continue
            q_gen_vol_src += float(vs.Q0) * float(np.sum(vol[m]))

        q_gen = q_gen_volume + q_gen_point + q_gen_vol_src

        # Поток через границу: q · n dS на каждом граничном треугольнике.
        # Извлекаем граничные грани ВМЕСТЕ с противоположной вершиной
        # породившего тетраэдра — это даёт надёжную внешнюю нормаль даже
        # для невыпуклых тел (тор, L-балка, U-канал, спираль).
        from fem3d.mesh import _extract_surface_faces_with_opposite
        faces, opp_vertices = _extract_surface_faces_with_opposite(self.elements)
        if faces is None or faces.shape[0] == 0:
            return None

        n0 = self.nodes[faces[:, 0]]
        n1 = self.nodes[faces[:, 1]]
        n2 = self.nodes[faces[:, 2]]
        normals_x2 = np.cross(n1 - n0, n2 - n0)  # 2A * n
        face_areas = 0.5 * np.linalg.norm(normals_x2, axis=1)
        face_normals = normals_x2 / np.maximum(2.0 * face_areas[:, None], 1e-30)
        # Внешняя нормаль направлена ОТ противоположной вершины тетраэдра.
        opp = self.nodes[opp_vertices]               # (Nf, 3)
        face_centers = (n0 + n1 + n2) / 3.0
        # Вектор от противоположной вершины к центру грани = «наружу».
        outward_dir = face_centers - opp
        inward = np.einsum("ij,ij->i", face_normals, outward_dir) < 0
        face_normals[inward] *= -1

        # Сопоставляем каждую граничную грань с её face_id и ГУ.
        # Затем считаем поток ПРЯМО из граничного условия — это намного
        # точнее, чем восстановление из узловых градиентов (производная в
        # МКЭ сходится медленнее самого поля T, особенно на криволинейной
        # поверхности).
        from fem3d.core_bridge import (BC_DIRICHLET as _BD, BC_NEUMANN as _BN,
                                          BC_ROBIN as _BR, BC_RADIATION as _BRD,
                                          STEFAN_BOLTZMANN)
        face_ids_per_tri = self._face_id_for_triangles(faces)

        T_face = (self.T[faces[:, 0]] + self.T[faces[:, 1]]
                  + self.T[faces[:, 2]]) / 3.0  # средняя T на грани

        # q_n > 0 — наружу (отвод тепла), < 0 — внутрь (нагрев).
        q_n_bc = np.zeros(faces.shape[0], dtype=np.float64)
        # Для граней Дирихле условие не даёт прямой формулы потока —
        # для них используем восстановленный поток (запасной вариант).
        dirichlet_mask = np.zeros(faces.shape[0], dtype=bool)

        for tri_idx in range(faces.shape[0]):
            fid = int(face_ids_per_tri[tri_idx])
            bc = self.bcs.get(fid) if fid >= 0 else None
            if bc is None:
                q_n_bc[tri_idx] = 0.0
            elif bc.type == _BN:
                # Нейман: заданный поток q0. Конвенция: q0 > 0 = нагрев
                # (тепло входит в тело), поэтому наружу = -q0.
                q_n_bc[tri_idx] = -bc.q0
            elif bc.type == _BR:
                # Конвекция: q_наружу = α·(T_wall − T_inf).
                q_n_bc[tri_idx] = bc.alpha * (T_face[tri_idx] - bc.T_inf)
            elif bc.type == _BRD:
                # Излучение: q = εσ(T_wall⁴ − T_env⁴), T в Кельвинах.
                Tw = T_face[tri_idx] + 273.15
                Te = bc.T_inf + 273.15
                eps = getattr(bc, "emissivity", 0.0)
                q_n_bc[tri_idx] = eps * STEFAN_BOLTZMANN * (Tw**4 - Te**4)
            elif bc.type == _BD:
                dirichlet_mask[tri_idx] = True
            else:
                q_n_bc[tri_idx] = 0.0

        # Для Дирихле-граней — точный поток через градиент P1 породившего
        # тетраэдра. Для линейного элемента ∇T постоянен и вычисляется точно,
        # что намного надёжнее усреднения узловых потоков.
        if np.any(dirichlet_mask):
            # Компоненты λ (для анизотропного материала — покомпонентно).
            if self.is_anisotropic:
                lam_vec = np.array([self.lambda_x, self.lambda_y, self.lambda_z])
            else:
                lam_vec = np.array([self.lambda_, self.lambda_, self.lambda_])
            d_idx = np.flatnonzero(dirichlet_mask)
            for t in d_idx:
                # 4 узла тетраэдра: 3 грани + противоположная вершина.
                ia, ib, ic = int(faces[t, 0]), int(faces[t, 1]), int(faces[t, 2])
                io = int(opp_vertices[t])
                P = self.nodes[[ia, ib, ic, io]]
                Tv = self.T[[ia, ib, ic, io]]
                M = P[1:] - P[0]
                dT = Tv[1:] - Tv[0]
                try:
                    grad = np.linalg.solve(M, dT)
                except np.linalg.LinAlgError:
                    grad = np.zeros(3)
                # Анизотропный закон Фурье: q_i = -λ_i · ∂T/∂x_i.
                q_vec = -lam_vec * grad
                q_n_bc[t] = float(np.dot(q_vec, face_normals[t]))

        # Интегрируем по площади. flux_signed > 0 — наружу, < 0 — внутрь.
        flux_signed = q_n_bc * face_areas
        flux_out = float(np.sum(np.maximum(flux_signed, 0.0)))
        flux_in  = float(np.sum(-np.minimum(flux_signed, 0.0)))
        # Алгебраическая сумма = суммарный отвод наружу.
        net_out = float(np.sum(flux_signed))
        # Полный «оборот» тепла через границу (для масштаба ошибки).
        throughput = float(np.sum(np.abs(flux_signed)))

        # Закон сохранения: суммарный отвод наружу = генерация внутри.
        net = net_out - q_gen
        scale = max(abs(q_gen), throughput * 0.5)
        if scale < 1e-9:
            rel_err = 0.0
        else:
            rel_err = abs(net) / scale
        return {
            "q_in_W":     flux_in,
            "q_out_W":    flux_out,
            "q_gen_W":    q_gen,
            "net_out_W":  net_out,
            "imbalance_W": net,
            "rel_err":     rel_err,
        }

    def hot_spot(self) -> Optional[tuple[int, float, float, float]]:
        """(индекс узла, x, y, z) узла с максимальной температурой, если есть."""
        if self.T is None or self.nodes is None:
            return None
        idx = int(np.argmax(self.T))
        x, y, z = self.nodes[idx]
        return idx, float(x), float(y), float(z)
