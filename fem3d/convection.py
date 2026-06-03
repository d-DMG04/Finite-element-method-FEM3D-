# -*- coding: utf-8 -*-
"""
fem3d.convection
================

Конвективный теплообмен при ОБТЕКАНИИ тела потоком (вынужденная конвекция).

Этот модуль реализует «прямую» постановку, которую обычно требуют в курсовых
и дипломных работах по теплопередаче:

    скорость и направление потока  →  число Рейнольдса Re
                                   →  число Нуссельта   Nu  (по эмпирической
                                                            корреляции для
                                                            формы тела)
                                   →  коэффициент теплоотдачи  h = Nu·λ_возд/L
                                   →  граничное условие конвекции (Робен) α = h
                                   →  тепловой поток  Q = h·A·(T_s − T∞)

В отличие от уже имевшейся функции postprocess.compute_nusselt (которая
ВОССТАНАВЛИВАЕТ фактический h из готового поля T), здесь h вычисляется ЗАРАНЕЕ,
по параметрам набегающего потока, и затем используется как ГУ при расчёте.

Физические основы (учебники: Incropera «Fundamentals of Heat and Mass
Transfer», Цветков/Григорьев «Тепломассообмен»):

    Re = U·L / ν              — число Рейнольдса (режим течения)
    Pr = ν / a = μ·c_p / λ    — число Прандтля (свойство среды)
    Nu = h·L / λ_среды        — число Нуссельта (безразмерная теплоотдача)
    h  = Nu·λ_среды / L       — коэффициент теплоотдачи, Вт/(м²·К)
    Q  = h·A·(T_s − T∞)       — конвективный тепловой поток, Вт

Все температуры — в °C (как и во всём проекте); внутри, где нужно, переводятся
в Кельвины.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. Теплофизические свойства воздуха (сухой, 1 атм).
# =============================================================================
# Таблица свойств сухого воздуха при атмосферном давлении.
# Источник: Incropera, Fundamentals of Heat and Mass Transfer, Table A.4.
# Столбцы: T[K], ρ[кг/м³], c_p[Дж/(кг·К)], μ·1e7[Па·с], ν·1e6[м²/с],
#          λ·1e3[Вт/(м·К)], a·1e6[м²/с] (темп.проводность), Pr.
_AIR_TABLE = np.array([
    # T(K)   rho      cp      mu*1e7   nu*1e6   k*1e3    a*1e6    Pr
    [250.0, 1.3947, 1006.0,  159.6,   11.44,   22.3,    15.9,    0.720],
    [300.0, 1.1614, 1007.0,  184.6,   15.89,   26.3,    22.5,    0.707],
    [350.0, 0.9950, 1009.0,  208.2,   20.92,   30.0,    29.9,    0.700],
    [400.0, 0.8711, 1014.0,  230.1,   26.41,   33.8,    38.3,    0.690],
    [450.0, 0.7740, 1021.0,  250.7,   32.39,   37.3,    47.2,    0.686],
    [500.0, 0.6964, 1030.0,  270.1,   38.79,   40.7,    56.7,    0.684],
    [550.0, 0.6329, 1040.0,  288.4,   45.57,   43.9,    66.7,    0.683],
    [600.0, 0.5804, 1051.0,  305.8,   52.69,   46.9,    76.9,    0.685],
])


@dataclass
class FluidProperties:
    """Свойства среды (воздуха) при температуре плёнки T_film."""
    name: str = "воздух"
    T_film_C: float = 20.0      # температура плёнки = (T_s + T∞)/2, °C
    rho: float = 1.2            # плотность, кг/м³
    cp: float = 1007.0          # теплоёмкость, Дж/(кг·К)
    mu: float = 1.82e-5         # динамическая вязкость, Па·с
    nu: float = 1.5e-5          # кинематическая вязкость, м²/с
    k: float = 0.026            # теплопроводность среды, Вт/(м·К)
    a: float = 2.1e-5           # температуропроводность, м²/с
    Pr: float = 0.71            # число Прандтля
    beta: float = 1.0 / 293.15  # коэф. объёмного расширения, 1/К (идеал. газ 1/T)

    def summary(self) -> str:
        return (f"{self.name} при T_плёнки={self.T_film_C:.1f} °C: "
                f"ρ={self.rho:.4f} кг/м³, ν={self.nu:.3e} м²/с, "
                f"λ={self.k:.4f} Вт/(м·К), Pr={self.Pr:.3f}")


def air_properties(T_film_C: float) -> FluidProperties:
    """Свойства сухого воздуха при заданной температуре плёнки (°C).

    Линейная интерполяция по табличным данным Incropera (Table A.4).
    За пределами таблицы — насыщение к крайним значениям.
    """
    T_K = float(T_film_C) + 273.15
    Ts = _AIR_TABLE[:, 0]
    T_clamped = float(np.clip(T_K, Ts[0], Ts[-1]))

    def interp(col):
        return float(np.interp(T_clamped, Ts, _AIR_TABLE[:, col]))

    return FluidProperties(
        name="воздух",
        T_film_C=float(T_film_C),
        rho=interp(1),
        cp=interp(2),
        mu=interp(3) * 1e-7,
        nu=interp(4) * 1e-6,
        k=interp(5) * 1e-3,
        a=interp(6) * 1e-6,
        Pr=interp(7),
        beta=1.0 / T_clamped,   # идеальный газ: β ≈ 1/T
    )


# =============================================================================
# 2. Числа Рейнольдса, Нуссельта (вынужденная конвекция при обтекании).
# =============================================================================

# Поддерживаемые формы обтекаемого тела.
SHAPE_PLATE = "plate"        # плоская пластина (продольное обтекание)
SHAPE_CYLINDER = "cylinder"  # цилиндр в поперечном потоке
SHAPE_SPHERE = "sphere"      # сфера / шар
SHAPE_CUBE = "cube"          # куб/параллелепипед (приближённо как пластина)

SHAPE_NAMES = {
    SHAPE_PLATE:    "плоская пластина (продольное обтекание)",
    SHAPE_CYLINDER: "цилиндр в поперечном потоке",
    SHAPE_SPHERE:   "сфера (шар)",
    SHAPE_CUBE:     "куб/параллелепипед (≈ пластина)",
}


def reynolds(speed: float, char_length: float, fluid: FluidProperties) -> float:
    """Число Рейнольдса Re = U·L/ν."""
    if char_length <= 0 or fluid.nu <= 0:
        return 0.0
    return float(speed) * float(char_length) / fluid.nu


def nusselt_forced(Re: float, Pr: float, shape: str = SHAPE_PLATE) -> Tuple[float, str]:
    """Число Нуссельта для вынужденной конвекции при обтекании.

    Возвращает (Nu, описание режима/корреляции).

    Применяемые корреляции:
      • Пластина (продольно):
          ламинар (Re<5·10⁵):     Nu = 0.664·Re^0.5·Pr^(1/3)
          турбулент./смешан.:     Nu = (0.037·Re^0.8 − 871)·Pr^(1/3)
      • Цилиндр (поперечно) — корреляция Черчилля–Бернстайна:
          Nu = 0.3 + 0.62·Re^0.5·Pr^(1/3)/[1+(0.4/Pr)^(2/3)]^0.25 ·
               [1+(Re/282000)^(5/8)]^(4/5)
      • Сфера — корреляция Уитакера:
          Nu = 2 + (0.4·Re^0.5 + 0.06·Re^(2/3))·Pr^0.4
      • Куб — приближённо как пластина.
    """
    Re = max(0.0, float(Re))
    Pr = max(1e-6, float(Pr))
    Pr13 = Pr ** (1.0 / 3.0)

    if shape in (SHAPE_PLATE, SHAPE_CUBE):
        Re_crit = 5.0e5
        if Re < Re_crit:
            Nu = 0.664 * math.sqrt(Re) * Pr13
            regime = (f"ламинарный пограничный слой (Re={Re:.3g} < 5·10⁵), "
                      "пластина: Nu = 0.664·Re^0.5·Pr^(1/3)")
        else:
            # Смешанный пограничный слой (ламинар. участок + турбулент.).
            Nu = (0.037 * Re ** 0.8 - 871.0) * Pr13
            regime = (f"смешанный/турбулентный слой (Re={Re:.3g} ≥ 5·10⁵), "
                      "пластина: Nu = (0.037·Re^0.8 − 871)·Pr^(1/3)")
        return float(max(Nu, 0.0)), regime

    if shape == SHAPE_CYLINDER:
        # Черчилль–Бернстайн (весь диапазон Re·Pr > 0.2).
        num = 0.62 * math.sqrt(Re) * Pr13
        den = (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25
        tail = (1.0 + (Re / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
        Nu = 0.3 + (num / den) * tail
        regime = ("цилиндр, корреляция Черчилля–Бернстайна "
                  f"(Re={Re:.3g})")
        return float(Nu), regime

    if shape == SHAPE_SPHERE:
        # Уитакер (3.5 < Re < 7.6·10⁴). Отношение вязкостей μ/μ_s ≈ 1.
        Nu = 2.0 + (0.4 * math.sqrt(Re) + 0.06 * Re ** (2.0 / 3.0)) * Pr ** 0.4
        regime = f"сфера, корреляция Уитакера (Re={Re:.3g})"
        return float(Nu), regime

    # По умолчанию — пластина.
    Nu = 0.664 * math.sqrt(Re) * Pr13
    return float(Nu), "пластина (по умолчанию)"


def nusselt_natural_vertical_plate(Ra: float, Pr: float) -> Tuple[float, str]:
    """Свободная (естественная) конвекция, вертикальная пластина.

    Корреляция Черчилля–Чу (для всего диапазона Ra):
        Nu = {0.825 + 0.387·Ra^(1/6) /
              [1+(0.492/Pr)^(9/16)]^(8/27)}²
    Используется как ОЦЕНКА «без обдува» для сравнения с вынужденной.
    """
    Ra = max(0.0, float(Ra))
    Pr = max(1e-6, float(Pr))
    denom = (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    Nu = (0.825 + 0.387 * Ra ** (1.0 / 6.0) / denom) ** 2
    return float(Nu), "вертикальная пластина, Черчилль–Чу (свободная конвекция)"


def rayleigh(delta_T: float, char_length: float, fluid: FluidProperties,
             g: float = 9.81) -> float:
    """Число Рэлея Ra = g·β·ΔT·L³ / (ν·a) — движущая сила свободной конвекции."""
    if fluid.nu <= 0 or fluid.a <= 0:
        return 0.0
    return (g * fluid.beta * abs(float(delta_T)) * char_length ** 3
            / (fluid.nu * fluid.a))


def heat_transfer_coefficient(Nu: float, char_length: float,
                              fluid: FluidProperties) -> float:
    """Коэффициент теплоотдачи h = Nu·λ_среды / L, Вт/(м²·К)."""
    if char_length <= 0:
        return 0.0
    return float(Nu) * fluid.k / float(char_length)


# =============================================================================
# 3. Площадь поверхности фигур и проекция на направление потока.
# =============================================================================

# Соответствие face_id → ось и внешняя нормаль (для box-геометрии).
_FACE_AXIS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
_FACE_SIGN = {0: -1.0, 1: +1.0, 2: -1.0, 3: +1.0, 4: -1.0, 5: +1.0}
_FACE_LABEL = {0: "X−", 1: "X+", 2: "Y−", 3: "Y+", 4: "Z−", 5: "Z+"}


def _triangle_areas(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Площади треугольников по координатам вершин."""
    p0 = nodes[tris[:, 0]]
    p1 = nodes[tris[:, 1]]
    p2 = nodes[tris[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _face_normals_box() -> Dict[int, np.ndarray]:
    n = {}
    for fid in range(6):
        v = np.zeros(3)
        v[_FACE_AXIS[fid]] = _FACE_SIGN[fid]
        n[fid] = v
    return n


def surface_areas(problem) -> dict:
    """Площадь поверхности фигуры: по каждой грани (face_id) и суммарно.

    Работает и для параметрического box, и для импортированной сетки.

    Возвращает словарь:
        {
          "per_face":   {face_id: площадь, м²},
          "total":      суммарная площадь поверхности, м²,
          "labels":     {face_id: 'X−'...},
        }
    """
    result = {"per_face": {}, "labels": dict(_FACE_LABEL)}

    # Импортированная сетка: суммируем площади граничных треугольников по face_id.
    if (problem.has_external_mesh()
            and problem.external_bnd_nodes is not None
            and problem.external_bnd_face_ids is not None
            and problem.nodes is not None):
        tris = problem.external_bnd_nodes
        fids = problem.external_bnd_face_ids
        areas = _triangle_areas(problem.nodes, tris)
        per_face = {}
        for fid in range(6):
            sel = fids == fid
            per_face[fid] = float(areas[sel].sum()) if np.any(sel) else 0.0
        result["per_face"] = per_face
        result["total"] = float(areas.sum())
        return result

    # Box-геометрия: прямоугольные грани.
    g = problem.geometry
    Lx, Ly, Lz = g.Lx, g.Ly, g.Lz
    per_face = {
        0: Ly * Lz, 1: Ly * Lz,   # X−, X+
        2: Lx * Lz, 3: Lx * Lz,   # Y−, Y+
        4: Lx * Ly, 5: Lx * Ly,   # Z−, Z+
    }
    result["per_face"] = {k: float(v) for k, v in per_face.items()}
    result["total"] = float(2.0 * (Lx * Ly + Ly * Lz + Lx * Lz))
    return result


def parse_direction(direction) -> np.ndarray:
    """Преобразовать направление потока в единичный вектор.

    Принимает:
      - строку: '+x','-x','+y','-y','+z','-z';
      - кортеж/список из 3 чисел (нормируется).
    """
    if isinstance(direction, str):
        d = direction.strip().lower().replace(" ", "")
        table = {
            "+x": (1, 0, 0), "x+": (1, 0, 0), "x": (1, 0, 0),
            "-x": (-1, 0, 0), "x-": (-1, 0, 0),
            "+y": (0, 1, 0), "y+": (0, 1, 0), "y": (0, 1, 0),
            "-y": (0, -1, 0), "y-": (0, -1, 0),
            "+z": (0, 0, 1), "z+": (0, 0, 1), "z": (0, 0, 1),
            "-z": (0, 0, -1), "z-": (0, 0, -1),
        }
        v = np.array(table.get(d, (1, 0, 0)), dtype=float)
    else:
        v = np.array(direction, dtype=float).reshape(3)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0])


def faces_exposed_to_flow(problem, direction) -> dict:
    """Классификация граней box по отношению к направлению потока.

    Для каждой грани вычисляется cosθ = n̂·û (n̂ — внешняя нормаль грани,
    û — направление потока):
        cosθ < 0  → наветренная грань (фронтальная, поток «бьёт» в неё);
        cosθ ≈ 0  → боковая (поток скользит вдоль) — продольное обтекание;
        cosθ > 0  → подветренная (в «тени» потока).

    Возвращает {face_id: {'cos': cosθ, 'role': 'наветр./боков./подветр.'}}.
    """
    u = parse_direction(direction)
    normals = _face_normals_box()
    out = {}
    for fid in range(6):
        c = float(np.dot(normals[fid], u))
        if c < -0.3:
            role = "наветренная"
        elif c > 0.3:
            role = "подветренная"
        else:
            role = "боковая"
        out[fid] = {"cos": c, "role": role}
    return out


def frontal_area(problem, direction) -> float:
    """Фронтальная (миделевая) площадь — проекция тела на плоскость,
    перпендикулярную потоку. Для box: A_front = Σ A_грани·max(0, −n̂·û).
    """
    u = parse_direction(direction)
    areas = surface_areas(problem)["per_face"]
    normals = _face_normals_box()
    A = 0.0
    for fid in range(6):
        proj = max(0.0, -float(np.dot(normals[fid], u)))
        A += areas.get(fid, 0.0) * proj
    return float(A)


# =============================================================================
# 4. Высокоуровневый анализ обтекания.
# =============================================================================

@dataclass
class ForcedConvectionResult:
    """Результат анализа вынужденной конвекции при обтекании."""
    speed: float
    direction: np.ndarray
    shape: str
    char_length: float
    T_inf: float
    T_surface: float
    fluid: FluidProperties
    Re: float
    Pr: float
    Nu: float
    h: float
    regime: str
    total_area: float
    frontal_area: float
    per_face_area: Dict[int, float]
    Q_total: float          # суммарный конв. поток по всей поверхности, Вт
    Nu_natural: float = 0.0  # для сравнения: свободная конвекция
    h_natural: float = 0.0
    Ra: float = 0.0

    def report_text(self) -> str:
        lines = [
            "════ Конвективный теплообмен при обтекании ════",
            f"  Форма тела:          {SHAPE_NAMES.get(self.shape, self.shape)}",
            f"  Скорость потока U:    {self.speed:.3g} м/с",
            f"  Направление потока:   ({self.direction[0]:+.2f}, "
            f"{self.direction[1]:+.2f}, {self.direction[2]:+.2f})",
            f"  Характерный размер L: {self.char_length:.4g} м",
            f"  Среда: {self.fluid.summary()}",
            "",
            f"  Re = U·L/ν       = {self.Re:.4g}",
            f"  Pr               = {self.Pr:.3f}",
            f"  Nu = h·L/λ       = {self.Nu:.4g}   ({self.regime})",
            f"  h  = Nu·λ/L      = {self.h:.4g} Вт/(м²·К)  ← коэф. теплообмена",
            "",
            f"  Площадь поверхности A_полн = {self.total_area:.4g} м²",
            f"  Фронтальная площадь A_фронт = {self.frontal_area:.4g} м²",
            f"  T_поверхности = {self.T_surface:.1f} °C,  T_среды = {self.T_inf:.1f} °C",
            f"  Конвективный поток Q = h·A·ΔT = {self.Q_total:.4g} Вт",
        ]
        if self.h_natural > 0:
            lines += [
                "",
                "  Для сравнения (без обдува, свободная конвекция):",
                f"    Ra = {self.Ra:.3g},  Nu_своб = {self.Nu_natural:.3g},  "
                f"h_своб = {self.h_natural:.3g} Вт/(м²·К)",
                f"    Обдув усиливает теплоотдачу в "
                f"{self.h / max(self.h_natural, 1e-9):.1f} раз(а).",
            ]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "speed": self.speed,
            "direction": self.direction.tolist(),
            "shape": self.shape,
            "char_length": self.char_length,
            "T_inf": self.T_inf,
            "T_surface": self.T_surface,
            "fluid": {
                "name": self.fluid.name, "T_film_C": self.fluid.T_film_C,
                "rho": self.fluid.rho, "nu": self.fluid.nu, "k": self.fluid.k,
                "Pr": self.fluid.Pr, "cp": self.fluid.cp,
            },
            "Re": self.Re, "Pr": self.Pr, "Nu": self.Nu, "h": self.h,
            "regime": self.regime,
            "total_area": self.total_area, "frontal_area": self.frontal_area,
            "per_face_area": {str(k): v for k, v in self.per_face_area.items()},
            "Q_total": self.Q_total,
            "Nu_natural": self.Nu_natural, "h_natural": self.h_natural,
            "Ra": self.Ra,
        }


def characteristic_length(problem, shape: str, direction="+x") -> float:
    """Характерный размер L для критериев подобия.

    • Пластина: длина вдоль потока (размер box по доминирующей оси потока).
    • Цилиндр/сфера: диаметр (берём средний поперечный размер).
    • Куб: размер вдоль потока.
    """
    g = problem.geometry
    L = {0: g.Lx, 1: g.Ly, 2: g.Lz}
    u = parse_direction(direction)
    flow_axis = int(np.argmax(np.abs(u)))  # ось, вдоль которой в основном дует

    if shape in (SHAPE_PLATE, SHAPE_CUBE):
        return float(L[flow_axis])
    if shape == SHAPE_CYLINDER:
        # Диаметр — поперечный размер (две не-осевые стороны, среднее).
        other = [L[a] for a in range(3) if a != flow_axis]
        return float(np.mean(other))
    if shape == SHAPE_SPHERE:
        return float(np.mean([g.Lx, g.Ly, g.Lz]))
    return float(L[flow_axis])


def analyze_forced_convection(problem,
                              speed: float,
                              direction="+x",
                              shape: str = SHAPE_PLATE,
                              T_inf: float = 20.0,
                              T_surface: Optional[float] = None,
                              char_length: Optional[float] = None,
                              g_accel: float = 9.81,
                              compare_natural: bool = True
                              ) -> ForcedConvectionResult:
    """Полный анализ обтекания: Re → Nu → h → Q.

    Параметры:
        speed       — скорость набегающего потока U, м/с;
        direction   — направление потока ('+x' / '-z' / вектор);
        shape       — форма тела (SHAPE_PLATE / CYLINDER / SPHERE / CUBE);
        T_inf       — температура набегающего потока, °C;
        T_surface   — средняя температура поверхности, °C. Если None, берётся
                      из результата расчёта problem.T (если он есть), иначе
                      T_inf + 20 как предварительная оценка;
        char_length — характерный размер L; если None — определяется по форме;
        compare_natural — добавить оценку свободной конвекции для сравнения.

    Возвращает ForcedConvectionResult.
    """
    u = parse_direction(direction)

    # Оценка температуры поверхности.
    if T_surface is None:
        if getattr(problem, "T", None) is not None and problem.T.size:
            T_surface = float(np.mean(problem.T))
        else:
            T_surface = float(T_inf) + 20.0

    # Свойства воздуха берём при температуре плёнки T_film = (T_s + T∞)/2.
    T_film = 0.5 * (float(T_surface) + float(T_inf))
    fluid = air_properties(T_film)

    # Характерный размер.
    if char_length is None:
        char_length = characteristic_length(problem, shape, direction)
    char_length = float(char_length)

    # Re, Nu, h.
    Re = reynolds(speed, char_length, fluid)
    Nu, regime = nusselt_forced(Re, fluid.Pr, shape)
    h = heat_transfer_coefficient(Nu, char_length, fluid)

    # Площади.
    areas = surface_areas(problem)
    A_total = areas["total"]
    A_front = frontal_area(problem, direction)

    # Конвективный поток по всей поверхности.
    Q_total = h * A_total * (float(T_surface) - float(T_inf))

    res = ForcedConvectionResult(
        speed=float(speed), direction=u, shape=shape,
        char_length=char_length, T_inf=float(T_inf),
        T_surface=float(T_surface), fluid=fluid,
        Re=Re, Pr=fluid.Pr, Nu=Nu, h=h, regime=regime,
        total_area=A_total, frontal_area=A_front,
        per_face_area=areas["per_face"], Q_total=Q_total,
    )

    # Сравнение со свободной конвекцией.
    if compare_natural:
        dT = abs(float(T_surface) - float(T_inf))
        Ra = rayleigh(dT, char_length, fluid, g=g_accel)
        Nu_n, _ = nusselt_natural_vertical_plate(Ra, fluid.Pr)
        res.Ra = Ra
        res.Nu_natural = Nu_n
        res.h_natural = heat_transfer_coefficient(Nu_n, char_length, fluid)

    return res


# =============================================================================
# 5. Применение как граничного условия (связь потока и решателя).
# =============================================================================

def apply_forced_convection_bc(problem,
                               speed: float,
                               direction="+x",
                               shape: str = SHAPE_PLATE,
                               T_inf: float = 20.0,
                               T_surface: Optional[float] = None,
                               char_length: Optional[float] = None,
                               faces: Optional[List[int]] = None,
                               orient_weighting: bool = False
                               ) -> ForcedConvectionResult:
    """Рассчитать h по параметрам потока и НАЗНАЧИТЬ конвекцию (Робен) на грани.

    Это «замыкает» цепочку: набегающий поток → h → ГУ, после чего обычный
    solve() даёт поле температуры с учётом обдува.

    faces — список граней (0..5), на которые ставить конвекцию.
            None → все грани, обращённые к потоку и боковые (т.е. не «в тени»),
            а при orient_weighting=False — просто все 6 граней.
    orient_weighting — если True, h на боковых/подветренных гранях уменьшается
            (грубая модель неравномерности теплоотдачи по обводу тела:
             наветренная ×1.0, боковая ×0.7, подветренная ×0.5).

    Возвращает ForcedConvectionResult (с тем же h, что назначен наветренной грани).
    """
    from .problem import BoundaryCondition
    from .core_bridge import BC_ROBIN

    res = analyze_forced_convection(
        problem, speed, direction=direction, shape=shape,
        T_inf=T_inf, T_surface=T_surface, char_length=char_length)

    roles = faces_exposed_to_flow(problem, direction)
    weight = {"наветренная": 1.0, "боковая": 0.7, "подветренная": 0.5}

    if faces is None:
        target_faces = list(range(6))
    else:
        target_faces = list(faces)

    for fid in target_faces:
        w = weight[roles[fid]["role"]] if orient_weighting else 1.0
        problem.bcs[fid] = BoundaryCondition(
            type=BC_ROBIN, alpha=res.h * w, T_inf=float(T_inf))

    return res
