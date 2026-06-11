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


def compute_streamlines(problem: Problem, seed_points: np.ndarray,
                          step: float = None, max_steps: int = 200,
                          backward: bool = False) -> list:
    """Построить линии тока теплового потока q = -λ∇T.

    Метод: интегрирование Рунге-Кутта 2-го порядка по полю q.
    seed_points: (S, 3) — стартовые точки.
    step: шаг интегрирования. По умолчанию = 0.5% от диагонали bbox.
    backward: если True, идти ПРОТИВ потока (вверх по T).
    """
    if problem.flux is None or problem.nodes is None:
        return []
    nodes = problem.nodes
    flux = problem.flux
    bbox_min = nodes.min(axis=0)
    bbox_max = nodes.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if step is None:
        step = 0.005 * diag
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(nodes)
        def find_idx(p): _, i = tree.query(p, k=1); return int(i)
    except ImportError:
        def find_idx(p):
            return int(np.argmin(np.sum((nodes - p) ** 2, axis=1)))
    sign = -1.0 if backward else 1.0
    streamlines = []
    for s in seed_points:
        line = [np.asarray(s, dtype=np.float64)]
        p = np.asarray(s, dtype=np.float64)
        for _ in range(max_steps):
            idx = find_idx(p)
            v = sign * flux[idx]
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-12: break
            v = v / v_norm
            p_mid = p + 0.5 * step * v
            idx2 = find_idx(p_mid)
            v2 = sign * flux[idx2]
            n2 = float(np.linalg.norm(v2))
            if n2 < 1e-12: break
            v2 = v2 / n2
            p_new = p + step * v2
            if (p_new < bbox_min - step).any() or (p_new > bbox_max + step).any():
                line.append(p_new); break
            line.append(p_new)
            p = p_new
        if len(line) > 1:
            streamlines.append(np.array(line))
    return streamlines


def compute_difference_field(T_a: np.ndarray, T_b: np.ndarray,
                              relative: bool = False) -> dict:
    """Сравнение двух полей температур, посчитанных на одной сетке.

    T_a, T_b — массивы (N,) одной длины.
    Возвращает dict со статистикой: delta, max_abs, mean_abs, rms.
    Если relative=True, также delta_rel, max_rel.
    """
    if T_a.shape != T_b.shape:
        raise ValueError(f"Размеры различаются: {T_a.shape} != {T_b.shape}.")
    delta = T_a - T_b
    abs_delta = np.abs(delta)
    res = {
        "delta":     delta,
        "max_abs":   float(abs_delta.max()),
        "mean_abs":  float(abs_delta.mean()),
        "rms":       float(np.sqrt(np.mean(delta * delta))),
        "argmax":    int(np.argmax(abs_delta)),
    }
    if relative:
        denom = np.maximum(np.abs(T_a), np.abs(T_b))
        denom = np.where(denom < 1e-12, 1.0, denom)
        rel = delta / denom
        res["delta_rel"] = rel
        res["max_rel"] = float(np.abs(rel).max())
    return res


def compute_temperature_profile(problem: Problem, point_a, point_b,
                                  n_samples: int = 100):
    """Профиль температуры вдоль отрезка между двумя точками.

    Для каждой точки сэмплирования находит ближайший узел сетки
    и берёт его температуру. Простой и устойчивый метод nearest-neighbor;
    не требует баричентрической интерполяции.

    point_a, point_b — tuple/array (x, y, z) в системе координат сетки.
    n_samples — число точек вдоль линии.

    Возвращает (distances, temperatures) — массивы длины n_samples.
        distances[i] = расстояние i-й точки от point_a [м];
        temperatures[i] = T в ближайшем узле к этой точке [°C].
    Если решение ещё не выполнено, возвращает (None, None).
    """
    if problem.T is None or problem.nodes is None:
        return None, None
    a = np.asarray(point_a, dtype=np.float64)
    b = np.asarray(point_b, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, n_samples)
    line_points = a[None, :] + (b - a)[None, :] * ts[:, None]
    temps = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        diffs = problem.nodes - line_points[i]
        idx = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        temps[i] = problem.T[idx]
    distances = ts * float(np.linalg.norm(b - a))
    return distances, temps


def export_pdf_report(problem: Problem, path: str,
                       screenshot_path: Optional[str] = None,
                       author: str = "",
                       title: str = "Отчёт о расчёте теплопроводности") -> None:
    """Сводный отчёт в PDF.

    Содержит:
        титульный лист с метаданными;
        постановку задачи (геометрия, материал, ГУ, источники);
        результаты (T_min, T_max, время решения, число итераций);
        энергобаланс;
        опционально — скриншот 3D и график распределения температур.
    Зависит только от matplotlib (PdfPages), который и так используется
    для графиков, поэтому не добавляет новых зависимостей.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Для PDF-отчёта требуется matplotlib. Установите: pip install matplotlib"
        ) from exc

    p = problem
    info = p.info
    Tmin, Tmax = p.temperature_range()

    with PdfPages(path) as pdf:
        # --- Стр. 1: Метаданные и постановка ---
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 портрет
        fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.05)
        ax = fig.add_subplot(111); ax.axis("off")
        lines = []
        lines.append(title.upper()); lines.append("")
        lines.append(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        if author:
            lines.append(f"Автор: {author}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("ПОСТАНОВКА ЗАДАЧИ")
        lines.append("=" * 60); lines.append("")
        lines.append(f"Геометрия:  {p.geometry.Lx:g} × {p.geometry.Ly:g} × {p.geometry.Lz:g} м")
        lines.append(f"Сетка:       {p.geometry.nx} × {p.geometry.ny} × {p.geometry.nz}")
        if p.nodes is not None:
            lines.append(f"Узлов: {p.nodes.shape[0]}, "
                         f"элементов: {p.elements.shape[0]}")
        lines.append("")
        lines.append(f"Материал: λ = {p.lambda_:g} Вт/(м·К)")
        lines.append(f"Объёмный источник Q = {p.Q:g} Вт/м³")
        if p.material_regions:
            lines.append(f"Регионов материалов: {len(p.material_regions)}")
            for r in p.material_regions:
                lines.append(f"  · {r.description()}")
        if p.point_sources:
            lines.append(f"Точечных источников: {len(p.point_sources)}")
        if p.volume_sources:
            lines.append(f"Объёмных источников: {len(p.volume_sources)}")
        lines.append("")
        lines.append("Граничные условия:")
        for fid in range(6):
            bc = p.bcs[fid]
            lines.append(f"  · {FACE_NAMES[fid]}: {bc.description()}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("РЕЗУЛЬТАТЫ")
        lines.append("=" * 60); lines.append("")
        lines.append(f"T_min = {Tmin:.4f} °C")
        lines.append(f"T_max = {Tmax:.4f} °C")
        lines.append(f"ΔT = {Tmax - Tmin:.4f} °C")
        if info is not None:
            lines.append("")
            lines.append(f"Решатель:  {info.iterations} итераций, "
                         f"невязка {info.residual:.3e}")
            lines.append(f"Время решения: {info.time_seconds * 1000:.1f} мс")
            lines.append(f"Сходимость: {'да' if info.converged else 'нет'}")
        bal = p.energy_balance()
        if bal is not None:
            lines.append("")
            lines.append(f"Энергобаланс:")
            lines.append(f"  Генерируется внутри:    {bal['q_gen_W']:.3f} Вт")
            lines.append(f"  Уходит через границу:   {bal['net_out_W']:.3f} Вт")
            lines.append(f"  Относительная ошибка:   {bal['rel_err'] * 100:.2f}%")
        text = "\n".join(lines)
        ax.text(0.05, 0.98, text, transform=ax.transAxes, fontsize=10,
                family="monospace", verticalalignment="top")
        pdf.savefig(fig); plt.close(fig)

        # --- Стр. 2: Гистограмма + скриншот ---
        if p.T is not None and p.T.size > 0:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax1 = fig.add_subplot(211)
            ax1.hist(p.T, bins=50, color="#7a6cf0", edgecolor="#3c4049")
            ax1.set_xlabel("T, °C"); ax1.set_ylabel("Число узлов")
            ax1.set_title("Распределение температур по узлам сетки")
            ax1.grid(alpha=0.3)
            ax1.axvline(Tmin, color="#3aa5ff", linestyle="--",
                        label=f"T_min = {Tmin:.2f} °C")
            ax1.axvline(Tmax, color="#ff7b3a", linestyle="--",
                        label=f"T_max = {Tmax:.2f} °C")
            ax1.legend()

            ax2 = fig.add_subplot(212); ax2.axis("off")
            if screenshot_path and os.path.isfile(screenshot_path):
                try:
                    from matplotlib.image import imread
                    img = imread(screenshot_path)
                    ax2.imshow(img); ax2.set_title("3D-визуализация поля температур")
                except Exception:
                    pass
            pdf.savefig(fig); plt.close(fig)


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

    # Конвекция при обдуве — все величины (если обдув задан).
    try:
        from . import convection as _cv
        if getattr(problem, "air_flow_enabled", False):
            lines.append("КОНВЕКЦИЯ ПРИ ОБТЕКАНИИ (ОБДУВ)")
            for ln in _cv.convection_summary_text(problem).splitlines():
                lines.append("  " + ln if ln else ln)
            lines.append("")
    except Exception:
        pass

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


# =============================================================================
# Число Нуссельта — критерий теплоотдачи.
# =============================================================================

def compute_nusselt(problem: Problem, face_id: int,
                     characteristic_length: float = None,
                     fluid_lambda: float = None) -> dict:
    """Число Нуссельта на грани с конвекцией.

    Nu = h · L / λ, где
        h  — реальный коэффициент теплоотдачи (из поля T и q)
        L  — характерный размер (например, длина грани)
        λ  — коэффициент теплопроводности СРЕДЫ (жидкости/газа).

    fluid_lambda — теплопроводность среды (воздуха и т.п.), Вт/(м·К).
        Физически Nu определяется именно по λ СРЕДЫ, а не тела. Если None,
        для обратной совместимости берётся λ тела (problem.lambda_); для
        корректного Nu при обдуве воздухом передавайте λ_воздуха ≈ 0.026.

    На грани с условием Робена (конвекции) расчёт даёт фактический h из:
        q_n = h · (T_wall - T_inf)  =>  h = q_n / (T_wall - T_inf)

    face_id — индекс грани 0..5.
    characteristic_length — если None, используется диагональ грани.
    """
    from .core_bridge import BC_ROBIN
    if problem.T is None or problem.flux is None or problem.nodes is None:
        return {"error": "Нет данных расчёта"}
    bc = problem.bcs.get(face_id)
    if bc is None or bc.type != BC_ROBIN:
        return {"error": f"На грани {face_id} нет конвекции"}

    # Узлы на грани (для Box-геометрии).
    g = problem.geometry
    axis = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}[face_id]
    val  = {0: 0.0, 1: g.Lx, 2: 0.0, 3: g.Ly, 4: 0.0, 5: g.Lz}[face_id]
    tol_d = 1e-6 * max(g.Lx, g.Ly, g.Lz)
    mask = np.abs(problem.nodes[:, axis] - val) < tol_d
    if not np.any(mask):
        return {"error": "Узлы грани не найдены"}

    T_wall = float(problem.T[mask].mean())
    delta_T = T_wall - bc.T_inf
    if abs(delta_T) < 1e-9:
        return {"warning": "Нулевая разность T_wall - T_inf",
                "T_wall": T_wall, "T_inf": bc.T_inf}

    # Поток через грань: нормальная компонента q · n.
    normal = np.zeros(3); normal[axis] = -1.0 if val == 0.0 else 1.0
    q_n_field = problem.flux[mask] @ normal
    q_n_mean = float(np.mean(q_n_field))
    # Внутренний коэффициент теплоотдачи (должен ≈ α_BC).
    h_actual = abs(q_n_mean / delta_T) if abs(delta_T) > 1e-12 else 0.0

    # Характерный размер.
    if characteristic_length is None:
        if axis == 0:   L = max(g.Ly, g.Lz)
        elif axis == 1: L = max(g.Lx, g.Lz)
        else:           L = max(g.Lx, g.Ly)
    else:
        L = float(characteristic_length)
    # λ среды для Nu (физически верно), λ тела для Bi (сопротивление в теле).
    lam_fluid = float(fluid_lambda) if fluid_lambda else problem.lambda_
    Nu = h_actual * L / lam_fluid
    Bi = bc.alpha * L / problem.lambda_  # число Био (по λ тела)
    return {
        "Nu":           Nu,
        "Bi":           Bi,
        "h_actual":     h_actual,
        "h_BC":         bc.alpha,
        "lambda_fluid": lam_fluid,
        "T_wall_mean":  T_wall,
        "T_inf":        bc.T_inf,
        "delta_T":      delta_T,
        "q_n_mean":     q_n_mean,
        "L":            L,
        "interpretation": _interpret_nusselt(Nu, Bi),
    }


def _interpret_nusselt(Nu: float, Bi: float) -> str:
    parts = []
    if Nu < 2.0:
        parts.append("Nu < 2 — теплоотдача в режиме чистой теплопроводности.")
    elif Nu < 10:
        parts.append("Nu = 2..10 — ламинарная свободная конвекция.")
    elif Nu < 100:
        parts.append("Nu = 10..100 — развитая свободная или слабая вынужденная "
                     "конвекция.")
    else:
        parts.append("Nu > 100 — интенсивная вынужденная конвекция.")
    if Bi < 0.1:
        parts.append("Bi < 0.1 — применима модель сосредоточенной теплоёмкости "
                     "(тело почти изотермическое).")
    elif Bi > 10:
        parts.append("Bi > 10 — поверхностное сопротивление мало, доминирует "
                     "теплопроводность в теле.")
    return " ".join(parts)


# =============================================================================
# Расширенный протокол расчёта.
# =============================================================================

def export_calculation_protocol(problem: Problem, path: str,
                                   author: str = "",
                                   organization: str = "",
                                   description: str = "") -> None:
    """Структурированный протокол расчёта в JSON.

    Содержит ВСЕ исходные данные и результаты в машиночитаемом формате,
    что позволяет точно воспроизвести расчёт или сравнить серию протоколов.

    Состав:
      - метаданные (дата, версия, автор, организация, комментарий)
      - геометрия и сетка (размеры, узлы/элементы, гистограмма качества)
      - материал (λ, ρ, c_p, ε, изотропия/анизотропия)
      - источники (точечные, объёмные)
      - граничные условия (тип, параметры на каждой грани)
      - решатель (метод, точность, итерации, время)
      - результаты (T_min, T_max, среднее, поле T в узлах опционально)
      - энергобаланс (генерация, отвод, ошибка)
      - физические критерии (если применимы): Bi, Nu

    Файл может быть прочитан другими инструментами для сравнения.
    """
    import json
    info = problem.info
    Tmin, Tmax = problem.temperature_range()
    g = problem.geometry
    bcs_data = []
    for fid in range(6):
        bc = problem.bcs.get(fid)
        if bc is None: continue
        bcs_data.append({
            "face":        FACE_NAMES[fid],
            "face_id":     fid,
            "type":        _bc_type_name(bc.type),
            "T0":          float(bc.T0),
            "q0":          float(bc.q0),
            "alpha":       float(bc.alpha),
            "T_inf":       float(bc.T_inf),
            "emissivity":  float(getattr(bc, "emissivity", 0.0)),
        })
    point_sources_data = [
        {"x": float(s.x), "y": float(s.y), "z": float(s.z),
         "power_W": float(s.power)}
        for s in problem.point_sources
    ]
    volume_sources_data = [
        {"x": float(s.x), "y": float(s.y), "z": float(s.z),
         "radius_m": float(s.radius),
         "Q_density_W_m3": float(s.Q_density)}
        for s in problem.volume_sources
    ]
    material_regions_data = [r.to_dict() if hasattr(r, "to_dict") else
                              {"description": r.description() if hasattr(r, "description") else str(r)}
                              for r in problem.material_regions]
    bal = problem.energy_balance() or {}
    protocol = {
        "metadata": {
            "format_version":  "1.0",
            "program":          "fem_heat3d",
            "program_version":  "1.11",
            "created":          datetime.now().isoformat(),
            "author":           author,
            "organization":     organization,
            "description":      description,
        },
        "geometry": {
            "Lx_m": g.Lx, "Ly_m": g.Ly, "Lz_m": g.Lz,
            "nx": g.nx, "ny": g.ny, "nz": g.nz,
            "volume_m3":   float(g.Lx * g.Ly * g.Lz),
            "surface_m2":  float(2 * (g.Lx * g.Ly + g.Ly * g.Lz + g.Lx * g.Lz)),
            "external_mesh": problem.has_external_mesh(),
        },
        "mesh_stats": _mesh_stats(problem),
        "material": {
            "lambda_W_m_K":   problem.lambda_,
            "rho_kg_m3":      problem.rho,
            "cp_J_kg_K":      problem.cp,
            "Q_W_m3":         problem.Q,
            "is_anisotropic": bool(getattr(problem, "is_anisotropic", False)),
        },
        "material_regions":     material_regions_data,
        "point_sources":         point_sources_data,
        "volume_sources":        volume_sources_data,
        "boundary_conditions":  bcs_data,
        "solver": {
            "method":           "CG with Jacobi preconditioner",
            "iterations":       int(info.iterations) if info else None,
            "residual":         float(info.residual) if info else None,
            "time_seconds":     float(info.time_seconds) if info else None,
            "converged":        bool(info.converged) if info else None,
        },
        "results": {
            "T_min_C":     Tmin, "T_max_C": Tmax,
            "T_mean_C":    float(problem.T.mean()) if problem.T is not None else None,
            "T_std_C":     float(problem.T.std()) if problem.T is not None else None,
            "delta_T_C":   Tmax - Tmin,
        },
        "energy_balance": {
            "generated_W":      bal.get("q_gen_W"),
            "outflow_W":        bal.get("net_out_W"),
            "imbalance_W":      bal.get("imbalance_W"),
            "relative_error":   bal.get("rel_err"),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2, default=str)


def _bc_type_name(t: int) -> str:
    return {0: "не задано", 1: "Дирихле (T)", 2: "Нейман (q)",
             3: "Робен (конвекция)", 4: "излучение"}.get(int(t), str(t))


def _mesh_stats(problem: Problem) -> dict:
    """Базовая статистика сетки: объёмы, площади граней, качество."""
    if problem.nodes is None or problem.elements is None:
        return {}
    nodes = problem.nodes
    tets = problem.elements
    a = nodes[tets[:, 0]]; b = nodes[tets[:, 1]]
    c = nodes[tets[:, 2]]; d = nodes[tets[:, 3]]
    vol = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0
    vol_abs = np.abs(vol)
    return {
        "n_nodes":            int(nodes.shape[0]),
        "n_elements":         int(tets.shape[0]),
        "volume_min_m3":      float(vol_abs.min()),
        "volume_max_m3":      float(vol_abs.max()),
        "volume_mean_m3":     float(vol_abs.mean()),
        "volume_total_m3":    float(vol_abs.sum()),
        "aspect_ratio":       float(vol_abs.max() / max(vol_abs.min(), 1e-30)),
    }


def sample_history_at_points(nodes: np.ndarray, T_history: np.ndarray,
                              points) -> np.ndarray:
    """Извлечь T(t) в заданных точках из истории нестационарного расчёта.

    nodes — (N, 3) координаты узлов.
    T_history — (n_save, N) температуры на каждом снимке.
    points — список/массив (P, 3) точек наблюдения.

    Для каждой точки берётся ближайший узел (nearest-neighbor).
    Возвращает массив (P, n_save): температура в каждой точке на каждом снимке.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = pts.shape[0]
    n_save = T_history.shape[0]
    out = np.zeros((P, n_save), dtype=np.float64)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(nodes)
        _, idxs = tree.query(pts, k=1)
    except ImportError:
        idxs = np.array([int(np.argmin(np.sum((nodes - p) ** 2, axis=1)))
                          for p in pts])
    for p in range(P):
        out[p, :] = T_history[:, int(idxs[p])]
    return out


# =============================================================================
# Отчёт по нестационарному расчёту.
# =============================================================================

def _transient_dynamics_table(times: np.ndarray, T_history: np.ndarray,
                              max_rows: int = 25):
    """Прореженная таблица динамики: (t, Tmin, Tmean, Tmax) по снимкам."""
    n = len(times)
    if n <= max_rows:
        idxs = np.arange(n)
    else:
        idxs = np.unique(np.round(np.linspace(0, n - 1, max_rows)).astype(int))
    rows = []
    for i in idxs:
        T = T_history[i]
        rows.append((float(times[i]), float(T.min()),
                     float(T.mean()), float(T.max())))
    return rows


def transient_summary(times: np.ndarray, T_history: np.ndarray) -> dict:
    """Интегральные характеристики переходного процесса.

    Возвращает словарь:
        direction      — 'нагрев' / 'остывание' / 'смешанный/стационарный'
        T_mean_0/T_mean_end — средняя T в начале и в конце, °C
        dT_total       — суммарное изменение средней T, °C
        t63 / t95      — время достижения 63.2 % / 95 % полного изменения, с
        settled        — степень выхода на стационар (0..1) по последним кадрам
        rate_initial   — начальная скорость изменения средней T, °C/с
        cooling_rate_m — темп процесса m (регулярный режим), 1/с, или None
    """
    times = np.asarray(times, dtype=np.float64)
    means = T_history.mean(axis=1)
    T0, Te = float(means[0]), float(means[-1])
    dT = Te - T0
    if abs(dT) < 1e-9:
        direction = "стационарный (средняя T не изменилась)"
    elif dT > 0:
        direction = "нагрев"
    else:
        direction = "остывание"

    # Время достижения 63.2 % и 95 % полного изменения средней T.
    t63 = t95 = None
    if abs(dT) > 1e-9:
        frac = (means - T0) / dT          # 0 → 1, монотонно для типовых задач
        for level, name in ((0.632, "t63"), (0.95, "t95")):
            above = np.nonzero(frac >= level)[0]
            if above.size:
                k = int(above[0])
                if k == 0:
                    t_val = float(times[0])
                else:
                    # Линейная интерполяция между кадрами k-1 и k.
                    f0, f1 = frac[k - 1], frac[k]
                    t0_, t1_ = times[k - 1], times[k]
                    t_val = float(t0_ + (level - f0) / max(f1 - f0, 1e-12)
                                  * (t1_ - t0_))
                if name == "t63":
                    t63 = t_val
                else:
                    t95 = t_val

    # Степень выхода на стационар: изменение за последние 10 % времени
    # относительно полного изменения.
    settled = None
    if abs(dT) > 1e-9 and len(times) >= 3:
        k_tail = max(1, int(0.9 * (len(times) - 1)))
        tail_change = abs(float(means[-1] - means[k_tail]))
        settled = max(0.0, min(1.0, 1.0 - tail_change / abs(dT)))

    # Начальная скорость изменения средней T.
    rate_initial = None
    if len(times) >= 2 and times[1] > times[0]:
        rate_initial = float((means[1] - means[0]) / (times[1] - times[0]))

    # Темп процесса m (метод регулярного режима): ln|Tmean - Tend| ~ -m·t.
    cooling_rate_m = None
    if abs(dT) > 1e-6 and len(times) >= 6:
        theta = np.abs(means - Te)
        # Берём средний участок (отбрасываем начальный нерегулярный режим
        # и хвост, где theta → 0 и логарифм шумит).
        lo, hi = int(0.2 * len(times)), int(0.8 * len(times))
        seg_t, seg_th = times[lo:hi], theta[lo:hi]
        mask = seg_th > 1e-6 * abs(dT)
        if int(mask.sum()) >= 3:
            coeff = np.polyfit(seg_t[mask], np.log(seg_th[mask]), 1)
            if coeff[0] < 0:
                cooling_rate_m = float(-coeff[0])

    return {
        "direction": direction,
        "T_mean_0": T0, "T_mean_end": Te, "dT_total": dT,
        "t63": t63, "t95": t95, "settled": settled,
        "rate_initial": rate_initial, "cooling_rate_m": cooling_rate_m,
    }


def export_transient_report(problem: Problem, times: np.ndarray,
                            T_history: np.ndarray, path: str,
                            params: Optional[dict] = None) -> None:
    """Сформировать отдельный отчёт о НЕСТАЦИОНАРНОМ расчёте.

    problem   — задача (геометрия, материал, ГУ, ρ, c_p).
    times     — (n_save,) моменты времени снимков, с.
    T_history — (n_save, N) температуры узлов на каждом снимке.
    path      — куда сохранить .txt.
    params    — параметры интегрирования из GUI (t_end, dt, T_init, ...).
    """
    times = np.asarray(times, dtype=np.float64)
    T_history = np.asarray(T_history, dtype=np.float64)
    if times.size < 2 or T_history.ndim != 2:
        raise RuntimeError("Нет данных нестационарного расчёта")

    g = problem.geometry
    p = params or {}
    dt = float(p.get("dt", times[1] - times[0] if times.size > 1 else 0.0))
    t_end = float(p.get("t_end", times[-1]))
    T_init = p.get("T_init")
    rho = float(p.get("rho", getattr(problem, "rho", 0.0)) or 0.0)
    cp = float(p.get("cp", getattr(problem, "cp", 0.0)) or 0.0)
    lam = float(problem.lambda_)
    n_steps = int(np.ceil(t_end / dt)) if dt > 0 else 0

    # Теплофизические производные величины.
    a = lam / (rho * cp) if rho > 0 and cp > 0 else None
    dims = [d for d in (g.Lx, g.Ly, g.Lz) if d and d > 0]
    L = float(min(dims)) if dims else None
    tau = (L * L / a) if (a and L) else None
    Fo = (a * t_end / (L * L)) if (a and L) else None

    summ = transient_summary(times, T_history)

    lines = []
    ap = lines.append
    ap("=" * 70)
    ap("Отчёт о НЕСТАЦИОНАРНОМ расчёте — программный комплекс МКЭ")
    ap(f"Сформирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ap("=" * 70)
    ap("")
    ap("РЕЖИМ")
    ap("  Нестационарная теплопроводность: ρ·c_p·∂T/∂t = ∇·(λ∇T) + Q")
    ap("  Схема интегрирования: неявная схема Эйлера 1-го порядка")
    ap("  (безусловно устойчива; точность O(Δt))")
    ap("")
    ap("ПАРАМЕТРЫ ИНТЕГРИРОВАНИЯ")
    ap(f"  Конечное время t_end:   {t_end:g} с")
    ap(f"  Шаг по времени Δt:      {dt:g} с")
    ap(f"  Число шагов:            {n_steps}")
    ap(f"  Сохранено снимков:      {len(times)}")
    if T_init is not None:
        ap(f"  Начальная T₀:           {float(T_init):g} °C")
    ap("")
    ap("ГЕОМЕТРИЯ И СЕТКА")
    ap(f"  Размеры: Lx = {g.Lx:g} м, Ly = {g.Ly:g} м, Lz = {g.Lz:g} м")
    if problem.nodes is not None:
        ap(f"  Узлов:      {problem.nodes.shape[0]}")
    if problem.elements is not None:
        ap(f"  Элементов:  {problem.elements.shape[0]}")
    ap("")
    ap("ТЕПЛОФИЗИЧЕСКИЕ СВОЙСТВА")
    ap(f"  λ   = {lam:g} Вт/(м·К)")
    ap(f"  ρ   = {rho:g} кг/м³")
    ap(f"  c_p = {cp:g} Дж/(кг·К)")
    ap(f"  Q   = {problem.Q:g} Вт/м³")
    if a is not None:
        ap(f"  Температуропроводность a = λ/(ρ·c_p) = {a:.4e} м²/с")
    if L is not None:
        ap(f"  Характерный размер L = {L:g} м")
    if tau is not None:
        ap(f"  Характерное время  τ = L²/a = {tau:.4g} с "
           f"(≈ {tau/60.0:.3g} мин)")
    if Fo is not None:
        ap(f"  Число Фурье Fo = a·t_end/L² = {Fo:.4g}")
        if Fo < 0.5:
            ap("    (Fo < 0.5 — процесс далёк от стационара)")
        elif Fo > 3:
            ap("    (Fo > 3 — процесс практически вышел в стационар)")
    ap("")
    ap("ГРАНИЧНЫЕ УСЛОВИЯ")
    for face_id in range(6):
        bc = problem.bcs[face_id]
        ap(f"  {FACE_NAMES[face_id]}: {_bc_describe(bc)}")
    ap("")
    ap("ДИНАМИКА ПЕРЕХОДНОГО ПРОЦЕССА")
    ap(f"  {'t, с':>12} | {'Tmin, °C':>10} | {'Tmean, °C':>10} | "
       f"{'Tmax, °C':>10}")
    ap("  " + "-" * 52)
    for (t, tmin, tmean, tmax) in _transient_dynamics_table(times, T_history):
        ap(f"  {t:>12.4g} | {tmin:>10.3f} | {tmean:>10.3f} | {tmax:>10.3f}")
    ap("")
    ap("АНАЛИЗ")
    ap(f"  Характер процесса:        {summ['direction']}")
    ap(f"  Средняя T:                {summ['T_mean_0']:.3f} °C → "
       f"{summ['T_mean_end']:.3f} °C (Δ = {summ['dT_total']:+.3f} °C)")
    if summ["rate_initial"] is not None:
        ap(f"  Начальная скорость:       {summ['rate_initial']:+.4g} °C/с")
    if summ["t63"] is not None:
        ap(f"  Время 63.2 % изменения:   {summ['t63']:.4g} с")
    if summ["t95"] is not None:
        ap(f"  Время 95 % изменения:     {summ['t95']:.4g} с")
    if summ["cooling_rate_m"] is not None:
        ap(f"  Темп процесса m:          {summ['cooling_rate_m']:.4e} 1/с "
           "(регулярный режим, ln θ ~ −m·t)")
    if summ["settled"] is not None:
        pct = summ["settled"] * 100.0
        ap(f"  Выход на стационар:       ≈ {pct:.1f} % "
           f"({'достигнут' if pct >= 95 else 'НЕ достигнут — увеличьте t_end'})")
    ap("")
    Tg_min = float(T_history.min())
    Tg_max = float(T_history.max())
    ap("ГЛОБАЛЬНЫЙ ДИАПАЗОН (шкала анимации)")
    ap(f"  Tmin = {Tg_min:.4f} °C,  Tmax = {Tg_max:.4f} °C")
    ap("")

    # Точки наблюдения: T(t) по каждой точке.
    obs = list(getattr(problem, "observation_points", []) or [])
    if obs and problem.nodes is not None:
        series = sample_history_at_points(problem.nodes, T_history, obs)
        ap("ТОЧКИ НАБЛЮДЕНИЯ — T(t)")
        n = len(times)
        idxs = (np.arange(n) if n <= 15 else
                np.unique(np.round(np.linspace(0, n - 1, 15)).astype(int)))
        header = f"  {'t, с':>12}"
        for i, (x, y, z) in enumerate(obs):
            header += f" | #{i+1} ({x:.3g},{y:.3g},{z:.3g})"
        ap(header)
        ap("  " + "-" * max(40, len(header) - 2))
        for k in idxs:
            row = f"  {times[k]:>12.4g}"
            for i in range(series.shape[0]):
                row += f" | {series[i, k]:>10.3f} °C"
            ap(row)
        ap("")

    ap("=" * 70)

    with open(path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines))


def export_transient_history_csv(problem: Problem, times: np.ndarray,
                                 T_history: np.ndarray, path: str) -> None:
    """CSV-выгрузка динамики: t; Tmin; Tmean; Tmax [; T в точках наблюдения]."""
    times = np.asarray(times, dtype=np.float64)
    T_history = np.asarray(T_history, dtype=np.float64)
    obs = list(getattr(problem, "observation_points", []) or [])
    series = None
    if obs and problem.nodes is not None:
        series = sample_history_at_points(problem.nodes, T_history, obs)
    with open(path, "w", encoding="utf-8") as fout:
        header = "t_s;T_min_C;T_mean_C;T_max_C"
        for i in range(len(obs)):
            x, y, z = obs[i]
            header += f";T_obs{i+1}({x:g},{y:g},{z:g})_C"
        fout.write(header + "\n")
        for k in range(len(times)):
            T = T_history[k]
            row = (f"{times[k]:.6g};{T.min():.6f};"
                   f"{T.mean():.6f};{T.max():.6f}")
            if series is not None:
                for i in range(series.shape[0]):
                    row += f";{series[i, k]:.6f}"
            fout.write(row + "\n")
