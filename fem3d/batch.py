# -*- coding: utf-8 -*-
"""
fem3d.batch
===========

Утилиты для пакетных и параметрических расчётов:
  - вариация одного параметра (λ, α, Q, T_inf, ...)
  - вариация двух параметров (сетка значений)
  - анализ чувствительности (∂T/∂p методом конечных разностей)

Возвращают сводные таблицы и серии полей T для дальнейшего постпроцессинга.
"""
from __future__ import annotations
from typing import Callable, List, Dict, Any
import copy
import numpy as np
from .core_bridge import CoreBridge
from .problem import Problem


def parametric_sweep(template: Problem,
                      parameter_name: str,
                      values: List[float],
                      modifier: Callable[[Problem, float], None] = None,
                      tol: float = 1e-8) -> List[Dict[str, Any]]:
    """Параметрическая серия расчётов.

    Принцип: для каждого значения `v` из `values` делается копия `template`,
    применяется модификатор `modifier(problem, v)` (или встроенный
    setattr(problem, parameter_name, v) если modifier=None),
    затем выполняется расчёт.

    Возвращает список словарей с ключами:
        value, T_min, T_max, T_mean, iterations, residual, converged,
        T_field, energy_balance.

    Параметры по умолчанию для имени:
        'lambda_'    — изотропный λ
        'Q'          — объёмный источник
        'alpha_all'  — α на всех гранях (использовать modifier)
    """
    results = []
    for v in values:
        p = copy.deepcopy(template)
        if modifier is not None:
            modifier(p, v)
        else:
            setattr(p, parameter_name, v)
        with CoreBridge() as br:
            p.build_mesh_in_core(br)
            info = p.solve(br, tol=tol)
        bal = p.energy_balance() or {}
        results.append({
            "value":       v,
            "T_min":       float(p.T.min()),
            "T_max":       float(p.T.max()),
            "T_mean":      float(p.T.mean()),
            "iterations":  int(info.iterations),
            "residual":    float(info.residual),
            "converged":   bool(info.converged),
            "T_field":     p.T.copy(),
            "rel_err_eb":  float(bal.get("rel_err", 0.0)),
        })
    return results


def sensitivity_analysis(template: Problem, parameter_name: str,
                          base_value: float, delta_rel: float = 0.05,
                          tol: float = 1e-8) -> Dict[str, Any]:
    """Чувствительность результата к параметру (метод конечных разностей).

    Считает (T(p + Δp) - T(p - Δp)) / (2·Δp) — центральная разность.
    delta_rel — относительное возмущение (по умолчанию ±5%).

    Возвращает:
        dT_dp_field — поле производных (N,)
        dTmax_dp    — производная T_max по параметру
        dTmean_dp   — производная средней T по параметру
        T_base      — поле T при базовом значении
    """
    dp = base_value * delta_rel
    if dp == 0.0:
        dp = 1e-6
    vals = [base_value - dp, base_value, base_value + dp]
    results = parametric_sweep(template, parameter_name, vals, tol=tol)
    T_m, T_0, T_p = results[0]["T_field"], results[1]["T_field"], results[2]["T_field"]
    dT_dp = (T_p - T_m) / (2.0 * dp)
    return {
        "parameter_name":  parameter_name,
        "base_value":      base_value,
        "delta_used":      dp,
        "dT_dp_field":     dT_dp,
        "dTmax_dp":        (results[2]["T_max"] - results[0]["T_max"]) / (2 * dp),
        "dTmean_dp":       (results[2]["T_mean"] - results[0]["T_mean"]) / (2 * dp),
        "T_base":          T_0,
        "results":         results,
    }


def export_sweep_csv(results: List[Dict[str, Any]], path: str,
                      parameter_label: str = "value") -> None:
    """Сохранить сводку серии расчётов в CSV (без полей T)."""
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([parameter_label, "T_min, °C", "T_max, °C", "T_mean, °C",
                    "итераций", "невязка", "сошёлся", "ошибка энергобаланса, %"])
        for r in results:
            w.writerow([
                f"{r['value']:.6g}",
                f"{r['T_min']:.4f}", f"{r['T_max']:.4f}", f"{r['T_mean']:.4f}",
                r["iterations"], f"{r['residual']:.3e}",
                "да" if r["converged"] else "нет",
                f"{r['rel_err_eb']*100:.2f}",
            ])


# =============================================================================
# Однопараметрическая оптимизация (нахождение λ или Q, минимизирующих T_max).
# =============================================================================

def optimize_parameter(template: Problem,
                        parameter_name: str,
                        target: str = "T_max",
                        bounds: tuple = (1.0, 1000.0),
                        objective: str = "min",
                        tol: float = 1e-3,
                        max_iter: int = 30) -> Dict[str, Any]:
    """Найти оптимальное значение параметра методом золотого сечения.

    parameter_name — имя поля в Problem (например 'lambda_', 'Q').
    target — какую величину минимизировать/максимизировать:
        'T_max', 'T_min', 'T_mean', 'rel_err_eb'.
    bounds — (low, high) — диапазон поиска.
    objective — 'min' (минимизация) или 'max' (максимизация).
    tol — критерий остановки по интервалу: |b - a| < tol·max(|a|, |b|).
    max_iter — максимум вычислений целевой функции.

    Возвращает dict с лучшим значением, целевой и историей.
    """
    sign = 1.0 if objective == "min" else -1.0
    phi = (1 + np.sqrt(5)) / 2  # золотое сечение
    a, b = bounds
    if a >= b:
        raise ValueError("bounds: low должно быть меньше high")

    history = []

    def eval_at(x: float) -> float:
        result = parametric_sweep(template, parameter_name, [x])[0]
        val = float(result[target])
        history.append({"x": x, target: val, "iters": result["iterations"]})
        return sign * val

    # Золотое сечение.
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = eval_at(c)
    fd = eval_at(d)
    used = 2
    while used < max_iter and abs(b - a) > tol * max(abs(a), abs(b)):
        if fc < fd:
            b = d; d = c; fd = fc
            c = b - (b - a) / phi
            fc = eval_at(c); used += 1
        else:
            a = c; c = d; fc = fd
            d = a + (b - a) / phi
            fd = eval_at(d); used += 1
    # Финальное значение.
    x_best = 0.5 * (a + b)
    final_result = parametric_sweep(template, parameter_name, [x_best])[0]
    return {
        "parameter":    parameter_name,
        "target":       target,
        "objective":    objective,
        "best_x":       x_best,
        "best_value":   float(final_result[target]),
        "iterations":   used,
        "converged":    abs(b - a) <= tol * max(abs(a), abs(b)),
        "history":      history,
        "final_T":      final_result["T_field"],
    }


# =============================================================================
# Фазовые переходы (PCM — Phase Change Materials).
# =============================================================================

def effective_cp_with_phase_change(T_field: np.ndarray,
                                     cp_solid: float,
                                     cp_liquid: float,
                                     T_melt: float,
                                     T_mushy_width: float,
                                     latent_heat: float) -> np.ndarray:
    """Эффективная c_p в окрестности фазового перехода (enthalpy method).

    Скрытая теплота плавления L размазывается по интервалу
    [T_melt - δ/2, T_melt + δ/2] (mushy zone) равномерным распределением:
        c_eff(T) = c_p_solid          при T < T_melt - δ/2
                 = c_p_liquid         при T > T_melt + δ/2
                 = c_mix(T) + L/δ     внутри mushy zone

    Метод позволяет учитывать аккумуляцию тепла при плавлении.

    Параметры:
        T_field        — поле температур (N,) в °C или K (одни и те же единицы)
        cp_solid, cp_liquid — теплоёмкости фаз, Дж/(кг·К)
        T_melt         — температура плавления, в тех же единицах
        T_mushy_width  — ширина переходной зоны δ (К). Например, 2-5 К для чистых
                          веществ, до 10-20 К для смесей.
        latent_heat    — скрытая теплота плавления L, Дж/кг

    Возвращает массив (N,) с эффективной c_p в каждой точке.

    Применение: на каждом шаге нестационарной задачи пересчитать ρ·c_p
    и заново собрать масс-матрицу. Это реализовано в специальном цикле
    solve_transient_pcm() (см. ниже).
    """
    T_lo = T_melt - 0.5 * T_mushy_width
    T_hi = T_melt + 0.5 * T_mushy_width
    cp_eff = np.where(T_field < T_lo, cp_solid,
                       np.where(T_field > T_hi, cp_liquid,
                                 0.5 * (cp_solid + cp_liquid)
                                 + latent_heat / T_mushy_width))
    return cp_eff


def _frozen_fraction(T_field: np.ndarray, T_melt: float,
                      T_mushy_width: float) -> np.ndarray:
    """Доля твёрдой фазы в каждой точке (1=твёрдая, 0=жидкая).
    Используется для визуализации PCM-задач."""
    T_lo = T_melt - 0.5 * T_mushy_width
    T_hi = T_melt + 0.5 * T_mushy_width
    frac = np.where(T_field < T_lo, 1.0,
                      np.where(T_field > T_hi, 0.0,
                                (T_hi - T_field) / T_mushy_width))
    return np.clip(frac, 0.0, 1.0)
