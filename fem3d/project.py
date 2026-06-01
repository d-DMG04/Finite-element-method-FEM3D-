# -*- coding: utf-8 -*-
"""
fem3d.project — сохранение и загрузка проектов в файлы .fem3d.

Формат .fem3d — ZIP-архив:
    problem.pkl     — pickle от Problem (геометрия, ГУ, материалы, источники,
                      регионы материалов; импортированная сетка, если есть);
    result.npz      — numpy-массивы T, flux, info (если расчёт выполнен);
    meta.json       — метаданные (версия программы, дата).

Этот формат компактен, переносим и не зависит от ОС.
"""

from __future__ import annotations

import json
import os
import pickle
import zipfile
from datetime import datetime
from typing import Optional

import numpy as np

from .problem import Problem

PROJECT_FORMAT_VERSION = 1
PROJECT_EXTENSION = ".fem3d"


def save_project(problem: Problem, path: str,
                 program_version: str = "1.5",
                 description: str = "") -> None:
    """Сохранить Problem в файл .fem3d."""
    if not path.endswith(PROJECT_EXTENSION):
        path = path + PROJECT_EXTENSION

    # Готовим состояние без тяжёлых numpy-массивов (их положим отдельно).
    problem_state = {
        "geometry":              problem.geometry,
        "lambda_":               problem.lambda_,
        "Q":                     problem.Q,
        "bcs":                   problem.bcs,
        "point_sources":         problem.point_sources,
        "volume_sources":        problem.volume_sources,
        "material_regions":      problem.material_regions,
        # Теплофизические свойства (нестационарная задача).
        "rho":                   getattr(problem, "rho", 0.0),
        "cp":                    getattr(problem, "cp", 0.0),
        # Анизотропия глобального материала.
        "is_anisotropic":        getattr(problem, "is_anisotropic", False),
        "lambda_x":              getattr(problem, "lambda_x", 0.0),
        "lambda_y":              getattr(problem, "lambda_y", 0.0),
        "lambda_z":              getattr(problem, "lambda_z", 0.0),
        "material_name":         getattr(problem, "material_name", ""),
        # Точки наблюдения (виртуальные термопары).
        "observation_points":    list(getattr(problem, "observation_points", [])),
        # Импортированную сетку сохраняем как массивы (см. ниже).
        "has_external_mesh":     problem.has_external_mesh(),
    }

    arrays = {}
    if problem.has_external_mesh():
        arrays["external_nodes"]         = problem.external_nodes
        arrays["external_elements"]      = problem.external_elements
        if problem.external_bnd_nodes is not None:
            arrays["external_bnd_nodes"]    = problem.external_bnd_nodes
            arrays["external_bnd_face_ids"] = problem.external_bnd_face_ids

    has_results = problem.T is not None
    if has_results:
        arrays["nodes"]    = problem.nodes
        arrays["elements"] = problem.elements
        arrays["T"]        = problem.T
        if problem.flux is not None:
            arrays["flux"] = problem.flux

    meta = {
        "format_version": PROJECT_FORMAT_VERSION,
        "program_version": program_version,
        "saved_at":       datetime.now().isoformat(timespec="seconds"),
        "description":    description,
        "has_results":    has_results,
        "has_external_mesh": problem.has_external_mesh(),
    }
    if problem.info is not None:
        meta["solver_iterations"] = int(problem.info.iterations)
        meta["solver_residual"]   = float(problem.info.residual)
        meta["solver_time_s"]     = float(problem.info.time_seconds)
        meta["solver_converged"]  = bool(problem.info.converged)

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        zf.writestr("problem.pkl", pickle.dumps(problem_state, protocol=4))
        if arrays:
            import io
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            zf.writestr("arrays.npz", buf.getvalue())


def load_project(path: str) -> Problem:
    """Загрузить Problem из файла .fem3d."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    with zipfile.ZipFile(path, mode="r") as zf:
        # meta.json
        with zf.open("meta.json") as f:
            meta = json.loads(f.read().decode("utf-8"))
        if meta.get("format_version", 0) != PROJECT_FORMAT_VERSION:
            raise RuntimeError(
                f"Неподдерживаемая версия формата: {meta.get('format_version')}"
            )

        # problem.pkl
        with zf.open("problem.pkl") as f:
            state = pickle.loads(f.read())

        # arrays.npz (опционально).
        arrays = {}
        if "arrays.npz" in zf.namelist():
            with zf.open("arrays.npz") as f:
                import io
                with np.load(io.BytesIO(f.read()), allow_pickle=False) as data:
                    for k in data.files:
                        arrays[k] = data[k]

    problem = Problem(
        geometry=state["geometry"],
        lambda_=state["lambda_"],
        Q=state["Q"],
        bcs=state["bcs"],
    )
    problem.point_sources    = state.get("point_sources", [])
    problem.volume_sources   = state.get("volume_sources", [])
    problem.material_regions = state.get("material_regions", [])
    # Теплофизические свойства и анизотропия.
    problem.rho            = state.get("rho", 0.0)
    problem.cp             = state.get("cp", 0.0)
    problem.is_anisotropic = state.get("is_anisotropic", False)
    problem.lambda_x       = state.get("lambda_x", 0.0)
    problem.lambda_y       = state.get("lambda_y", 0.0)
    problem.lambda_z       = state.get("lambda_z", 0.0)
    problem.material_name  = state.get("material_name", "")
    problem.observation_points = state.get("observation_points", [])

    # Импортированная сетка.
    if state.get("has_external_mesh", False):
        problem.external_nodes        = arrays.get("external_nodes")
        problem.external_elements     = arrays.get("external_elements")
        problem.external_bnd_nodes    = arrays.get("external_bnd_nodes")
        problem.external_bnd_face_ids = arrays.get("external_bnd_face_ids")

    # Результаты расчёта.
    if meta.get("has_results", False):
        problem.nodes    = arrays.get("nodes")
        problem.elements = arrays.get("elements")
        problem.T        = arrays.get("T")
        problem.flux     = arrays.get("flux")
        # Восстановим SolverInfo.
        from .core_bridge import SolverInfo
        problem.info = SolverInfo(
            iterations=meta.get("solver_iterations", 0),
            residual=meta.get("solver_residual", 0.0),
            time_seconds=meta.get("solver_time_s", 0.0),
            converged=meta.get("solver_converged", False),
        )

    return problem


def project_info(path: str) -> dict:
    """Возвращает meta-данные проекта без полной загрузки."""
    with zipfile.ZipFile(path, mode="r") as zf:
        with zf.open("meta.json") as f:
            return json.loads(f.read().decode("utf-8"))
