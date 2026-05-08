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
    BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN,
    CoreBridge, FACE_NAMES, SolverInfo,
    VOLSRC_BOX, VOLSRC_SPHERE,
)


# =============================================================================
# Описание граничного условия.
# =============================================================================

@dataclass
class BoundaryCondition:
    type: int = BC_NONE         # один из BC_*
    T0: float = 0.0             # для Dirichlet, °C
    q0: float = 0.0             # для Neumann, Вт/м²
    alpha: float = 0.0          # для Robin, Вт/(м²·К)
    T_inf: float = 20.0         # для Robin, °C

    def description(self) -> str:
        """Краткое текстовое описание для отображения в карточке."""
        if self.type == BC_DIRICHLET:
            return f"Дирихле: T = {self.T0:g} °C"
        if self.type == BC_NEUMANN:
            if abs(self.q0) < 1e-15:
                return "Нейман: ∂T/∂n = 0 (изоляция)"
            return f"Нейман: q = {self.q0:g} Вт/м²"
        if self.type == BC_ROBIN:
            return f"Робен: α = {self.alpha:g}, T∞ = {self.T_inf:g} °C"
        return "Не задано"


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
# Шаблоны граничных условий (Ф3.4 ТЗ).
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


def template_all_convection() -> Dict[int, BoundaryCondition]:
    """Все шесть граней — конвекция с воздухом 20 °C."""
    return {
        f: BoundaryCondition(type=BC_ROBIN, alpha=10.0, T_inf=20.0)
        for f in range(6)
    }


def template_reset() -> Dict[int, BoundaryCondition]:
    """Сбросить все условия в «не задано»."""
    return {f: BoundaryCondition() for f in range(6)}


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

    # --- Локальные источники (раздел 3.3.11 ПЗ) -----------------------------
    point_sources: list = field(default_factory=list)   # list[PointSource]
    volume_sources: list = field(default_factory=list)  # list[VolumeSource]

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

    def push_to_core(self, bridge: CoreBridge) -> None:
        """Передать материал, ГУ и локальные источники в ядро."""
        bridge.set_material(self.lambda_, self.Q)
        for face_id in range(6):
            bc = self.bcs[face_id]
            bridge.set_bc(
                face_id, bc.type,
                T0=bc.T0, q0=bc.q0,
                alpha=bc.alpha, T_inf=bc.T_inf,
            )
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
              tol: float = 1e-8, max_iter: int = 5000) -> SolverInfo:
        """Полный цикл: материал + ГУ → CG → результаты."""
        self.push_to_core(bridge)
        info = bridge.solve(tol=tol, max_iter=max_iter)
        self.T = bridge.get_temperature()
        self.flux = bridge.compute_fluxes()
        self.info = info
        return info

    # =========================================================================
    # Утилиты для GUI.
    # =========================================================================

    def temperature_range(self) -> tuple[float, float]:
        if self.T is None or self.T.size == 0:
            return (0.0, 0.0)
        return float(self.T.min()), float(self.T.max())

    def hot_spot(self) -> Optional[tuple[int, float, float, float]]:
        """(индекс узла, x, y, z) узла с максимальной температурой, если есть."""
        if self.T is None or self.nodes is None:
            return None
        idx = int(np.argmax(self.T))
        x, y, z = self.nodes[idx]
        return idx, float(x), float(y), float(z)
