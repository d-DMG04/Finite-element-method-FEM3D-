# -*- coding: utf-8 -*-
"""
fem3d.core_bridge
=================

Граница между Python-управляющим слоем и C++ вычислительным ядром.

Реализует то, что в пояснительной записке (раздел 3.2.2) описано как
«самый ответственный модуль управляющего слоя»: тонкий слой ctypes-обёрток
над четырнадцатью C-функциями, экспортируемыми ядром fem_core, плюс
высокоуровневый класс CoreBridge с автоматическим управлением ресурсами.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, byref, c_double, c_int32
from typing import Optional, Tuple

import numpy as np

# =============================================================================
# Загрузка разделяемой библиотеки fem_core.
# =============================================================================
# Стратегия поиска:
#   1) переменная окружения FEM_CORE_LIB (полный путь);
#   2) каталог рядом с этим модулем;
#   3) каталог fem_core/build относительно корня проекта;
#   4) системные пути загрузчика.
# =============================================================================

def _find_library() -> str:
    """Находит файл fem_core.so / fem_core.dll / fem_core.dylib."""
    if sys.platform.startswith("win"):
        names = ("fem_core.dll",)
    elif sys.platform == "darwin":
        names = ("fem_core.dylib", "fem_core.so")
    else:
        names = ("fem_core.so",)

    # 1) явный путь через окружение
    env_path = os.environ.get("FEM_CORE_LIB")
    if env_path and os.path.isfile(env_path):
        return env_path

    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))

    candidates = []
    for name in names:
        candidates.append(os.path.join(here, name))
        candidates.append(os.path.join(project_root, name))
        candidates.append(os.path.join(project_root, "fem_core", name))
        candidates.append(os.path.join(project_root, "fem_core", "build", name))

    for path in candidates:
        if os.path.isfile(path):
            return path

    # последняя надежда — пусть ОС ищет в LD_LIBRARY_PATH/PATH
    return names[0]


_LIB_PATH = _find_library()
try:
    _lib = ctypes.CDLL(_LIB_PATH)
except OSError as exc:  # pragma: no cover
    raise RuntimeError(
        f"Не удалось загрузить разделяемую библиотеку fem_core: {exc}.\n"
        f"Ожидаемое расположение: {_LIB_PATH}.\n"
        "Соберите ядро командой `make build` или укажите путь через "
        "переменную окружения FEM_CORE_LIB."
    ) from exc

# =============================================================================
# Описание сигнатур C-API.
# =============================================================================
# Без этих atypes/restype все аргументы трактуются как int — что приводит к
# тонким ошибкам при работе с указателями и double.
# =============================================================================

_lib.fem_generate_box.argtypes = [c_double] * 6 + [c_int32] * 3
_lib.fem_generate_box.restype = c_int32

_lib.fem_load_mesh.argtypes = [
    POINTER(c_double), c_int32,  # узлы
    POINTER(c_int32),  c_int32,  # элементы
    POINTER(c_int32),            # boundary_nodes (3 на грань)
    POINTER(c_int32),            # boundary_face_ids
    c_int32,                     # n_boundary_faces
]
_lib.fem_load_mesh.restype = c_int32

_lib.fem_get_node_count.argtypes = []
_lib.fem_get_node_count.restype = c_int32

_lib.fem_get_element_count.argtypes = []
_lib.fem_get_element_count.restype = c_int32

_lib.fem_get_boundary_face_count.argtypes = []
_lib.fem_get_boundary_face_count.restype = c_int32

_lib.fem_get_nodes.argtypes = [POINTER(c_double)]
_lib.fem_get_nodes.restype = c_int32

_lib.fem_get_elements.argtypes = [POINTER(c_int32)]
_lib.fem_get_elements.restype = c_int32

_lib.fem_set_material.argtypes = [c_double, c_double]
_lib.fem_set_material.restype = c_int32

_lib.fem_set_boundary_condition.argtypes = [
    c_int32, c_int32, c_double, c_double, c_double, c_double,
]
_lib.fem_set_boundary_condition.restype = c_int32

_lib.fem_solve.argtypes = [c_double, c_int32]
_lib.fem_solve.restype = c_int32

_lib.fem_get_temperature.argtypes = [POINTER(c_double)]
_lib.fem_get_temperature.restype = c_int32

_lib.fem_compute_fluxes.argtypes = [POINTER(c_double)]
_lib.fem_compute_fluxes.restype = c_int32

_lib.fem_get_solver_info.argtypes = [
    POINTER(c_int32), POINTER(c_double), POINTER(c_double), POINTER(c_int32),
]
_lib.fem_get_solver_info.restype = c_int32

_lib.fem_free.argtypes = []
_lib.fem_free.restype = c_int32

# Локальные источники.
_lib.fem_clear_sources.argtypes = []
_lib.fem_clear_sources.restype = c_int32

_lib.fem_add_point_source.argtypes = [c_int32, c_double]
_lib.fem_add_point_source.restype = c_int32

_lib.fem_add_volume_source.argtypes = [c_int32, POINTER(c_double), c_double]
_lib.fem_add_volume_source.restype = c_int32

# Поузельные переопределения Дирихле.
_lib.fem_set_node_dirichlet.argtypes = [c_int32, c_double]
_lib.fem_set_node_dirichlet.restype = c_int32

_lib.fem_clear_node_dirichlet.argtypes = []
_lib.fem_clear_node_dirichlet.restype = c_int32

# =============================================================================
# Константы для типов ГУ и идентификаторов граней.
# =============================================================================
BC_NONE = 0
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_ROBIN = 3

FACE_X_MINUS = 0
FACE_X_PLUS = 1
FACE_Y_MINUS = 2
FACE_Y_PLUS = 3
FACE_Z_MINUS = 4
FACE_Z_PLUS = 5

FACE_NAMES = {
    FACE_X_MINUS: "X−",
    FACE_X_PLUS: "X+",
    FACE_Y_MINUS: "Y−",
    FACE_Y_PLUS: "Y+",
    FACE_Z_MINUS: "Z− (низ)",
    FACE_Z_PLUS: "Z+ (верх)",
}

# Типы подобластей для объёмных источников.
VOLSRC_BOX    = 0
VOLSRC_SPHERE = 1


class CoreError(RuntimeError):
    """Ошибка, возвращённая C++ ядром."""


class SolverInfo:
    """Лёгкая структура с результатом запуска CG."""

    __slots__ = ("iterations", "residual", "time_seconds", "converged")

    def __init__(self, iterations: int, residual: float,
                 time_seconds: float, converged: bool) -> None:
        self.iterations = iterations
        self.residual = residual
        self.time_seconds = time_seconds
        self.converged = converged

    def __repr__(self) -> str:
        return (
            f"SolverInfo(iterations={self.iterations}, "
            f"residual={self.residual:.3e}, "
            f"time={self.time_seconds*1000:.2f} ms, "
            f"converged={self.converged})"
        )


# =============================================================================
# CoreBridge — удобный высокоуровневый интерфейс.
# =============================================================================

class CoreBridge:
    """
    Высокоуровневая обёртка над C-API ядра. Один экземпляр = одна задача.

    Использование:
        bridge = CoreBridge()
        bridge.generate_box(0, 1, 0, 1, 0, 1, 10, 10, 10)
        bridge.set_material(lambda_=237.0, Q=0.0)
        bridge.set_bc(FACE_Z_MINUS, BC_DIRICHLET, T0=0.0)
        bridge.set_bc(FACE_Z_PLUS,  BC_DIRICHLET, T0=100.0)
        info = bridge.solve(tol=1e-8, max_iter=5000)
        T = bridge.get_temperature()
    """

    def __init__(self) -> None:
        self._mesh_ready = False
        self._solved = False

    # --- Утилиты -------------------------------------------------------------
    @staticmethod
    def _check(rc: int, where: str) -> None:
        if rc < 0:
            raise CoreError(f"Ошибка C-API в {where} (код {rc})")

    # --- Сетка ---------------------------------------------------------------
    def generate_box(self, x_min: float, x_max: float,
                     y_min: float, y_max: float,
                     z_min: float, z_max: float,
                     nx: int, ny: int, nz: int) -> None:
        rc = _lib.fem_generate_box(
            float(x_min), float(x_max),
            float(y_min), float(y_max),
            float(z_min), float(z_max),
            int(nx), int(ny), int(nz),
        )
        self._check(rc, "fem_generate_box")
        self._mesh_ready = True
        self._solved = False

    def load_mesh(self, nodes_xyz: np.ndarray, elements: np.ndarray,
                  boundary_nodes: np.ndarray,
                  boundary_face_ids: np.ndarray) -> None:
        """
        Загрузить готовую сетку (для импорта).
            nodes_xyz: (N, 3) float64
            elements:  (Ne, 4) int32
            boundary_nodes: (Nf, 3) int32
            boundary_face_ids: (Nf,) int32
        """
        nodes_xyz = np.ascontiguousarray(nodes_xyz, dtype=np.float64)
        elements = np.ascontiguousarray(elements, dtype=np.int32)
        boundary_nodes = np.ascontiguousarray(boundary_nodes, dtype=np.int32)
        boundary_face_ids = np.ascontiguousarray(boundary_face_ids, dtype=np.int32)

        rc = _lib.fem_load_mesh(
            nodes_xyz.ctypes.data_as(POINTER(c_double)),
            c_int32(nodes_xyz.shape[0]),
            elements.ctypes.data_as(POINTER(c_int32)),
            c_int32(elements.shape[0]),
            boundary_nodes.ctypes.data_as(POINTER(c_int32)),
            boundary_face_ids.ctypes.data_as(POINTER(c_int32)),
            c_int32(boundary_nodes.shape[0]),
        )
        self._check(rc, "fem_load_mesh")
        self._mesh_ready = True
        self._solved = False

    @property
    def n_nodes(self) -> int:
        return int(_lib.fem_get_node_count())

    @property
    def n_elements(self) -> int:
        return int(_lib.fem_get_element_count())

    @property
    def n_boundary_faces(self) -> int:
        return int(_lib.fem_get_boundary_face_count())

    def get_nodes(self) -> np.ndarray:
        """Координаты узлов: (N, 3)."""
        n = self.n_nodes
        if n == 0:
            return np.empty((0, 3), dtype=np.float64)
        buf = np.empty(3 * n, dtype=np.float64)
        rc = _lib.fem_get_nodes(buf.ctypes.data_as(POINTER(c_double)))
        self._check(rc, "fem_get_nodes")
        return buf.reshape(n, 3)

    def get_elements(self) -> np.ndarray:
        """Связность элементов: (Ne, 4)."""
        ne = self.n_elements
        if ne == 0:
            return np.empty((0, 4), dtype=np.int32)
        buf = np.empty(4 * ne, dtype=np.int32)
        rc = _lib.fem_get_elements(buf.ctypes.data_as(POINTER(c_int32)))
        self._check(rc, "fem_get_elements")
        return buf.reshape(ne, 4)

    # --- Физика и ГУ ---------------------------------------------------------
    def set_material(self, lambda_: float, Q: float = 0.0) -> None:
        if lambda_ <= 0.0:
            raise CoreError("Коэффициент теплопроводности должен быть положительным")
        rc = _lib.fem_set_material(float(lambda_), float(Q))
        self._check(rc, "fem_set_material")
        self._solved = False

    def set_bc(self, face_id: int, bc_type: int,
               T0: float = 0.0, q0: float = 0.0,
               alpha: float = 0.0, T_inf: float = 0.0) -> None:
        rc = _lib.fem_set_boundary_condition(
            int(face_id), int(bc_type),
            float(T0), float(q0), float(alpha), float(T_inf),
        )
        self._check(rc, "fem_set_boundary_condition")
        self._solved = False

    # --- Локальные источники тепла ------------------------------------------
    def clear_sources(self) -> None:
        """Очистить списки точечных и объёмных источников."""
        rc = _lib.fem_clear_sources()
        self._check(rc, "fem_clear_sources")

    def add_point_source(self, node_idx: int, power: float) -> None:
        """
        Точечный источник в узле node_idx с мощностью power (Вт).
        Знак power: положительный — нагрев, отрицательный — отвод.
        """
        rc = _lib.fem_add_point_source(int(node_idx), float(power))
        self._check(rc, "fem_add_point_source")

    def add_volume_source_box(self, x_min: float, y_min: float, z_min: float,
                              x_max: float, y_max: float, z_max: float,
                              Q0: float) -> None:
        """Объёмный источник в параллелепипедной подобласти."""
        params = (c_double * 6)(x_min, y_min, z_min, x_max, y_max, z_max)
        rc = _lib.fem_add_volume_source(VOLSRC_BOX, params, float(Q0))
        self._check(rc, "fem_add_volume_source(box)")

    def add_volume_source_sphere(self, cx: float, cy: float, cz: float,
                                 radius: float, Q0: float) -> None:
        """Объёмный источник в сферической подобласти."""
        params = (c_double * 6)(cx, cy, cz, radius, 0.0, 0.0)
        rc = _lib.fem_add_volume_source(VOLSRC_SPHERE, params, float(Q0))
        self._check(rc, "fem_add_volume_source(sphere)")

    # --- Поузельные переопределения Дирихле ---------------------------------
    def set_node_dirichlet(self, node_idx: int, value: float) -> None:
        """Зафиксировать T = value в указанном узле сетки.

        Используется в верификации T3 (раздел 3.4.3 ПЗ) для задания узлового
        профиля Дирихле, а также в практических задачах с известной по
        датчикам поверхностной температурой.
        """
        rc = _lib.fem_set_node_dirichlet(int(node_idx), float(value))
        self._check(rc, "fem_set_node_dirichlet")

    def clear_node_dirichlet(self) -> None:
        """Снять все поузельные переопределения Дирихле."""
        rc = _lib.fem_clear_node_dirichlet()
        self._check(rc, "fem_clear_node_dirichlet")

    def set_node_dirichlet_array(self, node_indices, values) -> None:
        """Удобная массовая версия: задать значения для массива узлов."""
        idx = np.ascontiguousarray(node_indices, dtype=np.int32)
        val = np.ascontiguousarray(values, dtype=np.float64)
        if idx.shape != val.shape:
            raise CoreError("Размеры массивов node_indices и values не совпадают")
        for i in range(idx.size):
            self.set_node_dirichlet(int(idx[i]), float(val[i]))

    # --- Решение -------------------------------------------------------------
    def solve(self, tol: float = 1e-8, max_iter: int = 5000) -> SolverInfo:
        rc = _lib.fem_solve(float(tol), int(max_iter))
        if rc < 0:
            raise CoreError(f"fem_solve вернул ошибку (код {rc})")
        self._solved = True
        return self.solver_info()

    def solver_info(self) -> SolverInfo:
        iters = c_int32(0)
        residual = c_double(0.0)
        t = c_double(0.0)
        conv = c_int32(0)
        rc = _lib.fem_get_solver_info(byref(iters), byref(residual),
                                      byref(t), byref(conv))
        self._check(rc, "fem_get_solver_info")
        return SolverInfo(iters.value, residual.value, t.value, bool(conv.value))

    def get_temperature(self) -> np.ndarray:
        if not self._solved:
            raise CoreError("Решение ещё не получено: вызовите solve() сначала")
        n = self.n_nodes
        buf = np.empty(n, dtype=np.float64)
        rc = _lib.fem_get_temperature(buf.ctypes.data_as(POINTER(c_double)))
        self._check(rc, "fem_get_temperature")
        return buf

    def compute_fluxes(self) -> np.ndarray:
        """Узловые тепловые потоки: (N, 3)."""
        if not self._solved:
            raise CoreError("Решение ещё не получено: вызовите solve() сначала")
        n = self.n_nodes
        buf = np.empty(3 * n, dtype=np.float64)
        rc = _lib.fem_compute_fluxes(buf.ctypes.data_as(POINTER(c_double)))
        self._check(rc, "fem_compute_fluxes")
        return buf.reshape(n, 3)

    # --- Освобождение --------------------------------------------------------
    def free(self) -> None:
        _lib.fem_free()
        self._mesh_ready = False
        self._solved = False

    def __enter__(self) -> "CoreBridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.free()
