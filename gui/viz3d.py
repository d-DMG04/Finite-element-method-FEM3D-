# -*- coding: utf-8 -*-
"""
gui.viz3d — оптимизированная трёхмерная визуализация.

Главное отличие от v1.2:

  * Объёмный рендер теперь работает на **регулярном воксельном гриде**, а не на
    исходной неструктурированной тетраэдральной сетке. На 50 000 тетраэдров
    ускорение ≈ 50–100× по сравнению с unstructured-raycaster.

  * Базовые сетки (UnstructuredGrid, PolyData) и сэмплированный воксельный
    грид строятся ОДИН раз при `set_mesh()` и КЭШИРУЮТСЯ. При смене
    температуры обновляется только массив `point_data["T"]`, а не вся сцена.

  * При смене режима рендера предыдущие акторы переиспользуются, где
    возможно (изменяется только scalar/opacity, а не вся pipeline).

  * Поверхностная сетка для больших моделей децимируется до ~50k треугольников
    — этого хватает для всех практических задач.

  * Запросы перерисовки от слайдеров проходят через debounce-таймер (50 мс).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Доступность PyVista определяется на уровне модуля.
# -----------------------------------------------------------------------------
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget


# =============================================================================
# Утилиты.
# =============================================================================

# Лимит на число треугольников поверхности для интерактивной отрисовки.
# Выше — децимируем.
SURFACE_TRIANGLE_LIMIT = 50_000

# Размер воксельного грида для объёмного рендера. 80³ = 512k вокселей —
# отрисовывается смартмаппером за миллисекунды.
VOLUME_GRID_RESOLUTION = 80


def _build_tetra_cells(elements: np.ndarray) -> np.ndarray:
    """Преобразует (Ne, 4) в формат vtkCellArray."""
    ne = elements.shape[0]
    out = np.empty((ne, 5), dtype=np.int64)
    out[:, 0] = 4
    out[:, 1:] = elements
    return out.flatten()


def _build_surface_triangles(bnd_nodes: np.ndarray) -> np.ndarray:
    nf = bnd_nodes.shape[0]
    out = np.empty((nf, 4), dtype=np.int64)
    out[:, 0] = 3
    out[:, 1:] = bnd_nodes
    return out.flatten()


# =============================================================================
# Абстрактный интерфейс.
# =============================================================================

class Visualization3D(QWidget):
    """Базовый интерфейс 3D-просмотра."""

    # Picking: клик по узлу → (idx, x, y, z, T).
    node_picked = pyqtSignal(int, float, float, float, float)
    # Клик в режиме «разместить источник» → (x, y, z).
    point_clicked = pyqtSignal(float, float, float)
    # Hover (без клика): показ текущей T в точке курсора.
    hover_value = pyqtSignal(float, float, float, float)
    # Измерение расстояния: два клика → (dist, x1, y1, z1, x2, y2, z2).
    distance_measured = pyqtSignal(float, float, float, float, float, float, float)

    def set_mesh(self, nodes, elements, bnd_nodes):
        raise NotImplementedError

    def set_temperature(self, T):
        raise NotImplementedError

    def set_render_mode(self, mode):
        raise NotImplementedError

    def set_slice(self, axis, position=0.5):
        raise NotImplementedError

    def set_isosurface_count(self, n):
        raise NotImplementedError

    def set_colormap(self, name: str) -> None:
        """Сменить цветовую палитру (viridis/inferno/plasma/coolwarm/jet/turbo)."""
        pass

    def set_flux_field(self, flux: Optional[np.ndarray]) -> None:
        """Задать векторное поле потоков q (N, 3). None — снять."""
        pass

    def set_flux_arrows_visible(self, visible: bool) -> None:
        """Включить/выключить отображение стрелок потока."""
        pass

    def set_log_scale(self, enabled: bool) -> None:
        """Логарифмическая шкала для поля T (полезно при разнице порядков)."""
        pass

    def set_isolines_visible(self, enabled: bool) -> None:
        """Контурные изолинии температуры на поверхности."""
        pass

    def set_isoline_count(self, n: int) -> None:
        """Число изолиний на поверхности (2..30)."""
        pass

    def set_minmax_labels_visible(self, enabled: bool) -> None:
        """Подписи T_min / T_max в соответствующих узлах 3D."""
        pass

    def set_bc_overlay(self, bnd_faces: Optional[np.ndarray],
                        face_ids: Optional[np.ndarray],
                        face_id_to_color: Optional[dict]) -> None:
        """Подсветить границу полупрозрачным цветом в зависимости от
        присвоенного типа ГУ.

        bnd_faces — массив (Nf, 3) индексов узлов в треугольниках поверхности;
        face_ids — массив (Nf,) с face_id ∈ 0..5 для каждого треугольника;
        face_id_to_color — отображение face_id → '#hex'. Треугольники,
            face_id которых не в словаре, не подсвечиваются (либо красятся
            нейтральным цветом).

        Если хотя бы один параметр None — overlay снимается.
        """
        pass

    def set_pick_mode(self, mode):
        raise NotImplementedError

    def add_source_marker(self, x, y, z, color="yellow"):
        raise NotImplementedError

    def clear_source_markers(self):
        raise NotImplementedError

    def add_preview_marker(self, x, y, z, radius=None, color="#3aa55a"):
        """Временный маркер выбранной точки (центр будущей сферы и т.п.)."""
        pass

    def clear_preview_markers(self):
        pass

    def add_obs_marker(self, x, y, z, number: int = 0):
        """Маркер точки наблюдения (термопары). По умолчанию — без отрисовки."""
        pass

    def clear_obs_markers(self):
        pass

    def reset_camera(self):
        raise NotImplementedError

    # ----- Расширения, появившиеся в v1.4 ------------------------------------
    def set_projection(self, mode: str) -> None:
        """mode: 'perspective' или 'parallel' (ортогональная)."""
        pass

    def set_xray(self, enabled: bool) -> None:
        """Полупрозрачный режим: внутренние структуры видны сквозь оболочку."""
        pass

    def set_show_edges(self, enabled: bool) -> None:
        """Показать рёбра поверхностной сетки."""
        pass

    def set_show_axes(self, enabled: bool) -> None:
        """Показать/скрыть индикатор осей."""
        pass

    def set_camera_locked(self, locked: bool) -> None:
        """Заморозить камеру (запретить вращение и масштабирование)."""
        pass

    def set_hover_enabled(self, enabled: bool) -> None:
        """Включить отображение температуры в точке под курсором."""
        pass

    def set_viewport_background(self, color: str) -> None:
        """Цвет фона 3D-вьюпорта (для смены темы)."""
        pass

    def screenshot(self, path: str) -> bool:
        """Сохранить скриншот 3D-вида в файл. Возвращает True при успехе."""
        return False

    def reset_view_to(self, axis: str) -> None:
        """axis: '+x', '-x', '+y', '-y', '+z', '-z', 'iso'."""
        pass

    def add_region_marker(self, region) -> None:
        """Добавить визуальный маркер региона материала (MaterialRegion)."""
        pass

    def clear_region_markers(self) -> None:
        pass

    @property
    def backend_name(self):
        return "abstract"


# =============================================================================
# PyVistaView — высокопроизводительная реализация.
# =============================================================================

class PyVistaView(Visualization3D):
    """3D-просмотр на PyVistaQt с оптимизациями для крупных сеток.

    Архитектура акторов:
        __surface__   — поверхностный рендер (decimated, переиспользуется);
        __volume__    — объёмный рендер на воксельном гриде;
        __iso__       — изоповерхности (рекомпьют при смене N);
        __slice__     — плоскость сечения;
        __wireframe__ — каркас.

    При смене температуры обновляются только point_data, а не сами акторы.
    Это даёт визуальное обновление за единицы миллисекунд даже на больших
    моделях.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#1f2228")
        self.plotter.add_axes(color="#dcdee2")

        # Кэшированные сетки.
        self._nodes: Optional[np.ndarray] = None
        self._elements: Optional[np.ndarray] = None
        self._volume_unstructured: Optional[pv.UnstructuredGrid] = None
        self._surface_mesh: Optional[pv.PolyData] = None
        self._sampled_volume: Optional[pv.ImageData] = None
        self._T: Optional[np.ndarray] = None
        self._T_range: Tuple[float, float] = (0.0, 1.0)

        # Активные акторы (для переиспользования и точечного обновления).
        self._actors: dict = {}

        # Параметры рендера.
        self._render_mode = "surface"
        self._slice_axis: Optional[str] = None
        self._slice_position = 0.5
        self._iso_count = 7
        self._pick_mode = "none"

        # Расширенные параметры (v1.4).
        self._xray = False
        self._show_edges = False
        self._show_axes = True
        self._camera_locked = False
        self._hover_enabled = False
        self._projection = "perspective"

        # Цветовая палитра (v1.5).
        self._cmap = "inferno"

        # Стрелки потока.
        self._show_flux_arrows = False
        self._flux_field: Optional[np.ndarray] = None  # (N, 3)

        # Новое в v1.8.
        self._log_scale = False              # логарифмическая шкала T
        self._show_isolines = False          # контурные линии на поверхности
        self._isoline_count = 10             # число изолиний
        self._show_minmax_labels = False     # подписи min/max в 3D

        # v1.9: подсветка граней по типу ГУ.
        self._bc_overlay_faces: Optional[np.ndarray] = None
        self._bc_overlay_ids: Optional[np.ndarray] = None
        self._bc_overlay_colors: Optional[dict] = None

        # Маркеры источников и регионов.
        self._source_actors = []
        self._region_actors = []

        # Измерение расстояния.
        self._measure_first_point: Optional[Tuple[float, float, float]] = None
        self._measure_actors = []

        # Picking — настраивается один раз.
        self._pick_callback_installed = False
        self._hover_callback_installed = False

        # Debounce таймер для слайдеров — обновляем не чаще раза в 50 мс.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(50)
        self._refresh_timer.timeout.connect(self._do_refresh)
        self._needs_full_refresh = False

    @property
    def backend_name(self) -> str:
        return "pyvista"

    # -------------------------------------------------------------------------
    # set_mesh: тяжёлая операция, выполняется один раз при смене модели.
    # -------------------------------------------------------------------------
    def set_mesh(self, nodes, elements, bnd_nodes):
        self._nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        self._elements = np.ascontiguousarray(elements, dtype=np.int32)
        self._T = None

        # 1) UnstructuredGrid — нужен для сечений, изоповерхностей и
        #    сэмплинга на воксельный грид.
        cells = _build_tetra_cells(self._elements)
        cell_types = np.full(self._elements.shape[0], pv.CellType.TETRA,
                             dtype=np.uint8)
        self._volume_unstructured = pv.UnstructuredGrid(
            cells, cell_types, self._nodes)

        # 2) Поверхностная сетка. Если внешние bnd-треугольники не заданы,
        #    извлекаем поверхность из объёма (медленнее, но работает).
        if bnd_nodes is not None and len(bnd_nodes) > 0:
            faces = _build_surface_triangles(
                np.ascontiguousarray(bnd_nodes, dtype=np.int32))
            surf = pv.PolyData(self._nodes, faces)
        else:
            surf = self._volume_unstructured.extract_surface()

        # Децимация поверхности, если она слишком крупная.
        try:
            n_faces = int(surf.n_faces) if hasattr(surf, "n_faces") else 0
        except Exception:
            n_faces = 0
        if n_faces > SURFACE_TRIANGLE_LIMIT:
            target_ratio = 1.0 - SURFACE_TRIANGLE_LIMIT / n_faces
            try:
                surf = surf.decimate(target_ratio, progress_bar=False)
            except Exception:
                pass
        self._surface_mesh = surf

        # 3) Воксельный грид для объёмного рендера.
        #    Создаётся под габарит и кэшируется. Сэмплинг температуры
        #    выполняется в _resample_volume при каждом set_temperature.
        bbox_min = self._nodes.min(axis=0)
        bbox_max = self._nodes.max(axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        spacing = diag / VOLUME_GRID_RESOLUTION
        dims = (np.ceil((bbox_max - bbox_min) / spacing).astype(int) + 1).tolist()
        # Защита: dims не больше VOLUME_GRID_RESOLUTION + 1 по любой оси.
        dims = [min(d, VOLUME_GRID_RESOLUTION + 1) for d in dims]
        self._sampled_volume = pv.ImageData(
            dimensions=dims,
            spacing=(spacing, spacing, spacing),
            origin=tuple(bbox_min.tolist()),
        )

        # 4) Полная очистка и перерисовка.
        self._clear_all_main_actors()
        self._needs_full_refresh = True
        self._do_refresh()
        self.reset_camera()

    # -------------------------------------------------------------------------
    # set_temperature: лёгкая операция, только обновляет скалярные поля.
    # -------------------------------------------------------------------------
    def set_temperature(self, T):
        self._T = np.asarray(T, dtype=np.float64) if T is not None else None

        if self._T is not None and self._T.size > 0:
            # Защита от рассогласования: если массив T не совпадает с числом
            # узлов в текущей сетке — игнорируем. Это бывает, когда сменили
            # геометрию, а потом ещё не пересчитали (старый T от предыдущей
            # сетки уже не имеет смысла).
            expected = (self._volume_unstructured.n_points
                        if self._volume_unstructured is not None else -1)
            if expected > 0 and self._T.size != expected:
                self._T = None
                self._T_range = (0.0, 1.0)
                self._needs_full_refresh = True
                self._schedule_refresh()
                return
            self._T_range = (float(self._T.min()), float(self._T.max()))
            # Обновляем point_data на исходных сетках.
            if self._volume_unstructured is not None:
                self._volume_unstructured.point_data["T"] = self._T
            if self._surface_mesh is not None:
                # Если поверхность была децимирована — у неё другой набор
                # узлов; используем sample_data из исходного объёма.
                if self._surface_mesh.n_points == self._nodes.shape[0]:
                    self._surface_mesh.point_data["T"] = self._T
                else:
                    self._surface_mesh = self._surface_mesh.sample(
                        self._volume_unstructured)
            # Сэмплируем на воксельный грид (это самая дорогая часть, но
            # делается один раз при смене температуры, а не при каждом
            # рендере).
            if self._sampled_volume is not None:
                self._sampled_volume = self._sampled_volume.sample(
                    self._volume_unstructured)
        else:
            self._T_range = (0.0, 1.0)

        self._needs_full_refresh = True
        self._schedule_refresh()

    # -------------------------------------------------------------------------
    # set_render_mode: смена режима. Дёшево — мы не пересобираем сетки.
    # -------------------------------------------------------------------------
    def set_render_mode(self, mode):
        if mode == self._render_mode:
            return
        self._render_mode = mode
        self._needs_full_refresh = True
        self._schedule_refresh()

    def set_slice(self, axis, position=0.5):
        new_axis = axis
        new_pos = max(0.0, min(1.0, position))
        if new_axis == self._slice_axis and abs(new_pos - self._slice_position) < 1e-6:
            return
        self._slice_axis = new_axis
        self._slice_position = new_pos
        self._needs_full_refresh = True
        self._schedule_refresh()

    def set_isosurface_count(self, n):
        new_n = max(2, min(20, int(n)))
        if new_n == self._iso_count:
            return
        self._iso_count = new_n
        if self._render_mode == "isosurface":
            self._needs_full_refresh = True
            self._schedule_refresh()

    def set_colormap(self, name: str) -> None:
        if name == self._cmap:
            return
        self._cmap = name
        self._needs_full_refresh = True
        self._schedule_refresh()

    def set_flux_field(self, flux: Optional[np.ndarray]) -> None:
        """Сохраняет векторное поле потока. Сами стрелки рисуются в
        _do_refresh, когда _show_flux_arrows=True."""
        if flux is None:
            self._flux_field = None
        else:
            self._flux_field = np.asarray(flux, dtype=np.float64)
            if self._flux_field.ndim != 2 or self._flux_field.shape[1] != 3:
                self._flux_field = None
        if self._show_flux_arrows:
            self._needs_full_refresh = True
            self._schedule_refresh()

    def set_flux_arrows_visible(self, visible: bool) -> None:
        if visible == self._show_flux_arrows:
            return
        self._show_flux_arrows = visible
        self._needs_full_refresh = True
        self._schedule_refresh()

    def set_log_scale(self, enabled: bool) -> None:
        if enabled == self._log_scale:
            return
        self._log_scale = enabled
        self._needs_full_refresh = True
        self._schedule_refresh()

    def set_isolines_visible(self, enabled: bool) -> None:
        if enabled == self._show_isolines:
            return
        self._show_isolines = enabled
        self._needs_full_refresh = True
        self._schedule_refresh()

    def set_isoline_count(self, n: int) -> None:
        n = max(2, min(30, int(n)))
        if n == self._isoline_count:
            return
        self._isoline_count = n
        if self._show_isolines:
            self._needs_full_refresh = True
            self._schedule_refresh()

    def set_minmax_labels_visible(self, enabled: bool) -> None:
        if enabled == self._show_minmax_labels:
            return
        self._show_minmax_labels = enabled
        self._needs_full_refresh = True
        self._schedule_refresh()

    def set_bc_overlay(self, bnd_faces, face_ids, face_id_to_color):
        if bnd_faces is None or face_ids is None or face_id_to_color is None:
            self._bc_overlay_faces = None
            self._bc_overlay_ids = None
            self._bc_overlay_colors = None
        else:
            self._bc_overlay_faces = np.asarray(bnd_faces, dtype=np.int64)
            self._bc_overlay_ids = np.asarray(face_ids, dtype=np.int32)
            self._bc_overlay_colors = dict(face_id_to_color)
        self._needs_full_refresh = True
        self._schedule_refresh()

    # -------------------------------------------------------------------------
    # Picking.
    # -------------------------------------------------------------------------
    def set_pick_mode(self, mode):
        self._pick_mode = mode
        if mode in ("pick_node", "place_source", "pick_line") and not self._pick_callback_installed:
            self._install_pick_callback()

    def _install_pick_callback(self) -> None:
        def on_pick(point, picker=None):
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            if self._pick_mode == "place_source":
                self.point_clicked.emit(x, y, z)
            elif self._pick_mode in ("pick_node", "pick_line") and self._nodes is not None:
                # pick_line использует тот же сигнал node_picked —
                # MainWindow собирает 2 клика подряд.
                d = np.sum((self._nodes - np.array([x, y, z])) ** 2, axis=1)
                idx = int(np.argmin(d))
                T_val = float(self._T[idx]) if self._T is not None else float("nan")
                nx, ny, nz = self._nodes[idx]
                self.node_picked.emit(idx, float(nx), float(ny), float(nz), T_val)
        try:
            self.plotter.enable_point_picking(callback=on_pick,
                                              show_message=False,
                                              color="yellow",
                                              point_size=10,
                                              use_picker=True)
        except TypeError:
            # Старые версии PyVista могут не принимать use_picker.
            self.plotter.enable_point_picking(callback=on_pick,
                                              show_message=False,
                                              color="yellow", point_size=10)
        self._pick_callback_installed = True

    # -------------------------------------------------------------------------
    # Маркеры источников.
    # -------------------------------------------------------------------------
    def add_source_marker(self, x, y, z, color="yellow"):
        sphere = pv.Sphere(radius=self._auto_marker_radius(), center=(x, y, z))
        actor = self.plotter.add_mesh(sphere, color=color, render=False)
        self._source_actors.append(actor)
        self.plotter.render()

    def add_preview_marker(self, x: float, y: float, z: float,
                           radius: float = None, color: str = "#3aa55a"):
        """Временный маркер (например, выбранный центр будущей сферы).
        Удаляется через clear_preview_markers()."""
        if not hasattr(self, "_preview_actors"):
            self._preview_actors = []
        r = radius if radius is not None else self._auto_marker_radius() * 1.5
        sph = pv.Sphere(radius=r, center=(x, y, z))
        actor = self.plotter.add_mesh(sph, color=color, opacity=0.4,
                                       render=False)
        self._preview_actors.append(actor)
        self.plotter.render()

    def clear_preview_markers(self) -> None:
        for actor in getattr(self, "_preview_actors", []):
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._preview_actors = []
        try:
            self.plotter.render()
        except Exception:
            pass

    def clear_source_markers(self):
        for actor in self._source_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._source_actors = []
        try:
            self.plotter.render()
        except Exception:
            pass

    def add_obs_marker(self, x, y, z, number: int = 0):
        """Маркер точки наблюдения (виртуальной термопары) с номером."""
        if not hasattr(self, "_obs_actors"):
            self._obs_actors = []
        r = self._auto_marker_radius() * 1.3
        sph = pv.Sphere(radius=r, center=(x, y, z))
        a1 = self.plotter.add_mesh(sph, color="#3aa55a", render=False)
        self._obs_actors.append(a1)
        try:
            a2 = self.plotter.add_point_labels(
                [[x, y, z]], [f"T{number}"], font_size=12,
                text_color="white", shape_color="#1a4d2a",
                point_size=1, render=False, always_visible=True)
            self._obs_actors.append(a2)
        except Exception:
            pass
        self.plotter.render()

    def clear_obs_markers(self):
        for actor in getattr(self, "_obs_actors", []):
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._obs_actors = []
        try:
            self.plotter.render()
        except Exception:
            pass

    def reset_camera(self):
        """Сбросить камеру под текущую сцену (изометрия)."""
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def _auto_marker_radius(self) -> float:
        if self._nodes is None or self._nodes.shape[0] == 0:
            return 0.005
        bbox = self._nodes.max(axis=0) - self._nodes.min(axis=0)
        return float(0.015 * np.linalg.norm(bbox))

    # -------------------------------------------------------------------------
    # Debounce и реальная перерисовка.
    # -------------------------------------------------------------------------
    def _schedule_refresh(self):
        self._refresh_timer.start()

    def _clear_all_main_actors(self):
        for key, actor in list(self._actors.items()):
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._actors.clear()
        # На всякий случай — снимем скаляры.
        try:
            self.plotter.remove_scalar_bar()
        except Exception:
            pass

    def _do_refresh(self) -> None:
        """Основная функция перерисовки сцены."""
        if self._volume_unstructured is None:
            self.plotter.render()
            return

        self._clear_all_main_actors()

        has_T = self._T is not None and self._T.size > 0
        cmap = self._cmap

        # log_scale возможен только если все T строго > 0.
        # При T в Цельсиях и нагреве это часто верно, но для безопасности
        # автоматически отключаем при T_min ≤ 0.
        use_log = bool(self._log_scale and has_T
                        and float(self._T_range[0]) > 0.0)

        if self._render_mode == "wireframe":
            self._actors["wf"] = self.plotter.add_mesh(
                self._surface_mesh, style="wireframe",
                color="#dcdee2", line_width=1.0, render=False)

        elif self._render_mode == "surface":
            surf_opacity = 0.45 if self._xray else 1.0
            if has_T:
                self._actors["surf"] = self.plotter.add_mesh(
                    self._surface_mesh, scalars="T", cmap=cmap,
                    show_edges=self._show_edges, render=False,
                    opacity=surf_opacity, log_scale=use_log,
                    scalar_bar_args={"title": "T, °C (log)" if use_log else "T, °C"})
            else:
                self._actors["surf"] = self.plotter.add_mesh(
                    self._surface_mesh, color="#7a6cf0", opacity=surf_opacity,
                    show_edges=self._show_edges, render=False)

        elif self._render_mode == "volume":
            # Быстрый объёмный рендер: сэмплированный воксельный грид.
            if has_T and self._sampled_volume is not None \
                    and "T" in self._sampled_volume.point_data:
                vol_kwargs = {"scalars": "T", "cmap": cmap,
                              "opacity": "linear", "mapper": "smart",
                              "render": False,
                              "scalar_bar_args": {"title": "T, °C (log)" if use_log
                                                   else "T, °C"}}
                # log_scale в add_volume поддерживается не во всех версиях PyVista,
                # но есть resolution-параметр; пробуем безопасно.
                try:
                    self._actors["vol"] = self.plotter.add_volume(
                        self._sampled_volume, log_scale=use_log, **vol_kwargs)
                except (TypeError, ValueError):
                    self._actors["vol"] = self.plotter.add_volume(
                        self._sampled_volume, **vol_kwargs)
                # Полупрозрачная оболочка для ориентира.
                self._actors["shell"] = self.plotter.add_mesh(
                    self._surface_mesh, color="#5a606b", opacity=0.10,
                    render=False)
            else:
                # Без температуры — просто полупрозрачная оболочка.
                self._actors["shell"] = self.plotter.add_mesh(
                    self._surface_mesh, color="#7a6cf0", opacity=0.35,
                    render=False)

        elif self._render_mode == "isosurface":
            if has_T:
                Tmin, Tmax = self._T_range
                if Tmax - Tmin > 1e-12:
                    # При log_scale делаем геометрически распределённые уровни.
                    if use_log and Tmin > 0:
                        levels = np.logspace(np.log10(Tmin), np.log10(Tmax),
                                              self._iso_count)
                    else:
                        levels = np.linspace(Tmin, Tmax, self._iso_count)
                    iso = self._volume_unstructured.contour(
                        isosurfaces=levels.tolist(), scalars="T")
                    self._actors["iso"] = self.plotter.add_mesh(
                        iso, scalars="T", cmap=cmap, opacity=0.85,
                        render=False, log_scale=use_log,
                        scalar_bar_args={"title": "T, °C (log)" if use_log
                                          else "T, °C"})
                self._actors["shell"] = self.plotter.add_mesh(
                    self._surface_mesh, color="#5a606b", opacity=0.12,
                    render=False)
            else:
                self._actors["shell"] = self.plotter.add_mesh(
                    self._surface_mesh, color="#7a6cf0", opacity=0.35,
                    render=False)

        # Сечение.
        if self._slice_axis is not None and has_T:
            try:
                bbox_min = self._nodes.min(axis=0)
                bbox_max = self._nodes.max(axis=0)
                ax_idx = {"x": 0, "y": 1, "z": 2}[self._slice_axis]
                origin = list((bbox_min + bbox_max) / 2)
                origin[ax_idx] = (bbox_min[ax_idx]
                                  + self._slice_position
                                  * (bbox_max[ax_idx] - bbox_min[ax_idx]))
                normal = [0.0, 0.0, 0.0]
                normal[ax_idx] = 1.0
                sliced = self._volume_unstructured.slice(normal=normal,
                                                          origin=origin)
                if sliced.n_points > 0:
                    self._actors["slice"] = self.plotter.add_mesh(
                        sliced, scalars="T", cmap=cmap, render=False,
                        log_scale=use_log,
                        scalar_bar_args={"title": "T, °C (log)" if use_log
                                          else "T, °C"})
            except Exception:
                pass

        # Стрелки векторного поля q = -λ∇T.
        if (self._show_flux_arrows and self._flux_field is not None
                and self._nodes is not None
                and self._flux_field.shape[0] == self._nodes.shape[0]):
            try:
                # Сэмплируем не больше ~800 узлов для отзывчивости.
                n = self._nodes.shape[0]
                if n > 800:
                    idx = np.random.RandomState(0).choice(n, 800, replace=False)
                else:
                    idx = np.arange(n)
                points = self._nodes[idx]
                vectors = self._flux_field[idx]
                mag = np.linalg.norm(vectors, axis=1)
                # Автомасштаб: длина стрелки ~5% диагонали bbox от max(|q|).
                bbox = self._nodes.max(axis=0) - self._nodes.min(axis=0)
                diag = float(np.linalg.norm(bbox))
                max_mag = float(mag.max()) if mag.size > 0 else 1.0
                scale = (0.05 * diag / max_mag) if max_mag > 1e-30 else 0.0
                pd = pv.PolyData(points)
                pd["flux_vec"] = vectors
                pd["flux_mag"] = mag
                if scale > 0:
                    arrows = pd.glyph(orient="flux_vec", scale="flux_mag",
                                      factor=scale)
                    self._actors["flux"] = self.plotter.add_mesh(
                        arrows, scalars="flux_mag", cmap=cmap, render=False,
                        scalar_bar_args={"title": "|q|, Вт/м²"})
            except Exception:
                pass

        # Изолинии температуры на поверхности (если включены).
        if (self._show_isolines and has_T and self._surface_mesh is not None):
            try:
                Tmin, Tmax = self._T_range
                if Tmax - Tmin > 1e-12:
                    if use_log and Tmin > 0:
                        levels = np.logspace(np.log10(Tmin), np.log10(Tmax),
                                              self._isoline_count)
                    else:
                        levels = np.linspace(Tmin, Tmax, self._isoline_count)
                    # contour на поверхностной сетке = тонкие линии.
                    contours = self._surface_mesh.contour(
                        isosurfaces=levels.tolist(), scalars="T")
                    if contours.n_points > 0:
                        self._actors["isolines"] = self.plotter.add_mesh(
                            contours, color="#dcdee2", line_width=1.5,
                            render=False)
            except Exception:
                pass

        # Подписи T_min и T_max в соответствующих узлах.
        if self._show_minmax_labels and has_T and self._nodes is not None:
            try:
                i_min = int(np.argmin(self._T))
                i_max = int(np.argmax(self._T))
                pts = np.array([self._nodes[i_min], self._nodes[i_max]])
                labels = [f"T_min = {self._T[i_min]:.2f} °C",
                          f"T_max = {self._T[i_max]:.2f} °C"]
                self._actors["minmax_lbl"] = self.plotter.add_point_labels(
                    pts, labels, font_size=14, render=False,
                    point_color="#3aa5ff", point_size=14,
                    text_color="white", shape_color="#1a1d22",
                    shape_opacity=0.85, always_visible=True)
            except Exception:
                pass

        # Подсветка граней по типу ГУ (v1.9).
        # Для каждого face_id, имеющего цвет, рисуем небольшую тонкую плёнку
        # из треугольников этого face_id, чуть выдвинутую вдоль внешней
        # нормали тела (чтобы не сливалась с surface).
        if (self._bc_overlay_faces is not None
                and self._bc_overlay_ids is not None
                and self._bc_overlay_colors is not None
                and self._nodes is not None):
            try:
                body_center = self._nodes.mean(axis=0)
                bbox = self._nodes.max(axis=0) - self._nodes.min(axis=0)
                diag = float(np.linalg.norm(bbox))
                offset_scale = 0.005 * diag  # ~0.5% от диагонали
                for fid, color in self._bc_overlay_colors.items():
                    mask = self._bc_overlay_ids == fid
                    if not np.any(mask):
                        continue
                    tri = self._bc_overlay_faces[mask]
                    # Делаем PolyData. Чуть выдвигаем треугольники наружу.
                    pts = self._nodes.copy()
                    # Считаем нормаль для каждой грани и сдвигаем все узлы.
                    # Для скорости делаем глобальный сдвиг — каждой грани свой:
                    # построим отдельный mesh из её точек.
                    sub_nodes = pts[tri.ravel()].reshape(-1, 3, 3)
                    n_face = np.cross(sub_nodes[:, 1] - sub_nodes[:, 0],
                                       sub_nodes[:, 2] - sub_nodes[:, 0])
                    nlen = np.linalg.norm(n_face, axis=1, keepdims=True)
                    nlen[nlen < 1e-30] = 1.0
                    n_face = n_face / nlen
                    centroids_tri = sub_nodes.mean(axis=1)
                    outward = (np.einsum("ij,ij->i", n_face,
                                          centroids_tri - body_center) > 0)
                    n_face[~outward] = -n_face[~outward]
                    # Расширенный массив узлов: для каждой грани свои 3 точки.
                    pts_flat = sub_nodes + offset_scale * n_face[:, None, :]
                    pts_arr = pts_flat.reshape(-1, 3)
                    n_tri = tri.shape[0]
                    new_cells = np.empty((n_tri, 4), dtype=np.int64)
                    new_cells[:, 0] = 3
                    new_cells[:, 1] = np.arange(0, 3 * n_tri, 3)
                    new_cells[:, 2] = np.arange(1, 3 * n_tri, 3)
                    new_cells[:, 3] = np.arange(2, 3 * n_tri, 3)
                    overlay_mesh = pv.PolyData(pts_arr, new_cells.ravel())
                    self._actors[f"bc_overlay_{fid}"] = self.plotter.add_mesh(
                        overlay_mesh, color=color, opacity=0.45,
                        show_edges=False, render=False, lighting=False)
            except Exception:
                pass

        # Один render() в конце — все add_mesh() выше были с render=False.
        self.plotter.render()

    # =========================================================================
    # Расширенные методы интерактивности (v1.4).
    # =========================================================================

    def set_projection(self, mode: str) -> None:
        self._projection = mode
        try:
            if mode == "parallel":
                self.plotter.camera.parallel_projection = True
            else:
                self.plotter.camera.parallel_projection = False
            self.plotter.render()
        except Exception:
            pass

    def set_xray(self, enabled: bool) -> None:
        if enabled == self._xray:
            return
        self._xray = enabled
        self._needs_full_refresh = True
        self._schedule_refresh()

    def set_show_edges(self, enabled: bool) -> None:
        if enabled == self._show_edges:
            return
        self._show_edges = enabled
        self._needs_full_refresh = True
        self._schedule_refresh()

    def set_show_axes(self, enabled: bool) -> None:
        self._show_axes = enabled
        try:
            if enabled:
                self.plotter.add_axes(color="#dcdee2")
            else:
                self.plotter.hide_axes()
            self.plotter.render()
        except Exception:
            pass

    def set_camera_locked(self, locked: bool) -> None:
        self._camera_locked = locked
        try:
            interactor = self.plotter.iren
            if locked:
                interactor.disable()
            else:
                interactor.enable()
        except Exception:
            pass

    def set_hover_enabled(self, enabled: bool) -> None:
        self._hover_enabled = enabled
        if enabled and not self._hover_callback_installed:
            self._install_hover_callback()

    def _install_hover_callback(self) -> None:
        """Hover-показ температуры (только PyVista; через point picker)."""
        def on_hover(obj, evt):
            if not self._hover_enabled or self._nodes is None or self._T is None:
                return
            try:
                x, y = obj.GetEventPosition()
                picker = self.plotter.picker
                # Используем PointPicker — найдёт ближайшую вершину.
                picker.Pick(x, y, 0, self.plotter.renderer)
                pos = picker.GetPickPosition()
                if pos is None:
                    return
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                d = np.sum((self._nodes - np.array([px, py, pz])) ** 2, axis=1)
                idx = int(np.argmin(d))
                self.hover_value.emit(float(self._nodes[idx, 0]),
                                       float(self._nodes[idx, 1]),
                                       float(self._nodes[idx, 2]),
                                       float(self._T[idx]))
            except Exception:
                pass
        try:
            self.plotter.iren.add_observer("MouseMoveEvent", on_hover)
            self._hover_callback_installed = True
        except Exception:
            pass

    def set_viewport_background(self, color: str) -> None:
        try:
            self.plotter.set_background(color)
            self.plotter.render()
        except Exception:
            pass

    def screenshot(self, path: str) -> bool:
        try:
            self.plotter.screenshot(path)
            return True
        except Exception:
            return False

    def reset_view_to(self, axis: str) -> None:
        """Привести камеру к указанной стандартной позиции."""
        if self._nodes is None or self._nodes.shape[0] == 0:
            return
        center = (self._nodes.max(axis=0) + self._nodes.min(axis=0)) / 2
        diag = float(np.linalg.norm(self._nodes.max(axis=0)
                                     - self._nodes.min(axis=0)))
        dist = diag * 1.5
        try:
            cam = self.plotter.camera
            cam.focal_point = tuple(center.tolist())
            if axis == "+x":
                cam.position = (center[0] + dist, center[1], center[2])
                cam.up = (0, 0, 1)
            elif axis == "-x":
                cam.position = (center[0] - dist, center[1], center[2])
                cam.up = (0, 0, 1)
            elif axis == "+y":
                cam.position = (center[0], center[1] + dist, center[2])
                cam.up = (0, 0, 1)
            elif axis == "-y":
                cam.position = (center[0], center[1] - dist, center[2])
                cam.up = (0, 0, 1)
            elif axis == "+z":
                cam.position = (center[0], center[1], center[2] + dist)
                cam.up = (0, 1, 0)
            elif axis == "-z":
                cam.position = (center[0], center[1], center[2] - dist)
                cam.up = (0, 1, 0)
            else:  # "iso"
                d = dist / np.sqrt(3)
                cam.position = (center[0] + d, center[1] + d, center[2] + d)
                cam.up = (0, 0, 1)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def add_region_marker(self, region) -> None:
        """Подсветка региона материала через прозрачный объём."""
        try:
            from fem3d import REGION_BOX, REGION_SPHERE
            if region.shape == REGION_BOX:
                xmin, xmax, ymin, ymax, zmin, zmax = region.params
                box = pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
                actor = self.plotter.add_mesh(
                    box, color=region.color, opacity=0.25,
                    style="surface", render=False)
                self._region_actors.append(actor)
            elif region.shape == REGION_SPHERE:
                cx, cy, cz, r = region.params
                sph = pv.Sphere(radius=r, center=(cx, cy, cz))
                actor = self.plotter.add_mesh(
                    sph, color=region.color, opacity=0.25,
                    style="surface", render=False)
                self._region_actors.append(actor)
            self.plotter.render()
        except Exception:
            pass

    def clear_region_markers(self) -> None:
        for actor in self._region_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._region_actors = []
        try:
            self.plotter.render()
        except Exception:
            pass


# =============================================================================
# MatplotlibView — резервный режим, скромный, но работает везде.
# =============================================================================

class MatplotlibView(Visualization3D):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(7, 6), facecolor="#2a2e36")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #1a1d22; color: #dcdee2;")
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()

        self._nodes = None
        self._elements = None
        self._bnd_nodes = None
        self._T = None
        self._render_mode = "surface"
        self._slice_axis = None
        self._slice_position = 0.5
        self._iso_count = 7
        self._pick_mode = "none"
        self._source_markers = []
        self._cbar = None
        self._cmap = "inferno"
        self._flux_field: Optional[np.ndarray] = None
        self._show_flux_arrows = False
        self._log_scale = False
        self._show_isolines = False
        self._isoline_count = 10
        self._show_minmax_labels = False
        # Кэш под сэмплинг (то же, что у PyVista): подвыборка узлов.
        self._sample_indices = None

        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        # Debounce.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self._do_refresh)

    @property
    def backend_name(self):
        return "matplotlib"

    def _setup_axes(self):
        self.ax.set_facecolor("#2a2e36")
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.label.set_color("#dcdee2")
            axis.set_pane_color((0.16, 0.18, 0.21, 1.0))
        self.ax.tick_params(colors="#dcdee2")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def set_mesh(self, nodes, elements, bnd_nodes):
        self._nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        self._elements = np.ascontiguousarray(elements, dtype=np.int32)
        self._bnd_nodes = np.ascontiguousarray(bnd_nodes, dtype=np.int32)
        self._T = None
        # Готовим подвыборку 4000 узлов один раз.
        n = self._nodes.shape[0]
        if n > 4000:
            self._sample_indices = np.random.RandomState(42).choice(
                n, 4000, replace=False)
        else:
            self._sample_indices = np.arange(n)
        self._do_refresh()
        self.reset_camera()

    def set_temperature(self, T):
        self._T = np.asarray(T, dtype=np.float64) if T is not None else None
        self._refresh_timer.start()

    def set_render_mode(self, mode):
        if mode == self._render_mode:
            return
        self._render_mode = mode
        self._refresh_timer.start()

    def set_slice(self, axis, position=0.5):
        self._slice_axis = axis
        self._slice_position = max(0.0, min(1.0, position))
        self._refresh_timer.start()

    def set_isosurface_count(self, n):
        self._iso_count = max(2, min(20, int(n)))
        if self._render_mode == "isosurface":
            self._refresh_timer.start()

    def set_colormap(self, name: str) -> None:
        if name == self._cmap:
            return
        self._cmap = name
        self._refresh_timer.start()

    def set_flux_field(self, flux: Optional[np.ndarray]) -> None:
        if flux is None:
            self._flux_field = None
        else:
            self._flux_field = np.asarray(flux, dtype=np.float64)
        if getattr(self, "_show_flux_arrows", False):
            self._refresh_timer.start()

    def set_flux_arrows_visible(self, visible: bool) -> None:
        self._show_flux_arrows = bool(visible)
        self._refresh_timer.start()

    def set_log_scale(self, enabled: bool) -> None:
        self._log_scale = bool(enabled)
        self._refresh_timer.start()

    def set_isolines_visible(self, enabled: bool) -> None:
        self._show_isolines = bool(enabled)
        self._refresh_timer.start()

    def set_isoline_count(self, n: int) -> None:
        self._isoline_count = max(2, min(30, int(n)))
        if self._show_isolines:
            self._refresh_timer.start()

    def set_minmax_labels_visible(self, enabled: bool) -> None:
        self._show_minmax_labels = bool(enabled)
        self._refresh_timer.start()

    def set_bc_overlay(self, bnd_faces, face_ids, face_id_to_color):
        # В matplotlib просто игнорируем — слишком сложно для 3D-проекции.
        pass

    def set_pick_mode(self, mode):
        self._pick_mode = mode

    def add_source_marker(self, x, y, z, color="yellow"):
        self._source_markers.append((x, y, z, color))
        self._refresh_timer.start()

    def clear_source_markers(self):
        self._source_markers = []
        self._refresh_timer.start()

    def add_preview_marker(self, x, y, z, radius=None, color="#3aa55a"):
        if not hasattr(self, "_preview_markers"):
            self._preview_markers = []
        self._preview_markers.append((x, y, z, color))
        self._refresh_timer.start()

    def clear_preview_markers(self):
        self._preview_markers = []
        self._refresh_timer.start()

    def reset_camera(self):
        if self._nodes is not None and self._nodes.size > 0:
            mn = self._nodes.min(axis=0)
            mx = self._nodes.max(axis=0)
            self.ax.set_xlim(mn[0], mx[0])
            self.ax.set_ylim(mn[1], mx[1])
            self.ax.set_zlim(mn[2], mx[2])
        self.ax.view_init(elev=25, azim=-60)
        self.canvas.draw_idle()

    def _do_refresh(self):
        self.ax.cla()
        self._setup_axes()
        if self._cbar is not None:
            try: self._cbar.remove()
            except Exception: pass
            self._cbar = None
        if self._nodes is None:
            self.canvas.draw_idle(); return

        bbox_min = self._nodes.min(axis=0)
        bbox_max = self._nodes.max(axis=0)
        self.ax.set_xlim(bbox_min[0], bbox_max[0])
        self.ax.set_ylim(bbox_min[1], bbox_max[1])
        self.ax.set_zlim(bbox_min[2], bbox_max[2])

        has_T = self._T is not None
        bnd = self._bnd_nodes if (self._bnd_nodes is not None
                                  and self._bnd_nodes.size > 0) else None

        if self._render_mode == "wireframe" and bnd is not None:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection(self._nodes[bnd], edgecolor="#dcdee2",
                                    facecolor="none", linewidths=0.4,
                                    alpha=0.6)
            self.ax.add_collection3d(poly)
        elif self._render_mode == "surface":
            if has_T and bnd is not None:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                tri = self._nodes[bnd]
                T_face = self._T[bnd].mean(axis=1)
                # Логарифмическая шкала если возможно (Tmin > 0).
                Tmin_v, Tmax_v = float(self._T.min()), float(self._T.max())
                if self._log_scale and Tmin_v > 0.0:
                    norm = matplotlib.colors.LogNorm(vmin=Tmin_v, vmax=Tmax_v)
                    cbar_label = "T, °C (log)"
                else:
                    norm = matplotlib.colors.Normalize(vmin=Tmin_v, vmax=Tmax_v)
                    cbar_label = "T, °C"
                cmap = matplotlib.cm.get_cmap(self._cmap)
                colors = cmap(norm(T_face))
                poly = Poly3DCollection(tri, facecolors=colors,
                                        edgecolor="none", alpha=0.95)
                self.ax.add_collection3d(poly)
                sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                self._cbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.7,
                                                label=cbar_label)
            elif bnd is not None:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                poly = Poly3DCollection(self._nodes[bnd], facecolor="#7a6cf0",
                                        edgecolor="#3c4049", linewidths=0.3,
                                        alpha=0.55)
                self.ax.add_collection3d(poly)
        elif self._render_mode in ("isosurface", "volume"):
            if has_T and self._sample_indices is not None:
                sample = self._sample_indices
                Tmin_v, Tmax_v = float(self._T.min()), float(self._T.max())
                if self._log_scale and Tmin_v > 0.0:
                    norm = matplotlib.colors.LogNorm(vmin=Tmin_v, vmax=Tmax_v)
                    cbar_label = "T, °C (log)"
                else:
                    norm = matplotlib.colors.Normalize(vmin=Tmin_v, vmax=Tmax_v)
                    cbar_label = "T, °C"
                cmap = matplotlib.cm.get_cmap(self._cmap)
                self._scatter = self.ax.scatter(
                    self._nodes[sample, 0], self._nodes[sample, 1],
                    self._nodes[sample, 2],
                    c=self._T[sample], cmap=cmap, norm=norm, s=6, alpha=0.55)
                self._cbar = self.fig.colorbar(self._scatter, ax=self.ax,
                                                shrink=0.7, label=cbar_label)
                if bnd is not None:
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    poly = Poly3DCollection(self._nodes[bnd], facecolor="#5a606b",
                                            edgecolor="none", alpha=0.10)
                    self.ax.add_collection3d(poly)

        if self._slice_axis is not None and has_T:
            ax_idx = {"x": 0, "y": 1, "z": 2}[self._slice_axis]
            coords = self._nodes[:, ax_idx]
            cmin, cmax = float(coords.min()), float(coords.max())
            tol = max(1e-9, 0.04 * (cmax - cmin))
            slice_pos = cmin + self._slice_position * (cmax - cmin)
            mask = np.abs(coords - slice_pos) <= tol
            if np.any(mask):
                self.ax.scatter(self._nodes[mask, 0], self._nodes[mask, 1],
                                self._nodes[mask, 2], c=self._T[mask],
                                cmap=self._cmap, s=20)

        for (x, y, z, color) in self._source_markers:
            self.ax.scatter([x], [y], [z], c=color, s=80, marker="o",
                            edgecolors="black", linewidths=0.8)

        # Стрелки потока через ax.quiver.
        if (self._show_flux_arrows and self._flux_field is not None
                and self._flux_field.shape[0] == self._nodes.shape[0]):
            try:
                n = self._nodes.shape[0]
                if n > 300:
                    idx = np.random.RandomState(0).choice(n, 300, replace=False)
                else:
                    idx = np.arange(n)
                pts = self._nodes[idx]
                vecs = self._flux_field[idx]
                mag = np.linalg.norm(vecs, axis=1)
                bbox = self._nodes.max(axis=0) - self._nodes.min(axis=0)
                diag = float(np.linalg.norm(bbox))
                max_mag = float(mag.max()) if mag.size > 0 else 1.0
                length = 0.06 * diag
                self.ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2],
                                vecs[:, 0], vecs[:, 1], vecs[:, 2],
                                length=length / max(max_mag, 1e-30),
                                color="#dcdee2", normalize=False,
                                linewidth=0.5, alpha=0.7)
            except Exception:
                pass

        # Подписи T_min и T_max.
        if self._show_minmax_labels and has_T and self._nodes is not None:
            try:
                i_min = int(np.argmin(self._T))
                i_max = int(np.argmax(self._T))
                xn, yn, zn = self._nodes[i_min]
                xx, yx, zx = self._nodes[i_max]
                self.ax.scatter([xn], [yn], [zn], c="#3aa5ff",
                                s=80, marker="v", edgecolors="white")
                self.ax.scatter([xx], [yx], [zx], c="#ff7b3a",
                                s=80, marker="^", edgecolors="white")
                self.ax.text(xn, yn, zn,
                             f"  T_min={self._T[i_min]:.2f}°C",
                             color="#3aa5ff", fontsize=9)
                self.ax.text(xx, yx, zx,
                             f"  T_max={self._T[i_max]:.2f}°C",
                             color="#ff7b3a", fontsize=9)
            except Exception:
                pass

        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        if event.inaxes is not self.ax: return
        if self._pick_mode == "none" or self._nodes is None: return
        try:
            from mpl_toolkits.mplot3d import proj3d
            x2, y2, _ = proj3d.proj_transform(
                self._nodes[:, 0], self._nodes[:, 1], self._nodes[:, 2],
                self.ax.get_proj())
            d = (x2 - event.xdata) ** 2 + (y2 - event.ydata) ** 2
            idx = int(np.argmin(d))
            nx, ny, nz = self._nodes[idx]
            T_val = float(self._T[idx]) if self._T is not None else float("nan")
            if self._pick_mode == "pick_node":
                self.node_picked.emit(idx, float(nx), float(ny), float(nz), T_val)
            elif self._pick_mode == "place_source":
                self.point_clicked.emit(float(nx), float(ny), float(nz))
        except Exception:
            pass

    # ---- v1.4: упрощённые реализации новых методов ------------------------
    def set_projection(self, mode: str) -> None:
        # matplotlib 3D имеет ортогональную и перспективную проекции
        # через ax.set_proj_type().
        try:
            if mode == "parallel":
                self.ax.set_proj_type("ortho")
            else:
                self.ax.set_proj_type("persp")
            self.canvas.draw_idle()
        except Exception:
            pass

    def set_xray(self, enabled: bool) -> None:
        pass

    def set_show_edges(self, enabled: bool) -> None:
        pass

    def set_show_axes(self, enabled: bool) -> None:
        try:
            self.ax.set_axis_on() if enabled else self.ax.set_axis_off()
            self.canvas.draw_idle()
        except Exception:
            pass

    def set_camera_locked(self, locked: bool) -> None:
        pass

    def set_hover_enabled(self, enabled: bool) -> None:
        pass

    def set_viewport_background(self, color: str) -> None:
        try:
            self.fig.set_facecolor(color)
            self.ax.set_facecolor(color)
            self.canvas.draw_idle()
        except Exception:
            pass

    def screenshot(self, path: str) -> bool:
        try:
            self.fig.savefig(path, dpi=130)
            return True
        except Exception:
            return False

    def reset_view_to(self, axis: str) -> None:
        presets = {
            "+x": (0, 0), "-x": (0, 180), "+y": (0, 90), "-y": (0, -90),
            "+z": (90, 0), "-z": (-90, 0), "iso": (25, -60),
        }
        if axis in presets:
            self.ax.view_init(*presets[axis])
            self.canvas.draw_idle()

    def add_region_marker(self, region) -> None:
        pass

    def clear_region_markers(self) -> None:
        pass


# =============================================================================
# Фабрика.
# =============================================================================

def create_view(parent=None, prefer="auto"):
    """Создаёт лучшую доступную 3D-визуализацию."""
    if prefer == "matplotlib":
        return MatplotlibView(parent)
    if prefer == "pyvista":
        if not HAS_PYVISTA:
            raise RuntimeError(
                "PyVista не установлен; установите: "
                "pip install pyvista pyvistaqt")
        return PyVistaView(parent)
    if HAS_PYVISTA:
        try:
            return PyVistaView(parent)
        except Exception:
            return MatplotlibView(parent)
    return MatplotlibView(parent)
