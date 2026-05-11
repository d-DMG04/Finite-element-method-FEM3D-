# -*- coding: utf-8 -*-
"""
gui.viz3d — трёхмерная визуализация модели и температурного поля.

Архитектура:
    Visualization3D — абстрактный интерфейс QWidget с методами:
        set_mesh(nodes, elements, bnd_nodes, bnd_face_ids)
        set_temperature(T)
        set_render_mode("surface" | "isosurface" | "volume")
        set_slice(axis: str | None, position: float)
        enable_pick(callback) — клик по узлу → callback(node_idx, x, y, z, T)

    Реализации:
        PyVistaView      — лучший рендер, требует pyvista + pyvistaqt;
        MatplotlibView   — резерв, всегда доступен.

    create_view(parent) — фабрика, возвращает PyVistaView если возможно.
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

# matplotlib присутствует всегда (требование в requirements.txt).
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget


# =============================================================================
# Утилиты, общие для обеих реализаций.
# =============================================================================

def _build_surface_triangles(bnd_nodes: np.ndarray) -> np.ndarray:
    """Преобразует (Nf, 3) в формат vtkCellArray: [3, n1, n2, n3, 3, ...]."""
    nf = bnd_nodes.shape[0]
    out = np.empty((nf, 4), dtype=np.int64)
    out[:, 0] = 3
    out[:, 1:] = bnd_nodes
    return out.flatten()


def _build_tetra_cells(elements: np.ndarray) -> np.ndarray:
    """Преобразует (Ne, 4) в формат vtkCellArray для тетраэдров."""
    ne = elements.shape[0]
    out = np.empty((ne, 5), dtype=np.int64)
    out[:, 0] = 4
    out[:, 1:] = elements
    return out.flatten()


# =============================================================================
# Абстрактный QWidget — интерфейс, реализуемый подклассами.
# =============================================================================

class Visualization3D(QWidget):
    """Базовый интерфейс. Подклассы должны переопределить все публичные методы."""

    # Сигнал испускается, когда пользователь кликнул по узлу в режиме picking.
    node_picked = pyqtSignal(int, float, float, float, float)
    # Аргументы: node_idx, x, y, z, T (T = NaN, если решение ещё не получено).

    # Сигнал: пользователь кликнул в режиме «добавить точечный источник».
    point_clicked = pyqtSignal(float, float, float)
    # Аргументы: x, y, z мирового пространства.

    def set_mesh(self, nodes: np.ndarray, elements: np.ndarray,
                 bnd_nodes: np.ndarray) -> None:
        raise NotImplementedError

    def set_temperature(self, T: Optional[np.ndarray]) -> None:
        raise NotImplementedError

    def set_render_mode(self, mode: str) -> None:
        """mode: 'surface', 'isosurface', 'volume', 'wireframe'."""
        raise NotImplementedError

    def set_slice(self, axis: Optional[str], position: float = 0.5) -> None:
        """axis: 'x', 'y', 'z' или None для отключения сечения."""
        raise NotImplementedError

    def set_isosurface_count(self, n: int) -> None:
        """Число изоповерхностей (для режима 'isosurface')."""
        raise NotImplementedError

    def set_pick_mode(self, mode: str) -> None:
        """mode: 'none', 'pick_node', 'place_source'."""
        raise NotImplementedError

    def add_source_marker(self, x: float, y: float, z: float,
                          color: str = "yellow") -> None:
        """Добавить визуальный маркер локального источника."""
        raise NotImplementedError

    def clear_source_markers(self) -> None:
        raise NotImplementedError

    def reset_camera(self) -> None:
        raise NotImplementedError

    @property
    def backend_name(self) -> str:
        return "abstract"


# =============================================================================
# Реализация на PyVista — лучший рендер.
# =============================================================================

class PyVistaView(Visualization3D):
    """Полноценный 3D-просмотр через PyVistaQt.

    Возможности:
      - Поверхностный рендер с цветом по температуре;
      - Volume rendering — полупрозрачное температурное поле внутри объёма;
      - Изоповерхности (5–15 уровней температуры);
      - Сечения по любой оси с ползунком;
      - Picking: клик по узлу → излучается node_picked;
      - Размещение источника кликом → излучается point_clicked;
      - Стандартное OpenGL-управление: ЛКМ — вращение, СКМ — панорамирование,
        колесо — масштаб.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # Состояние модели.
        self._nodes: Optional[np.ndarray] = None
        self._elements: Optional[np.ndarray] = None
        self._bnd_nodes: Optional[np.ndarray] = None
        self._T: Optional[np.ndarray] = None
        self._volume_grid: Optional[pv.UnstructuredGrid] = None
        self._surface_mesh: Optional[pv.PolyData] = None

        # Параметры рендера.
        self._render_mode = "surface"
        self._slice_axis: Optional[str] = None
        self._slice_position = 0.5  # доля габарита
        self._iso_count = 7
        self._pick_mode = "none"

        # Markers.
        self._source_actors = []

        # Тёмный фон.
        self.plotter.set_background("#1f2228")
        self.plotter.add_axes(color="#dcdee2")

        # Picking — настраивается лениво при первом включении.
        self._pick_callback_installed = False

    @property
    def backend_name(self) -> str:
        return "pyvista"

    # -------------------------------------------------------------------------
    def set_mesh(self, nodes: np.ndarray, elements: np.ndarray,
                 bnd_nodes: np.ndarray) -> None:
        self._nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        self._elements = np.ascontiguousarray(elements, dtype=np.int32)
        self._bnd_nodes = np.ascontiguousarray(bnd_nodes, dtype=np.int32)
        self._T = None

        # Объёмная сетка для volume / isosurface рендера.
        cells = _build_tetra_cells(self._elements)
        cell_types = np.full(self._elements.shape[0], pv.CellType.TETRA, dtype=np.uint8)
        self._volume_grid = pv.UnstructuredGrid(cells, cell_types, self._nodes)

        # Поверхностная сетка для быстрого рендера.
        if self._bnd_nodes.size > 0:
            faces = _build_surface_triangles(self._bnd_nodes)
            self._surface_mesh = pv.PolyData(self._nodes, faces)
        else:
            # Если нет граничных треугольников, извлекаем поверхность из объёма.
            self._surface_mesh = self._volume_grid.extract_surface()

        self._refresh_view()
        self.reset_camera()

    def set_temperature(self, T: Optional[np.ndarray]) -> None:
        self._T = np.asarray(T, dtype=np.float64) if T is not None else None
        if self._T is not None:
            if self._volume_grid is not None:
                self._volume_grid["T"] = self._T
            if self._surface_mesh is not None and self._nodes is not None:
                self._surface_mesh["T"] = self._T
        self._refresh_view()

    def set_render_mode(self, mode: str) -> None:
        self._render_mode = mode
        self._refresh_view()

    def set_slice(self, axis: Optional[str], position: float = 0.5) -> None:
        self._slice_axis = axis
        self._slice_position = max(0.0, min(1.0, position))
        self._refresh_view()

    def set_isosurface_count(self, n: int) -> None:
        self._iso_count = max(2, min(20, int(n)))
        if self._render_mode == "isosurface":
            self._refresh_view()

    def set_pick_mode(self, mode: str) -> None:
        self._pick_mode = mode
        if mode in ("pick_node", "place_source") and not self._pick_callback_installed:
            self._install_pick_callback()

    def _install_pick_callback(self) -> None:
        """Устанавливает обработчик клика мыши."""
        def on_pick(point, picker):
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            if self._pick_mode == "place_source":
                self.point_clicked.emit(x, y, z)
            elif self._pick_mode == "pick_node" and self._nodes is not None:
                # Находим ближайший узел.
                d = np.sum((self._nodes - np.array([x, y, z])) ** 2, axis=1)
                idx = int(np.argmin(d))
                T_val = float(self._T[idx]) if self._T is not None else float("nan")
                nx, ny, nz = self._nodes[idx]
                self.node_picked.emit(idx, float(nx), float(ny), float(nz), T_val)
        self.plotter.enable_point_picking(callback=on_pick, show_message=False,
                                          color="yellow", point_size=10,
                                          use_picker=True)
        self._pick_callback_installed = True

    def add_source_marker(self, x: float, y: float, z: float,
                          color: str = "yellow") -> None:
        sphere = pv.Sphere(radius=self._auto_marker_radius(), center=(x, y, z))
        actor = self.plotter.add_mesh(sphere, color=color, render=False)
        self._source_actors.append(actor)
        self.plotter.render()

    def clear_source_markers(self) -> None:
        for actor in self._source_actors:
            self.plotter.remove_actor(actor, render=False)
        self._source_actors = []
        self.plotter.render()

    def reset_camera(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()

    # -------------------------------------------------------------------------
    def _auto_marker_radius(self) -> float:
        if self._nodes is None or self._nodes.shape[0] == 0:
            return 0.005
        bbox = self._nodes.max(axis=0) - self._nodes.min(axis=0)
        return float(0.015 * np.linalg.norm(bbox))

    def _refresh_view(self) -> None:
        # Снимаем все основные акторы (но не маркеры источников).
        actor_keys = [k for k in list(self.plotter.actors.keys())
                      if k.startswith("__main__")]
        for k in actor_keys:
            self.plotter.remove_actor(k, render=False)
        # Не сбрасываем _source_actors — они отдельные.

        if self._volume_grid is None:
            self.plotter.render()
            return

        has_T = self._T is not None
        cmap = "inferno"

        if self._render_mode == "wireframe":
            self.plotter.add_mesh(self._surface_mesh, style="wireframe",
                                  color="#dcdee2", line_width=1.0,
                                  name="__main__surface")
        elif self._render_mode == "surface":
            scalars = "T" if has_T else None
            self.plotter.add_mesh(
                self._surface_mesh, scalars=scalars, cmap=cmap,
                show_edges=False, name="__main__surface",
                opacity=1.0,
                scalar_bar_args={"title": "T, °C"} if has_T else None,
            )
        elif self._render_mode == "volume":
            if has_T:
                # Volume rendering показывает поле внутри объёма.
                self.plotter.add_volume(
                    self._volume_grid, scalars="T", cmap=cmap,
                    name="__main__volume",
                    scalar_bar_args={"title": "T, °C"},
                    opacity="sigmoid",
                )
            else:
                # Без температуры — просто полупрозрачная поверхность.
                self.plotter.add_mesh(
                    self._surface_mesh, color="#7a6cf0", opacity=0.35,
                    name="__main__surface")
        elif self._render_mode == "isosurface":
            if has_T:
                Tmin, Tmax = float(self._T.min()), float(self._T.max())
                if Tmax - Tmin > 1e-12:
                    levels = np.linspace(Tmin, Tmax, self._iso_count)
                    iso = self._volume_grid.contour(
                        isosurfaces=levels.tolist(), scalars="T")
                    self.plotter.add_mesh(
                        iso, scalars="T", cmap=cmap, opacity=0.85,
                        name="__main__iso",
                        scalar_bar_args={"title": "T, °C"})
                # Полупрозрачная оболочка для контекста.
                self.plotter.add_mesh(
                    self._surface_mesh, color="#5a606b", opacity=0.15,
                    name="__main__surface")
            else:
                self.plotter.add_mesh(
                    self._surface_mesh, color="#7a6cf0", opacity=0.35,
                    name="__main__surface")

        # Сечение поверх всего.
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
                sliced = self._volume_grid.slice(normal=normal, origin=origin)
                self.plotter.add_mesh(
                    sliced, scalars="T", cmap=cmap,
                    name="__main__slice",
                    scalar_bar_args={"title": "T, °C"})
            except Exception:
                pass  # сечение может быть пустым в крайних случаях

        self.plotter.render()


# =============================================================================
# Реализация на matplotlib — резерв, всегда доступен.
# =============================================================================

class MatplotlibView(Visualization3D):
    """Резервный 3D-вид на matplotlib.

    Возможности (более скромные, чем у PyVista, но достаточные):
      - Каркас тетраэдральной сетки или поверхностная сетка;
      - Цвет по температуре на узлах поверхности или сечении;
      - Сечение по любой оси с ползунком;
      - Picking: клик мышью по сфере узлов;
      - Стандартное управление matplotlib (ЛКМ — вращение, ПКМ — масштаб).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
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

        self._nodes: Optional[np.ndarray] = None
        self._elements: Optional[np.ndarray] = None
        self._bnd_nodes: Optional[np.ndarray] = None
        self._T: Optional[np.ndarray] = None
        self._render_mode = "surface"
        self._slice_axis: Optional[str] = None
        self._slice_position = 0.5
        self._iso_count = 7
        self._pick_mode = "none"
        self._source_markers = []  # список (x, y, z, color)
        self._cbar = None

        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    @property
    def backend_name(self) -> str:
        return "matplotlib"

    def _setup_axes(self) -> None:
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
        self._refresh()
        self.reset_camera()

    def set_temperature(self, T):
        self._T = np.asarray(T, dtype=np.float64) if T is not None else None
        self._refresh()

    def set_render_mode(self, mode):
        self._render_mode = mode
        self._refresh()

    def set_slice(self, axis, position=0.5):
        self._slice_axis = axis
        self._slice_position = max(0.0, min(1.0, position))
        self._refresh()

    def set_isosurface_count(self, n):
        self._iso_count = max(2, min(20, int(n)))
        if self._render_mode == "isosurface":
            self._refresh()

    def set_pick_mode(self, mode):
        self._pick_mode = mode

    def add_source_marker(self, x, y, z, color="yellow"):
        self._source_markers.append((x, y, z, color))
        self._refresh()

    def clear_source_markers(self):
        self._source_markers = []
        self._refresh()

    def reset_camera(self):
        if self._nodes is not None and self._nodes.size > 0:
            mn = self._nodes.min(axis=0)
            mx = self._nodes.max(axis=0)
            self.ax.set_xlim(mn[0], mx[0])
            self.ax.set_ylim(mn[1], mx[1])
            self.ax.set_zlim(mn[2], mx[2])
        self.ax.view_init(elev=25, azim=-60)
        self.canvas.draw_idle()

    def _refresh(self):
        self.ax.cla()
        self._setup_axes()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        if self._nodes is None:
            self.canvas.draw_idle()
            return

        bbox_min = self._nodes.min(axis=0)
        bbox_max = self._nodes.max(axis=0)
        self.ax.set_xlim(bbox_min[0], bbox_max[0])
        self.ax.set_ylim(bbox_min[1], bbox_max[1])
        self.ax.set_zlim(bbox_min[2], bbox_max[2])

        has_T = self._T is not None

        # Базовый рендер: либо поверхность с тенью, либо точки на поверхности с цветом.
        bnd = self._bnd_nodes if (self._bnd_nodes is not None
                                  and self._bnd_nodes.size > 0) else None

        if self._render_mode == "wireframe" and bnd is not None:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            tri = self._nodes[bnd]
            poly = Poly3DCollection(tri, edgecolor="#dcdee2", facecolor="none",
                                    linewidths=0.4, alpha=0.6)
            self.ax.add_collection3d(poly)

        elif self._render_mode == "surface":
            if has_T and bnd is not None:
                # Поверхность с цветом по T (берём среднее по треугольнику).
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                tri = self._nodes[bnd]
                T_face = self._T[bnd].mean(axis=1)
                norm = matplotlib.colors.Normalize(
                    vmin=float(self._T.min()), vmax=float(self._T.max()))
                cmap = matplotlib.cm.get_cmap("inferno")
                colors = cmap(norm(T_face))
                poly = Poly3DCollection(tri, facecolors=colors,
                                        edgecolor="none", alpha=0.95)
                self.ax.add_collection3d(poly)
                sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                self._cbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.7,
                                                label="T, °C")
            elif bnd is not None:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                tri = self._nodes[bnd]
                poly = Poly3DCollection(tri, facecolor="#7a6cf0",
                                        edgecolor="#3c4049", linewidths=0.3,
                                        alpha=0.55)
                self.ax.add_collection3d(poly)

        elif self._render_mode in ("isosurface", "volume"):
            # matplotlib плохо умеет volume rendering; рисуем подвыборку
            # внутренних точек цветом по T.
            if has_T:
                n = self._nodes.shape[0]
                # Берём не больше 6000 точек для скорости.
                if n > 6000:
                    sample = np.random.RandomState(42).choice(n, 6000, replace=False)
                else:
                    sample = np.arange(n)
                norm = matplotlib.colors.Normalize(
                    vmin=float(self._T.min()), vmax=float(self._T.max()))
                cmap = matplotlib.cm.get_cmap("inferno")
                self._scatter = self.ax.scatter(
                    self._nodes[sample, 0], self._nodes[sample, 1],
                    self._nodes[sample, 2],
                    c=self._T[sample], cmap=cmap, norm=norm, s=6, alpha=0.55)
                self._cbar = self.fig.colorbar(
                    self._scatter, ax=self.ax, shrink=0.7, label="T, °C")
                if bnd is not None:
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    tri = self._nodes[bnd]
                    poly = Poly3DCollection(tri, facecolor="#5a606b",
                                            edgecolor="none", alpha=0.10)
                    self.ax.add_collection3d(poly)

        # Сечение.
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
                                cmap="inferno", s=20)

        # Источники-маркеры.
        for (x, y, z, color) in self._source_markers:
            self.ax.scatter([x], [y], [z], c=color, s=80,
                            marker="o", edgecolors="black", linewidths=0.8)

        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        if event.inaxes is not self.ax:
            return
        if self._pick_mode == "none" or self._nodes is None:
            return
        # matplotlib не даёт прямых 3D-координат под курсором, поэтому
        # используем простой подход: проецируем все узлы в экран и берём
        # ближайший к (event.xdata, event.ydata). Не идеально, но рабочее.
        try:
            from mpl_toolkits.mplot3d import proj3d
            x2, y2, _ = proj3d.proj_transform(
                self._nodes[:, 0], self._nodes[:, 1], self._nodes[:, 2],
                self.ax.get_proj())
            # Преобразуем в координаты данных осей и сравниваем с event.xdata.
            # event.xdata — это координата в системе осей matplotlib.
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


# =============================================================================
# Фабрика.
# =============================================================================

def create_view(parent: Optional[QWidget] = None,
                prefer: str = "auto") -> Visualization3D:
    """Создаёт лучшую доступную 3D-визуализацию.

    prefer:
        'auto'        — PyVista, иначе matplotlib;
        'pyvista'     — только PyVista (исключение, если недоступен);
        'matplotlib'  — только matplotlib.
    """
    if prefer == "matplotlib":
        return MatplotlibView(parent)
    if prefer == "pyvista":
        if not HAS_PYVISTA:
            raise RuntimeError("PyVista не установлен; установите: pip install pyvista pyvistaqt")
        return PyVistaView(parent)
    # auto
    if HAS_PYVISTA:
        try:
            return PyVistaView(parent)
        except Exception:
            return MatplotlibView(parent)
    return MatplotlibView(parent)
