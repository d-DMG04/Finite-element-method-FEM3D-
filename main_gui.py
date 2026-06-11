# -*- coding: utf-8 -*-
"""
main_gui.py
===========

Главное приложение программного комплекса МКЭ для трёхмерной теплопроводности.

Версия 1.4 — компактный интерфейс без скроллбаров:

  * Все группы настроек (геометрия, материал, источники, ГУ) вынесены в
    компактные карточки с кнопкой "Изменить...", открывающей диалог
    редактирования. Никаких скроллбаров — всё помещается в одно окно.
  * Темы: тёмная / светлая / бежевая (для длительной работы).
  * Регионы материалов: разный λ и Q в разных частях детали.
  * Расширенный тулбар над 3D: режим рендера, сечение, проекция,
    каркас, X-Ray, скриншот, сброс камеры по осям, изоповерхности.
  * Picking узлов и размещение источников кликом мыши.
  * Hover-подсказка температуры в точке курсора.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, Optional

try:
    from PyQt5.QtCore import Qt, QSettings, QThread, QTimer, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QPalette
    from PyQt5.QtWidgets import (QAction, QActionGroup, QApplication, QCheckBox,
                                 QComboBox, QDialog, QDoubleSpinBox,
                                 QFileDialog, QFrame, QGridLayout, QGroupBox,
                                 QHBoxLayout, QLabel, QListWidget,
                                 QListWidgetItem, QMainWindow, QMenuBar,
                                 QMessageBox, QProgressBar, QPushButton,
                                 QSizePolicy, QSlider, QSpinBox, QSplitter,
                                 QStatusBar, QTabWidget, QToolBar,
                                 QToolButton, QVBoxLayout, QWidget)
except ImportError as exc:
    sys.stderr.write(f"\nОшибка импорта PyQt5: {exc}\nУстановите: pip install PyQt5\n")
    sys.exit(1)

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fem3d import (BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN,
                   HEATING_TEMPLATES, MATERIALS, PRESETS, PROJECT_EXTENSION,
                   SHAPE_PRESETS,
                   BoundaryCondition, BoxGeometry, BoxPreset,
                   CoreBridge, CoreError, FACE_NAMES, FACE_X_MINUS, FACE_X_PLUS,
                   FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS,
                   Material, MaterialRegion, PointSource, Problem,
                   REGION_BOX, REGION_SPHERE, ShapePreset, SolverInfo,
                   SUPPORTED_IMPORT_EXTENSIONS, VolumeSource,
                   VOLSRC_BOX, VOLSRC_SPHERE, compute_mesh_info,
                   import_mesh_file, load_project, save_project,
                   template_reset)
from fem3d.postprocess import (compute_temperature_profile,
                                 export_csv, export_pdf_report,
                                 export_report, export_vtu)

from gui import (AppSettings, BoundaryConditionsDialog, CalculationRecord,
                 CalculationsView, GeometryDialog, HAS_PYVISTA, HelpDialog,
                 MaterialDialog, MaterialRegionsDialog,
                 PlotsView, PointSourceDialog, SettingsDialog, SolverWorker,
                 TransientParamsDialog, WhatIfView,
                 ForcedConvectionDialog,
                 VolumeSourceDialog, build_palette, build_stylesheet,
                 create_view, current_theme, set_theme)


# =============================================================================
# Компактная карточка свойства (название + краткое описание + кнопка).
# Заменяет длинные группы настроек на левой/правой панели.
# =============================================================================

class PropertyCard(QFrame):
    """Карточка с заголовком, кратким описанием и кнопкой «Изменить»."""

    def __init__(self, title: str, on_edit, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(2)

        head = QHBoxLayout()
        self.title_label = QLabel(f"<b>{title}</b>")
        head.addWidget(self.title_label, 1)
        self.btn_edit = QPushButton("Изменить...")
        self.btn_edit.setMaximumWidth(110)
        self.btn_edit.clicked.connect(on_edit)
        head.addWidget(self.btn_edit, 0)
        outer.addLayout(head)

        self.summary_label = QLabel("—")
        self.summary_label.setStyleSheet(
            f"color: {current_theme().text_dim}; font-size: 9pt;")
        self.summary_label.setWordWrap(True)
        outer.addWidget(self.summary_label)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(text)

    def apply_theme(self) -> None:
        self.summary_label.setStyleSheet(
            f"color: {current_theme().text_dim}; font-size: 9pt;")


# =============================================================================
# Карточка с двумя кнопками — для редактирования источников и регионов
# (открыть диалог + быстрая очистка).
# =============================================================================

class ListSummaryCard(QFrame):
    """Карточка-сводка по списку (источники, регионы и т.п.)."""

    def __init__(self, title: str, on_edit, on_clear,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(2)

        # Заголовок отдельной строкой — иначе в узкой панели длинный заголовок
        # («Регионы материалов») и две кнопки не помещаются и накладываются.
        self.title_label = QLabel(f"<b>{title}</b>")
        outer.addWidget(self.title_label)

        head = QHBoxLayout()
        head.setSpacing(4)
        head.addStretch(1)
        self.btn_edit = QPushButton("Изменить...")
        self.btn_edit.setMaximumWidth(110)
        self.btn_edit.clicked.connect(on_edit)
        head.addWidget(self.btn_edit)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.setMaximumWidth(80)
        self.btn_clear.clicked.connect(on_clear)
        head.addWidget(self.btn_clear)
        outer.addLayout(head)

        self.summary_label = QLabel("Пусто")
        self.summary_label.setStyleSheet(
            f"color: {current_theme().text_dim}; font-size: 9pt;")
        self.summary_label.setWordWrap(True)
        outer.addWidget(self.summary_label)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(text)

    def apply_theme(self) -> None:
        self.summary_label.setStyleSheet(
            f"color: {current_theme().text_dim}; font-size: 9pt;")


# =============================================================================
# Расширенная карточка источников: помимо "Изменить" есть кнопки
# "🖱 +Точка" и "🖱 +Сфера" для размещения источника кликом мыши.
# =============================================================================

class SourcesCard(QFrame):
    """Карточка локальных источников с быстрыми кнопками размещения мышью."""

    def __init__(self, on_edit, on_clear,
                 on_pick_point, on_pick_sphere,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6); outer.setSpacing(2)

        self.title_label = QLabel("<b>Локальные источники</b>")
        outer.addWidget(self.title_label)

        head = QHBoxLayout()
        head.setSpacing(4)
        head.addStretch(1)
        self.btn_edit = QPushButton("Список...")
        self.btn_edit.setMaximumWidth(95)
        self.btn_edit.clicked.connect(on_edit)
        head.addWidget(self.btn_edit)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.setMaximumWidth(80)
        self.btn_clear.clicked.connect(on_clear)
        head.addWidget(self.btn_clear)
        outer.addLayout(head)

        # Кнопки «разместить кликом» — основная фишка v1.5.
        pick_row = QHBoxLayout(); pick_row.setSpacing(4)
        self.btn_pick_point = QToolButton()
        self.btn_pick_point.setText("🖱 +Точка")
        self.btn_pick_point.setCheckable(True)
        self.btn_pick_point.setToolTip("Кликните в 3D-виде, чтобы поставить "
                                        "точечный источник в указанной точке")
        self.btn_pick_point.toggled.connect(on_pick_point)
        pick_row.addWidget(self.btn_pick_point)

        self.btn_pick_sphere = QToolButton()
        self.btn_pick_sphere.setText("🖱 +Сфера")
        self.btn_pick_sphere.setCheckable(True)
        self.btn_pick_sphere.setToolTip("2 клика в 3D: первый — центр сферы, "
                                         "второй — точка на границе (радиус)")
        self.btn_pick_sphere.toggled.connect(on_pick_sphere)
        pick_row.addWidget(self.btn_pick_sphere)
        pick_row.addStretch(1)
        outer.addLayout(pick_row)

        self.summary_label = QLabel("Пусто")
        self.summary_label.setStyleSheet(
            f"color: {current_theme().text_dim}; font-size: 9pt;")
        self.summary_label.setWordWrap(True)
        outer.addWidget(self.summary_label)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(text)

    def set_pick_point_active(self, active: bool) -> None:
        self.btn_pick_point.blockSignals(True)
        self.btn_pick_point.setChecked(active)
        self.btn_pick_point.blockSignals(False)

    def set_pick_sphere_active(self, active: bool) -> None:
        self.btn_pick_sphere.blockSignals(True)
        self.btn_pick_sphere.setChecked(active)
        self.btn_pick_sphere.blockSignals(False)

    def apply_theme(self) -> None:
        self.summary_label.setStyleSheet(
            f"color: {current_theme().text_dim}; font-size: 9pt;")


# =============================================================================
# Карточки граней в виде «чипов» — 2x3 сетка, не требует скролла.
# =============================================================================

class FaceChip(QPushButton):
    """Компактная кнопка одной грани — показывает имя, тип ГУ цветом."""

    def __init__(self, face_id: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.face_id = face_id
        self.setObjectName("Chip")
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.bc = BoundaryCondition()
        self._refresh()

    def set_bc(self, bc: BoundaryCondition) -> None:
        self.bc = bc
        self._refresh()

    def _refresh(self) -> None:
        from gui.theme import bc_colors
        color = bc_colors().get(self.bc.type, "#3c4049")
        # Текст на двух строках: имя + краткое описание (без обрезки).
        self.setText(f"{FACE_NAMES[self.face_id]}\n{self.bc.short_description()}")
        self.setToolTip(self.bc.description())
        self.setStyleSheet(
            f"QPushButton#Chip {{"
            f"  background-color: {current_theme().input_bg};"
            f"  color: {current_theme().text};"
            f"  border-left: 4px solid {color};"
            f"  border-top: 1px solid {current_theme().border};"
            f"  border-right: 1px solid {current_theme().border};"
            f"  border-bottom: 1px solid {current_theme().border};"
            f"  padding: 4px 8px; text-align: left; font-size: 9pt;"
            f"}}"
            f"QPushButton#Chip:hover {{"
            f"  border-color: {current_theme().border_strong};"
            f"  border-left: 4px solid {color};"
            f"}}"
        )


# =============================================================================
# Главное окно.
# =============================================================================

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Программный комплекс МКЭ — расчёт теплопроводности")
        self.setMinimumSize(1280, 780)
        self.setAcceptDrops(True)

        # QSettings: персистентные настройки между запусками.
        # На Linux это ini-файл в ~/.config; на Windows — реестр.
        self._qsettings = QSettings("FemHeat3D", "App")

        self.problem = Problem()
        self.settings = AppSettings()
        self._thread: Optional[QThread] = None
        self._worker: Optional[SolverWorker] = None

        # Состояние «разместить сферу 2 кликами»:
        #   None = режим не активен;
        #   "awaiting_center" = ждём первый клик;
        #   "awaiting_radius" = ждём второй клик (центр уже выбран).
        self._sphere_pick_state: Optional[str] = None
        self._sphere_center: Optional[tuple] = None

        # Состояние «измерить профиль T(x) по двум кликам»:
        #   None        — выключен;
        #   "awaiting_a" — ждём первый клик (точка A);
        #   "awaiting_b" — ждём второй клик (точка B);
        # После второго клика открывается окно с графиком профиля.
        self._line_pick_state: Optional[str] = None
        self._line_point_a: Optional[tuple] = None

        # Путь к текущему .fem3d-проекту (для Ctrl+S).
        self._current_project_path: Optional[str] = None

        # Recent files: список последних открытых проектов (макс. 10).
        self._recent_files: list = []

        # Загружаем сохранённые настройки до построения UI — чтобы тема
        # применилась к виджетам с первого кадра.
        self._load_qsettings()

        self._build_menus()
        self._build_ui()
        self._sync_summaries()
        self._refresh_view_from_problem(rebuild_geometry=True)
        self.apply_theme_to_widgets()
        self._refresh_recent_files_menu()
        # Применяем сохранённый режим рендера и colormap к виджетам.
        idx = self.mode_combo.findData(self.settings.render_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        saved_cmap = self._qsettings.value("viz/colormap", "inferno", type=str)
        cmap_idx = self.cmap_combo.findData(saved_cmap)
        if cmap_idx >= 0:
            self.cmap_combo.setCurrentIndex(cmap_idx)
            self.viz.set_colormap(saved_cmap)

        if not HAS_PYVISTA:
            self.statusBar().showMessage(
                "Используется matplotlib 3D-вид. Для лучшей визуализации: "
                "pip install pyvista pyvistaqt", 9000)

    # =========================================================================
    # QSettings: загрузка / сохранение.
    # =========================================================================

    def _load_qsettings(self) -> None:
        s = self._qsettings
        # Тема.
        theme_name = s.value("ui/theme", "dark", type=str)
        if theme_name in ("dark", "light", "sepia"):
            self.settings.theme = theme_name
            set_theme(theme_name)
        # Размер и позиция окна.
        geom = s.value("ui/geometry")
        if geom is not None:
            try:
                self.restoreGeometry(geom)
            except Exception:
                pass
        # Состояние сплиттера — восстанавливается после _build_ui.
        # Параметры решателя.
        tol = s.value("solver/tolerance", None)
        if tol is not None:
            try: self.settings.cg_tolerance = float(tol)
            except (TypeError, ValueError): pass
        mxi = s.value("solver/max_iter", None)
        if mxi is not None:
            try: self.settings.cg_max_iter = int(mxi)
            except (TypeError, ValueError): pass
        omp = s.value("solver/omp_threads", None)
        if omp is not None:
            try: self.settings.omp_threads = int(omp)
            except (TypeError, ValueError): pass
        # Режим визуализации по умолчанию.
        mode = s.value("viz/render_mode", None, type=str)
        if mode in ("surface", "volume", "isosurface", "wireframe"):
            self.settings.render_mode = mode
        # Color map.
        cmap = s.value("viz/colormap", None, type=str)
        if cmap:
            self.settings.colormap = cmap if hasattr(self.settings, "colormap") else cmap
        # Recent files.
        recent = s.value("recent_files", [])
        if isinstance(recent, list):
            self._recent_files = [str(p) for p in recent if p]
        elif isinstance(recent, str):  # QSettings иногда возвращает строку
            self._recent_files = [recent] if recent else []

    def _save_qsettings(self) -> None:
        s = self._qsettings
        s.setValue("ui/theme", self.settings.theme)
        s.setValue("ui/geometry", self.saveGeometry())
        s.setValue("solver/tolerance", float(self.settings.cg_tolerance))
        s.setValue("solver/max_iter", int(self.settings.cg_max_iter))
        s.setValue("solver/omp_threads", int(self.settings.omp_threads))
        s.setValue("viz/render_mode", self.settings.render_mode)
        if hasattr(self, "cmap_combo"):
            s.setValue("viz/colormap", self.cmap_combo.currentData())
        s.setValue("recent_files", self._recent_files[:10])
        s.sync()

    def closeEvent(self, event) -> None:
        """Перед закрытием — сохраняем настройки."""
        try:
            self._save_qsettings()
        except Exception:
            pass
        super().closeEvent(event)

    # =========================================================================
    # Recent files.
    # =========================================================================

    def _add_recent_file(self, path: str) -> None:
        """Добавляет файл в начало списка недавних, убирая дубли."""
        path = os.path.abspath(path)
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[:10]
        self._refresh_recent_files_menu()

    def _refresh_recent_files_menu(self) -> None:
        """Перестраивает подменю «Файл → Недавние»."""
        if not hasattr(self, "_recent_menu") or self._recent_menu is None:
            return
        self._recent_menu.clear()
        if not self._recent_files:
            empty = QAction("(пусто)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for i, path in enumerate(self._recent_files):
            name = os.path.basename(path)
            act = QAction(f"&{i+1}. {name}", self)
            act.setStatusTip(path)
            act.triggered.connect(
                lambda chk=False, p=path: self._load_project_from_path(p))
            self._recent_menu.addAction(act)
        self._recent_menu.addSeparator()
        act_clear = QAction("Очистить список", self)
        act_clear.triggered.connect(self._on_clear_recent)
        self._recent_menu.addAction(act_clear)

    def _on_clear_recent(self) -> None:
        self._recent_files = []
        self._refresh_recent_files_menu()

    # =========================================================================
    # Меню.
    # =========================================================================

    def _build_menus(self) -> None:
        m = self.menuBar()

        # === Файл ============================================================
        file_menu = m.addMenu("&Файл")

        a = QAction("&Новый проект", self)
        a.setShortcut("Ctrl+N"); a.triggered.connect(self._on_new_project)
        file_menu.addAction(a)
        a = QAction("&Открыть проект...", self)
        a.setShortcut("Ctrl+O"); a.triggered.connect(self._on_open_project)
        file_menu.addAction(a)
        a = QAction("&Сохранить проект", self)
        a.setShortcut("Ctrl+S"); a.triggered.connect(self._on_save_project)
        file_menu.addAction(a)
        a = QAction("Сохранить проект &как...", self)
        a.setShortcut("Ctrl+Shift+S"); a.triggered.connect(self._on_save_project_as)
        file_menu.addAction(a)

        # Подменю «Недавние проекты».
        self._recent_menu = file_menu.addMenu("&Недавние проекты")
        file_menu.addSeparator()

        a = QAction("&Импортировать сетку...", self)
        a.setShortcut("Ctrl+I"); a.triggered.connect(self._on_import_mesh)
        file_menu.addAction(a)
        file_menu.addSeparator()
        a = QAction("Экспорт &VTU...", self); a.triggered.connect(self._export_vtu)
        file_menu.addAction(a)
        a = QAction("Экспорт &CSV...", self); a.triggered.connect(self._export_csv)
        file_menu.addAction(a)
        a = QAction("Сохранить &отчёт...", self); a.triggered.connect(self._export_report)
        file_menu.addAction(a)
        a = QAction("Сохранить отчёт в &PDF...", self); a.triggered.connect(self._export_pdf)
        file_menu.addAction(a)
        a = QAction("Сохранить &скриншот 3D...", self); a.triggered.connect(self._save_screenshot)
        file_menu.addAction(a)
        file_menu.addSeparator()
        a = QAction("В&ыход", self); a.setShortcut("Ctrl+Q"); a.triggered.connect(self.close)
        file_menu.addAction(a)

        # === Вид =============================================================
        view_menu = m.addMenu("&Вид")
        a = QAction("&Сбросить камеру", self)
        a.setShortcut("Home"); a.triggered.connect(lambda: self.viz.reset_camera())
        view_menu.addAction(a)
        view_menu.addSeparator()

        # Стандартные позиции камеры.
        for label, key, sc in [("Спереди (+X)", "+x", "Ctrl+1"),
                                 ("Сзади (-X)",  "-x", "Ctrl+2"),
                                 ("Справа (+Y)", "+y", "Ctrl+3"),
                                 ("Слева (-Y)",  "-y", "Ctrl+4"),
                                 ("Сверху (+Z)", "+z", "Ctrl+5"),
                                 ("Снизу (-Z)",  "-z", "Ctrl+6"),
                                 ("Изометрия",   "iso","Ctrl+0")]:
            ac = QAction(label, self)
            if sc: ac.setShortcut(sc)
            ac.triggered.connect(lambda chk=False, k=key: self.viz.reset_view_to(k))
            view_menu.addAction(ac)
        view_menu.addSeparator()

        self.act_pick_node = QAction("&Узнать T в точке", self)
        self.act_pick_node.setCheckable(True); self.act_pick_node.setShortcut("P")
        self.act_pick_node.triggered.connect(
            lambda c: self._set_pick_mode("pick_node" if c else "none"))
        view_menu.addAction(self.act_pick_node)

        self.act_place_source = QAction("&Поставить источник кликом", self)
        self.act_place_source.setCheckable(True); self.act_place_source.setShortcut("S")
        self.act_place_source.triggered.connect(
            lambda c: self._set_pick_mode("place_source" if c else "none"))
        view_menu.addAction(self.act_place_source)

        self.act_hover = QAction("Показать T под &курсором", self)
        self.act_hover.setCheckable(True)
        self.act_hover.triggered.connect(
            lambda c: self.viz.set_hover_enabled(c))
        view_menu.addAction(self.act_hover)

        # Профиль T(x) по двум кликам.
        self.act_pick_line = QAction("&Профиль T вдоль линии (2 клика)", self)
        self.act_pick_line.setCheckable(True); self.act_pick_line.setShortcut("L")
        self.act_pick_line.triggered.connect(self._on_pick_line_mode)
        view_menu.addAction(self.act_pick_line)

        view_menu.addSeparator()
        a = QAction("Добавить точку наблюдения (термопару)", self)
        a.triggered.connect(self._on_add_observation_point)
        view_menu.addAction(a)
        a = QAction("Очистить точки наблюдения", self)
        a.triggered.connect(self._on_clear_observation_points)
        view_menu.addAction(a)

        # === Источники =======================================================
        src_menu = m.addMenu("&Источники")
        a = QAction("Точечный...", self); a.triggered.connect(self._on_add_point_source)
        src_menu.addAction(a)
        a = QAction("Объёмный (сфера)...", self); a.triggered.connect(self._on_add_volume_source)
        src_menu.addAction(a)
        src_menu.addSeparator()
        a = QAction("Очистить все источники", self); a.triggered.connect(self._on_clear_sources)
        src_menu.addAction(a)

        # === Материал ========================================================
        mat_menu = m.addMenu("&Материал")
        a = QAction("Глобальный материал...", self); a.triggered.connect(self._on_edit_material)
        mat_menu.addAction(a)
        a = QAction("&Регионы материалов...", self); a.triggered.connect(self._on_edit_regions)
        mat_menu.addAction(a)

        # === Обдув / конвекция ==============================================
        conv_menu = m.addMenu("&Обдув")
        a = QAction("Конвекция при обтекании потоком...", self)
        a.triggered.connect(self._on_forced_convection)
        conv_menu.addAction(a)
        a = QAction("Площадь поверхности / число Нуссельта", self)
        a.triggered.connect(self._on_surface_and_nusselt)
        conv_menu.addAction(a)

        # === Настройки =======================================================
        settings_menu = m.addMenu("&Настройки")
        a = QAction("&Параметры...", self); a.setShortcut("Ctrl+,")
        a.triggered.connect(self._on_open_settings)
        settings_menu.addAction(a)

        # Быстрый выбор темы в подменю.
        theme_menu = settings_menu.addMenu("&Тема")
        self._theme_group = QActionGroup(self)
        from gui.theme import THEMES
        for key, palette in THEMES.items():
            ta = QAction(palette.title, self, checkable=True)
            ta.setData(key)
            ta.setChecked(key == self.settings.theme)
            ta.triggered.connect(lambda chk=False, k=key: self._switch_theme(k))
            self._theme_group.addAction(ta)
            theme_menu.addAction(ta)

        # === Справка =========================================================
        help_menu = m.addMenu("Спр&авка")
        a = QAction("&Содержание", self); a.setShortcut("F1")
        a.triggered.connect(self._on_show_help)
        help_menu.addAction(a)
        a = QAction("&Проверить точность на эталоне...", self)
        a.triggered.connect(self._on_calibration_test)
        help_menu.addAction(a)
        a = QAction("О программе...", self); a.triggered.connect(self._on_show_about)
        help_menu.addAction(a)

    # =========================================================================
    # Построение интерфейса.
    # =========================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 760, 320])
        root.addWidget(splitter, 1)
        root.addWidget(self._build_bottom_panel(), 0)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    # ---- Левая панель ------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        panel = QFrame(); panel.setObjectName("Panel")
        panel.setMinimumWidth(300); panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)

        title = QLabel("<b>Модель</b>"); title.setStyleSheet("font-size: 11pt;")
        layout.addWidget(title)

        # Вкладки группируют настройки модели, чтобы не перегружать панель.
        model_tabs = QTabWidget()
        model_tabs.setObjectName("ModelTabs")
        # Растягиваем вкладки на всю ширину и не обрезаем подписи
        # («Источники» переставала помещаться и обрезалась до «Источни…»).
        model_tabs.tabBar().setExpanding(True)
        model_tabs.setElideMode(Qt.ElideNone)
        model_tabs.setUsesScrollButtons(False)

        # --- Вкладка «Геометрия» ---
        geo_tab = QWidget(); geo_l = QVBoxLayout(geo_tab)
        geo_l.setContentsMargins(4, 8, 4, 4); geo_l.setSpacing(8)
        self.card_geometry = PropertyCard("Геометрия", self._on_edit_geometry)
        geo_l.addWidget(self.card_geometry)
        geo_l.addStretch(1)
        model_tabs.addTab(geo_tab, "Геометрия")

        # --- Вкладка «Материал» ---
        mat_tab = QWidget(); mat_l = QVBoxLayout(mat_tab)
        mat_l.setContentsMargins(4, 8, 4, 4); mat_l.setSpacing(8)
        self.card_material = PropertyCard("Материал", self._on_edit_material)
        mat_l.addWidget(self.card_material)
        self.card_regions = ListSummaryCard(
            "Регионы материалов", self._on_edit_regions, self._on_clear_regions)
        mat_l.addWidget(self.card_regions)
        mat_l.addStretch(1)
        model_tabs.addTab(mat_tab, "Материал")

        # --- Вкладка «Источники» ---
        src_tab = QWidget(); src_l = QVBoxLayout(src_tab)
        src_l.setContentsMargins(4, 8, 4, 4); src_l.setSpacing(8)
        self.card_sources = SourcesCard(
            on_edit=self._on_edit_sources,
            on_clear=self._on_clear_sources,
            on_pick_point=self._on_pick_point_mode,
            on_pick_sphere=self._on_pick_sphere_mode,
        )
        src_l.addWidget(self.card_sources)
        src_l.addStretch(1)
        model_tabs.addTab(src_tab, "Источники")

        layout.addWidget(model_tabs, 1)

        # Кнопка построения сетки — всегда видна под вкладками.
        self.gen_button = QPushButton("Построить сетку")
        self.gen_button.setObjectName("AccentButton")
        self.gen_button.setMinimumHeight(36)
        self.gen_button.clicked.connect(self._on_generate_mesh)
        layout.addWidget(self.gen_button)

        # Сводка сетки — всегда видна.
        info_card = QFrame(); info_card.setObjectName("Card")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(8, 6, 8, 6)
        info_layout.addWidget(QLabel("<b>Сетка</b>"))
        self.info_label = QLabel("Не построена.")
        self.info_label.setStyleSheet(
            f"color: {current_theme().text_dim}; font-size: 9pt;")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_card)

        return panel

    # ---- Центральная панель -----------------------------------------------
    def _build_center_panel(self) -> QWidget:
        panel = QFrame(); panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ------ Вкладка 3D-вид ---------------------------------------------
        tab_3d = QWidget()
        v3d = QVBoxLayout(tab_3d)
        v3d.setContentsMargins(6, 6, 6, 6); v3d.setSpacing(4)

        # ТУЛБАР: режим + проекция + опции.
        toolbar = QHBoxLayout(); toolbar.setSpacing(4)

        toolbar.addWidget(QLabel("Режим:"))
        self.mode_combo = QComboBox()
        for label, value in [("Поверхность", "surface"),
                              ("Объём", "volume"),
                              ("Изоповерхности", "isosurface"),
                              ("Каркас", "wireframe")]:
            self.mode_combo.addItem(label, value)
        self.mode_combo.currentIndexChanged.connect(
            lambda _i: self.viz.set_render_mode(self.mode_combo.currentData()))
        toolbar.addWidget(self.mode_combo)

        self.btn_xray = QToolButton(); self.btn_xray.setText("X-Ray")
        self.btn_xray.setCheckable(True)
        self.btn_xray.setToolTip("Полупрозрачный режим — видно внутреннюю структуру")
        self.btn_xray.toggled.connect(lambda c: self.viz.set_xray(c))
        toolbar.addWidget(self.btn_xray)

        self.btn_edges = QToolButton(); self.btn_edges.setText("Рёбра")
        self.btn_edges.setCheckable(True)
        self.btn_edges.setToolTip("Показать рёбра сетки")
        self.btn_edges.toggled.connect(lambda c: self.viz.set_show_edges(c))
        toolbar.addWidget(self.btn_edges)

        self.btn_ortho = QToolButton(); self.btn_ortho.setText("Орто")
        self.btn_ortho.setCheckable(True)
        self.btn_ortho.setToolTip("Ортогональная (параллельная) проекция")
        self.btn_ortho.toggled.connect(
            lambda c: self.viz.set_projection("parallel" if c else "perspective"))
        toolbar.addWidget(self.btn_ortho)

        self.btn_lock = QToolButton(); self.btn_lock.setText("🔒")
        self.btn_lock.setCheckable(True)
        self.btn_lock.setToolTip("Заморозить камеру")
        self.btn_lock.toggled.connect(lambda c: self.viz.set_camera_locked(c))
        toolbar.addWidget(self.btn_lock)

        # Слои визуализации сгруппированы в одну кнопку-меню, чтобы не
        # перегружать тулбар. Каждый пункт — переключаемый чекбокс.
        from PyQt5.QtWidgets import QMenu
        self.btn_layers = QToolButton()
        self.btn_layers.setText("Слои ▾")
        self.btn_layers.setPopupMode(QToolButton.InstantPopup)
        self.btn_layers.setToolTip("Дополнительные слои визуализации")
        layers_menu = QMenu(self.btn_layers)

        self.act_flux = layers_menu.addAction("⇒ Стрелки потока")
        self.act_flux.setCheckable(True)
        self.act_flux.toggled.connect(
            lambda c: self.viz.set_flux_arrows_visible(c))

        self.act_isolines = layers_menu.addAction("≣ Изолинии")
        self.act_isolines.setCheckable(True)
        self.act_isolines.toggled.connect(
            lambda c: self.viz.set_isolines_visible(c))

        self.act_minmax = layers_menu.addAction("▼▲ Подписи Min/Max")
        self.act_minmax.setCheckable(True)
        self.act_minmax.toggled.connect(
            lambda c: self.viz.set_minmax_labels_visible(c))

        self.act_bc_overlay = layers_menu.addAction("◧ Подсветка ГУ")
        self.act_bc_overlay.setCheckable(True)
        self.act_bc_overlay.setChecked(True)
        self.act_bc_overlay.toggled.connect(self._on_toggle_bc_overlay)

        self.btn_layers.setMenu(layers_menu)
        toolbar.addWidget(self.btn_layers)

        # Шкала: палитра + log в одной кнопке-меню.
        self.btn_log = QToolButton(); self.btn_log.setText("log")
        self.btn_log.setCheckable(True)
        self.btn_log.setToolTip("Логарифмическая цветовая шкала.\n"
                                 "Работает только при T_min > 0 — полезно "
                                 "для задач с разницей температур на порядки.")
        self.btn_log.toggled.connect(
            lambda c: self.viz.set_log_scale(c))
        toolbar.addWidget(self.btn_log)

        toolbar.addWidget(QLabel("Палитра:"))
        self.cmap_combo = QComboBox()
        for cmap_name in ("inferno", "viridis", "plasma", "magma",
                          "coolwarm", "jet", "turbo"):
            self.cmap_combo.addItem(cmap_name, cmap_name)
        self.cmap_combo.setMaximumWidth(110)
        self.cmap_combo.currentIndexChanged.connect(
            lambda _i: self.viz.set_colormap(self.cmap_combo.currentData()))
        toolbar.addWidget(self.cmap_combo)

        toolbar.addStretch(1)

        btn_reset = QPushButton("Сбросить")
        btn_reset.setToolTip("Home — сброс камеры")
        btn_reset.clicked.connect(lambda: self.viz.reset_camera())
        toolbar.addWidget(btn_reset)

        btn_screenshot = QPushButton("📷")
        btn_screenshot.setToolTip("Сохранить скриншот 3D-вида")
        btn_screenshot.setMaximumWidth(36)
        btn_screenshot.clicked.connect(self._save_screenshot)
        toolbar.addWidget(btn_screenshot)

        v3d.addLayout(toolbar)

        # Сам 3D-вид.
        self.viz = create_view(self)
        self.viz.node_picked.connect(self._on_node_picked)
        self.viz.point_clicked.connect(self._on_point_clicked)
        self.viz.hover_value.connect(self._on_hover)
        v3d.addWidget(self.viz, 1)

        # Нижняя строка: сечение и изоповерхности.
        ctrl = QHBoxLayout(); ctrl.setSpacing(6)
        ctrl.addWidget(QLabel("Сечение:"))
        self.slice_combo = QComboBox()
        self.slice_combo.addItem("выкл", None)
        for a in ("x", "y", "z"):
            self.slice_combo.addItem(a, a)
        self.slice_combo.currentIndexChanged.connect(self._on_slice_changed)
        ctrl.addWidget(self.slice_combo)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 100); self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        ctrl.addWidget(self.slice_slider, 1)

        ctrl.addWidget(QLabel("Изо:"))
        self.iso_spin = QSpinBox(); self.iso_spin.setRange(2, 20); self.iso_spin.setValue(7)
        self.iso_spin.valueChanged.connect(
            lambda v: self.viz.set_isosurface_count(int(v)))
        ctrl.addWidget(self.iso_spin)
        v3d.addLayout(ctrl)

        # Строка состояния под 3D.
        self.pick_status_label = QLabel("")
        self.pick_status_label.setStyleSheet(
            f"color: {current_theme().text_dim}; font-size: 9pt;")
        v3d.addWidget(self.pick_status_label)

        self.tabs.addTab(tab_3d, "3D-вид")

        # ------ Вкладка Графики -------------------------------------------
        self.plots_view = PlotsView()
        self.tabs.addTab(self.plots_view, "Графики")

        # ------ Вкладка Расчёты -------------------------------------------
        self.calc_view = CalculationsView()
        self.calc_view.selected.connect(self._on_calc_selected)
        self.tabs.addTab(self.calc_view, "Расчёты")

        # ------ Вкладка «Что будет, если…» --------------------------------
        self.whatif_view = WhatIfView()
        self.whatif_view.recompute_requested.connect(self._on_whatif_recompute)
        self.tabs.addTab(self.whatif_view, "Что если")

        return panel

    # ---- Правая панель: чипы граней -----------------------------------------
    def _build_right_panel(self) -> QWidget:
        panel = QFrame(); panel.setObjectName("Panel")
        panel.setMinimumWidth(300); panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)

        title = QLabel("<b>Граничные условия</b>")
        title.setStyleSheet("font-size: 11pt;")
        layout.addWidget(title)

        # Кнопка «Изменить все» открывает табличный диалог.
        btn_edit_all = QPushButton("Изменить все...")
        btn_edit_all.clicked.connect(self._on_edit_all_bcs)
        layout.addWidget(btn_edit_all)

        # Сетка 2×3 чипов граней.
        grid = QGridLayout()
        grid.setSpacing(6)
        order = [(FACE_Z_PLUS, 0, 0), (FACE_Z_MINUS, 0, 1),
                 (FACE_X_PLUS, 1, 0), (FACE_X_MINUS, 1, 1),
                 (FACE_Y_PLUS, 2, 0), (FACE_Y_MINUS, 2, 1)]
        self.face_chips: Dict[int, FaceChip] = {}
        for fid, r, c in order:
            chip = FaceChip(fid)
            chip.clicked.connect(lambda chk=False, f=fid: self._on_edit_single_face(f))
            self.face_chips[fid] = chip
            grid.addWidget(chip, r, c)
        layout.addLayout(grid)

        # Шаблоны нагрева — кнопка открывает галерею с миниатюрами.
        layout.addWidget(QLabel("<b>Шаблоны нагрева</b>"))
        self.btn_template_gallery = QPushButton("📋 Выбрать сценарий…")
        self.btn_template_gallery.setToolTip(
            "Открыть галерею готовых сценариев граничных условий "
            "с наглядными миниатюрами.")
        self.btn_template_gallery.clicked.connect(self._on_open_template_gallery)
        layout.addWidget(self.btn_template_gallery)

        layout.addStretch(1)
        return panel

    # ---- Нижняя панель -----------------------------------------------------
    def _build_bottom_panel(self) -> QWidget:
        panel = QFrame(); panel.setObjectName("Panel")
        panel.setFixedHeight(82)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8); layout.setSpacing(12)

        self.run_button = QPushButton("▶  Запустить расчёт")
        self.run_button.setObjectName("RunButton")
        self.run_button.clicked.connect(self._on_run)
        layout.addWidget(self.run_button)

        # Чекбокс «Нестационарный режим» — компактно рядом с кнопкой.
        self.transient_check = QCheckBox("τ Нестационарный")
        self.transient_check.setToolTip(
            "Нестационарная задача: расчёт серии снимков T(t) "
            "по неявной схеме Эйлера.\n"
            "Требует заданных ρ и c_p в материале.")
        self.transient_check.toggled.connect(self._on_toggle_transient)
        layout.addWidget(self.transient_check)

        self.cancel_button = QPushButton("⏹ Прервать")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._on_cancel_solve)
        layout.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(180)
        layout.addWidget(self.progress, 1)

        self.result_label = QLabel("Готов к расчёту")
        self.result_label.setAlignment(Qt.AlignCenter)
        # Моноширинный шрифт: числа (T, невязка, время) выровнены и не «прыгают».
        self.result_label.setStyleSheet(
            'font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", '
            '"Courier New", monospace; font-size: 9pt;')
        layout.addWidget(self.result_label, 2)

        self.btn_vtu = QPushButton("VTU"); self.btn_vtu.setEnabled(False)
        self.btn_vtu.clicked.connect(self._export_vtu)
        layout.addWidget(self.btn_vtu)
        self.btn_csv = QPushButton("CSV"); self.btn_csv.setEnabled(False)
        self.btn_csv.clicked.connect(self._export_csv)
        layout.addWidget(self.btn_csv)
        self.btn_report = QPushButton("Отчёт"); self.btn_report.setEnabled(False)
        self.btn_report.clicked.connect(self._export_report)
        layout.addWidget(self.btn_report)
        self.btn_treport = QPushButton("Отчёт τ")
        self.btn_treport.setEnabled(False)
        self.btn_treport.setToolTip(
            "Отдельный отчёт по нестационарному расчёту:\n"
            "параметры схемы, теплофизика (a, τ, Fo), таблица динамики\n"
            "Tmin/Tmean/Tmax(t), время выхода на стационар, точки\n"
            "наблюдения. Рядом сохраняется CSV с историей T(t).")
        self.btn_treport.clicked.connect(self._export_transient_report)
        layout.addWidget(self.btn_treport)
        return panel

    # =========================================================================
    # Синхронизация карточек.
    # =========================================================================

    def _sync_summaries(self) -> None:
        # Геометрия.
        g = self.problem.geometry
        if self.problem.has_external_mesh():
            self.card_geometry.set_summary(
                f"Импортированная сетка.<br>"
                f"Bbox: {g.Lx:.4g} × {g.Ly:.4g} × {g.Lz:.4g} м"
            )
        else:
            self.card_geometry.set_summary(
                f"Параллелепипед: {g.Lx:g} × {g.Ly:g} × {g.Lz:g} м<br>"
                f"Сетка: {g.nx} × {g.ny} × {g.nz}"
            )
        # Материал.
        name = getattr(self.problem, "material_name", "") or "Произвольный"
        if self.problem.is_anisotropic:
            lam_str = (f"λ = ({self.problem.lambda_x:g}, "
                       f"{self.problem.lambda_y:g}, {self.problem.lambda_z:g}) "
                       f"Вт/(м·К)")
        else:
            lam_str = f"λ = {self.problem.lambda_:g} Вт/(м·К)"
        extra = ""
        if self.problem.rho > 0 and self.problem.cp > 0:
            extra = (f"<br>ρ = {self.problem.rho:g} кг/м³, "
                     f"c_p = {self.problem.cp:g} Дж/(кг·К)")
        self.card_material.set_summary(
            f"<b>{name}</b><br>{lam_str}<br>Q = {self.problem.Q:g} Вт/м³{extra}"
        )
        # Регионы.
        if not self.problem.material_regions:
            self.card_regions.set_summary("Не заданы (однородный материал)")
        else:
            lines = []
            for i, r in enumerate(self.problem.material_regions[:3]):
                lines.append(f"• {r.name} (λ={r.lambda_:g})")
            if len(self.problem.material_regions) > 3:
                lines.append(f"... и ещё {len(self.problem.material_regions) - 3}")
            self.card_regions.set_summary("<br>".join(lines))
        # Источники.
        n_pt = len(self.problem.point_sources)
        n_vs = len(self.problem.volume_sources)
        if n_pt == 0 and n_vs == 0:
            self.card_sources.set_summary("Нет")
        else:
            self.card_sources.set_summary(
                f"Точечные: {n_pt}<br>Объёмные: {n_vs}")
        # Чипы граней.
        for fid, chip in self.face_chips.items():
            chip.set_bc(self.problem.bcs.get(fid, BoundaryCondition()))

    # =========================================================================
    # Слоты: геометрия / материал / регионы / источники / ГУ.
    # =========================================================================

    def _on_edit_geometry(self) -> None:
        dlg = GeometryDialog(self.problem, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        kind = dlg.result_kind()
        if kind == "box":
            Lx, Ly, Lz, nx, ny, nz = dlg.box_params()
            self.problem.external_nodes = None
            self.problem.external_elements = None
            self.problem.external_bnd_nodes = None
            self.problem.external_bnd_face_ids = None
            self.problem.geometry = BoxGeometry(Lx=Lx, Ly=Ly, Lz=Lz,
                                                 nx=nx, ny=ny, nz=nz)
            self._sync_summaries()
        elif kind == "shape":
            sps = dlg.shape_preset()
            if sps is None:
                return
            density = dlg.shape_density() if hasattr(dlg, "shape_density") else 1.0
            try:
                self.statusBar().showMessage(
                    f"Построение сетки фигуры (плотность ×{density:g})…")
                QApplication.processEvents()
                nodes, tets, bnd_nodes, bnd_face_ids = sps.build(density=density)
                self.statusBar().clearMessage()
            except Exception as exc:
                self.statusBar().clearMessage()
                QMessageBox.critical(self, "Ошибка", f"Не удалось построить:\n{exc}")
                return
            self._apply_external_mesh(nodes, tets, bnd_nodes, bnd_face_ids)
        elif kind == "import":
            path = dlg.import_path()
            if not path:
                return
            try:
                nodes, tets, bnd_nodes, bnd_face_ids = import_mesh_file(path)
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка импорта", str(exc))
                return
            self._apply_external_mesh(nodes, tets, bnd_nodes, bnd_face_ids)

    def _on_import_mesh(self) -> None:
        exts = " ".join("*" + e for e in SUPPORTED_IMPORT_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл сетки", "",
            f"Сетки ({exts});;Все файлы (*.*)"
        )
        if not path:
            return
        try:
            nodes, tets, bnd_nodes, bnd_face_ids = import_mesh_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка импорта", str(exc))
            return
        self._apply_external_mesh(nodes, tets, bnd_nodes, bnd_face_ids)

    def _apply_external_mesh(self, nodes, tets, bnd_nodes, bnd_face_ids) -> None:
        self.problem.external_nodes = nodes
        self.problem.external_elements = tets
        self.problem.external_bnd_nodes = bnd_nodes
        self.problem.external_bnd_face_ids = bnd_face_ids
        bbox_min = nodes.min(axis=0); bbox_max = nodes.max(axis=0)
        self.problem.geometry = BoxGeometry(
            Lx=float(bbox_max[0] - bbox_min[0]),
            Ly=float(bbox_max[1] - bbox_min[1]),
            Lz=float(bbox_max[2] - bbox_min[2]),
            nx=1, ny=1, nz=1,
        )
        self._sync_summaries()

    def _on_edit_material(self) -> None:
        dlg = MaterialDialog(self.problem, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        # Сохраняем ВСЕ свойства материала, а не только λ и Q.
        mat = dlg.selected_material()
        self.problem.lambda_ = mat.effective_lambda()
        self.problem.Q = float(dlg.q_spin.value())
        self.problem.rho = float(mat.rho)
        self.problem.cp = float(mat.cp)
        # Анизотропия.
        self.problem.is_anisotropic = bool(mat.is_anisotropic)
        if mat.is_anisotropic:
            self.problem.lambda_x = mat.lambda_x
            self.problem.lambda_y = mat.lambda_y
            self.problem.lambda_z = mat.lambda_z
        # Запоминаем имя материала для отображения.
        self.problem.material_name = mat.name
        self._sync_summaries()
        self.statusBar().showMessage(
            f"Материал: {mat.name} (λ={self.problem.lambda_:g} Вт/(м·К), "
            f"ρ={mat.rho:g}, c_p={mat.cp:g})", 5000)

    def _on_edit_regions(self) -> None:
        dlg = MaterialRegionsDialog(
            self.problem.material_regions, self.problem.geometry, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        self.problem.material_regions = dlg.regions()
        self._sync_summaries()
        # Подсветить регионы в 3D.
        self.viz.clear_region_markers()
        for r in self.problem.material_regions:
            self.viz.add_region_marker(r)

    def _on_clear_regions(self) -> None:
        self.problem.material_regions = []
        self.viz.clear_region_markers()
        self._sync_summaries()

    def _on_edit_sources(self) -> None:
        """Список источников редактируется через QListWidget в диалоге."""
        # Простой диалог — список + кнопки добавить точечный/объёмный/удалить.
        from PyQt5.QtWidgets import QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Локальные источники"); dlg.resize(540, 380)
        outer = QVBoxLayout(dlg)
        info = QLabel("Точечные источники привязаны к узлам сетки. "
                       "Объёмные задаются как сферические области.")
        info.setStyleSheet(f"color: {current_theme().text_dim};")
        info.setWordWrap(True); outer.addWidget(info)

        lst = QListWidget()
        outer.addWidget(lst, 1)

        def refresh():
            lst.clear()
            for ps in self.problem.point_sources:
                it = QListWidgetItem(f"⊙  {ps.description()}")
                it.setData(Qt.UserRole, ("point", ps)); lst.addItem(it)
            for vs in self.problem.volume_sources:
                it = QListWidgetItem(f"◯  {vs.description()}")
                it.setData(Qt.UserRole, ("volume", vs)); lst.addItem(it)
        refresh()

        btn_row = QHBoxLayout()
        b_add_pt = QPushButton("+ Точечный")
        b_add_vol = QPushButton("+ Сферический")
        b_remove = QPushButton("Удалить")
        btn_row.addWidget(b_add_pt); btn_row.addWidget(b_add_vol)
        btn_row.addWidget(b_remove); btn_row.addStretch(1)
        outer.addLayout(btn_row)

        def on_add_pt():
            if self.problem.nodes is None:
                QMessageBox.information(dlg, "Нет сетки",
                                        "Сначала постройте сетку.")
                return
            d = PointSourceDialog(self.problem.geometry, parent=dlg)
            if d.exec_() == QDialog.Accepted:
                x, y, z, P = d.values()
                diff = self.problem.nodes - np.array([x, y, z])
                idx = int(np.argmin(np.sum(diff * diff, axis=1)))
                self.problem.point_sources.append(
                    PointSource(node_idx=idx, power=P))
                refresh()

        def on_add_vol():
            d = VolumeSourceDialog(self.problem.geometry, parent=dlg)
            if d.exec_() == QDialog.Accepted:
                cx, cy, cz, r, Q0 = d.values()
                self.problem.volume_sources.append(VolumeSource(
                    shape=VOLSRC_SPHERE, params=(cx, cy, cz, r), Q0=Q0))
                refresh()

        def on_remove():
            it = lst.currentItem()
            if not it: return
            kind, src = it.data(Qt.UserRole)
            if kind == "point":
                self.problem.point_sources.remove(src)
            else:
                self.problem.volume_sources.remove(src)
            refresh()

        b_add_pt.clicked.connect(on_add_pt)
        b_add_vol.clicked.connect(on_add_vol)
        b_remove.clicked.connect(on_remove)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept)
        bb.button(QDialogButtonBox.Close).clicked.connect(dlg.accept)
        outer.addWidget(bb)
        dlg.exec_()
        self._sync_summaries()
        self._refresh_source_markers()

    def _on_add_point_source(self) -> None:
        if self.problem.nodes is None:
            QMessageBox.information(self, "Нет сетки", "Сначала постройте сетку.")
            return
        d = PointSourceDialog(self.problem.geometry, parent=self)
        if d.exec_() == QDialog.Accepted:
            x, y, z, P = d.values()
            diff = self.problem.nodes - np.array([x, y, z])
            idx = int(np.argmin(np.sum(diff * diff, axis=1)))
            self.problem.point_sources.append(PointSource(node_idx=idx, power=P))
            self._sync_summaries(); self._refresh_source_markers()

    def _on_add_volume_source(self) -> None:
        d = VolumeSourceDialog(self.problem.geometry, parent=self)
        if d.exec_() == QDialog.Accepted:
            cx, cy, cz, r, Q0 = d.values()
            self.problem.volume_sources.append(VolumeSource(
                shape=VOLSRC_SPHERE, params=(cx, cy, cz, r), Q0=Q0))
            self._sync_summaries(); self._refresh_source_markers()

    def _on_clear_sources(self) -> None:
        self.problem.point_sources = []
        self.problem.volume_sources = []
        self._sync_summaries(); self._refresh_source_markers()

    def _refresh_source_markers(self) -> None:
        self.viz.clear_source_markers()
        if self.problem.nodes is None:
            return
        for ps in self.problem.point_sources:
            if 0 <= ps.node_idx < self.problem.nodes.shape[0]:
                nx, ny, nz = self.problem.nodes[ps.node_idx]
                self.viz.add_source_marker(float(nx), float(ny), float(nz),
                                            color="#ffd24a")
        for vs in self.problem.volume_sources:
            if vs.shape == VOLSRC_SPHERE:
                cx, cy, cz, _r = vs.params[:4]
                self.viz.add_source_marker(float(cx), float(cy), float(cz),
                                            color="#ff7b3a")

    def _on_edit_all_bcs(self, focus_face: int = 0) -> None:
        dlg = BoundaryConditionsDialog(self.problem, self,
                                        focus_face=focus_face)
        if dlg.exec_() == QDialog.Accepted:
            self.problem.bcs = dlg.result_bcs()
            # Сохраняем параметры обдува в задачу (для вывода и сохранения).
            flow = dlg.air_flow_result()
            self.problem.air_flow_enabled = flow["enabled"]
            self.problem.air_flow_speed = flow["speed"]
            self.problem.air_flow_direction = flow["direction"]
            self.problem.air_flow_T_inf = flow["T_inf"]
            # Частичное погружение в жидкость.
            self.problem.immersion = dlg.immersion_result()
            self._sync_summaries()
            self._refresh_bc_overlay()

    def _on_edit_single_face(self, fid: int) -> None:
        # Тот же диалог, но фокус сразу на нужной грани.
        self._on_edit_all_bcs(focus_face=int(fid))

    def _on_open_template_gallery(self) -> None:
        from gui import TemplateGalleryDialog
        dlg = TemplateGalleryDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        factory = dlg.chosen_factory()
        if factory is None:
            return
        bcs = factory()
        for fid, bc in bcs.items():
            self.problem.bcs[fid] = bc
        self._sync_summaries()
        self._refresh_bc_overlay()
        self.statusBar().showMessage("Сценарий ГУ применён", 4000)

    # =========================================================================
    # Построение сетки и расчёт.
    # =========================================================================

    def _on_generate_mesh(self) -> None:
        # Старые результаты больше неактуальны — сетка меняется.
        self.problem.T = None
        self.problem.flux = None
        self.problem.info = None
        try:
            with CoreBridge() as bridge:
                self.problem.build_mesh_in_core(bridge)
                n_elems = bridge.n_elements
                n_faces = bridge.n_boundary_faces
        except CoreError as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить сетку:\n{exc}")
            return

        info = compute_mesh_info(self.problem.nodes, n_elems, n_faces)
        self.info_label.setText(
            f"Узлов: <b>{info.n_nodes}</b>; "
            f"элементов: <b>{info.n_elements}</b><br>"
            f"Габариты: {info.bbox_max[0]-info.bbox_min[0]:.4g} × "
            f"{info.bbox_max[1]-info.bbox_min[1]:.4g} × "
            f"{info.bbox_max[2]-info.bbox_min[2]:.4g} м<br>"
            f"Память ≈ {info.memory_mb:.2f} МБ"
        )
        self._refresh_view_from_problem(rebuild_geometry=True)

    def _refresh_view_from_problem(self, rebuild_geometry: bool) -> None:
        if self.problem.nodes is None or self.problem.elements is None:
            return
        if rebuild_geometry:
            bnd = (self.problem.external_bnd_nodes
                   if (self.problem.has_external_mesh()
                       and self.problem.external_bnd_nodes is not None)
                   else self._compute_box_boundary_triangles())
            self.viz.set_mesh(self.problem.nodes, self.problem.elements, bnd)
            self._refresh_source_markers()
            # Перерисовать регионы.
            self.viz.clear_region_markers()
            for r in self.problem.material_regions:
                self.viz.add_region_marker(r)
        # Обновим подсветку граней по типу ГУ.
        self._refresh_bc_overlay()
        self.viz.set_temperature(self.problem.T)

    def _refresh_bc_overlay(self) -> None:
        """Передать в viz цвета граней, основанные на текущих ГУ."""
        if hasattr(self, "act_bc_overlay") and not self.act_bc_overlay.isChecked():
            self.viz.set_bc_overlay(None, None, None)
            return
        from gui.theme import bc_colors
        if self.problem.nodes is None or self.problem.elements is None:
            return
        # Поверхностные треугольники + их face_id.
        if (self.problem.has_external_mesh()
                and self.problem.external_bnd_face_ids is not None
                and self.problem.external_bnd_nodes is not None):
            bnd_faces = self.problem.external_bnd_nodes
            face_ids = self.problem.external_bnd_face_ids
        else:
            # Box: воспроизведём face_id для каждой треугольной грани.
            bnd_faces = self._compute_box_boundary_triangles()
            face_ids = self._compute_box_face_ids(bnd_faces)
        # Цвет каждой грани = цвет её ГУ.
        color_map_by_bc = bc_colors()
        face_id_to_color = {}
        for fid in range(6):
            bc = self.problem.bcs.get(fid)
            if bc is None:
                continue
            face_id_to_color[fid] = color_map_by_bc.get(bc.type, "#3c4049")
        self.viz.set_bc_overlay(bnd_faces, face_ids, face_id_to_color)

    def _on_toggle_bc_overlay(self, enabled: bool) -> None:
        """Включить/выключить подсветку граней."""
        if enabled:
            self._refresh_bc_overlay()
        else:
            self.viz.set_bc_overlay(None, None, None)

    def _compute_box_face_ids(self, bnd_faces: np.ndarray) -> np.ndarray:
        if bnd_faces.shape[0] == 0 or self.problem.nodes is None:
            return np.empty(0, dtype=np.int32)
        g = self.problem.geometry
        n0 = self.problem.nodes[bnd_faces[:, 0]]
        n1 = self.problem.nodes[bnd_faces[:, 1]]
        n2 = self.problem.nodes[bnd_faces[:, 2]]
        center = (n0 + n1 + n2) / 3.0
        eps = 1e-6 * max(g.Lx, g.Ly, g.Lz)
        face_ids = np.full(bnd_faces.shape[0], -1, dtype=np.int32)
        face_ids[np.abs(center[:, 0]) < eps] = 0           # X-
        face_ids[np.abs(center[:, 0] - g.Lx) < eps] = 1    # X+
        face_ids[np.abs(center[:, 1]) < eps] = 2           # Y-
        face_ids[np.abs(center[:, 1] - g.Ly) < eps] = 3    # Y+
        face_ids[np.abs(center[:, 2]) < eps] = 4           # Z-
        face_ids[np.abs(center[:, 2] - g.Lz) < eps] = 5    # Z+
        return face_ids

    def _compute_box_boundary_triangles(self) -> np.ndarray:
        from fem3d.mesh import _extract_surface_faces
        if self.problem.elements is None:
            return np.empty((0, 3), dtype=np.int32)
        return _extract_surface_faces(self.problem.elements).astype(np.int32)

    def _on_toggle_transient(self, enabled: bool) -> None:
        """Открыть диалог параметров нестационарной задачи."""
        if not enabled:
            return
        dlg = TransientParamsDialog(self, problem=self.problem)
        if dlg.exec_() != QDialog.Accepted:
            self.transient_check.setChecked(False)
            return
        p = dlg.params()
        self._transient_params = p
        # ρ и c_p из диалога становятся свойствами задачи (и попадут в отчёт).
        self.problem.rho = float(p["rho"])
        self.problem.cp = float(p["cp"])
        self.statusBar().showMessage(
            f"Нестационарный режим: t_end={p['t_end']:.1f} с, "
            f"Δt={p['dt']:.3f} с, T_init={p['T_init']:.1f} °C, "
            f"{p['n_save']} снимков, {p['fps']} кадр/с", 5000)

    def _on_run(self) -> None:
        types = {bc.type for bc in self.problem.bcs.values()}
        if BC_DIRICHLET not in types and BC_ROBIN not in types:
            QMessageBox.warning(self, "Неполные условия",
                "Хотя бы на одной грани должно быть задано Дирихле или Робен — "
                "иначе задача не определена однозначно.")
            return
        if self.problem.nodes is None:
            QMessageBox.information(self, "Нет сетки", "Сначала постройте сетку.")
            return

        # Развилка стационар/нестационар.
        if self.transient_check.isChecked():
            return self._run_transient()
        return self._run_steady()

    def _run_transient(self) -> None:
        """Запуск нестационарного расчёта (синхронно, без отдельного потока)."""
        if not hasattr(self, "_transient_params") or self._transient_params is None:
            QMessageBox.information(self, "Нет параметров",
                "Сначала задайте параметры нестационарного режима "
                "(снимите и снова поставьте галочку «τ Нестационарный»).")
            return
        p = self._transient_params
        # Требуется ρ и c_p.
        if self.problem.rho <= 0 or self.problem.cp <= 0:
            # Попытаемся взять из активного материала.
            mat = self._active_material()
            if mat is not None:
                self.problem.rho = float(mat.rho)
                self.problem.cp  = float(mat.cp)
        if self.problem.rho <= 0 or self.problem.cp <= 0:
            QMessageBox.warning(self, "Нет ρ и c_p",
                "Для нестационарного расчёта нужны плотность и теплоёмкость.\n"
                "Откройте «Материал → Изменить...» и задайте ρ и c_p, "
                "либо выберите материал из библиотеки.")
            return
        self.run_button.setEnabled(False)
        self.result_label.setText("Идёт нестационарный расчёт...")
        QApplication.processEvents()
        try:
            with CoreBridge() as br:
                self.problem.build_mesh_in_core(br)
                times, T_hist = self.problem.solve_transient(
                    br, t_end=p["t_end"], dt=p["dt"],
                    T_init=p["T_init"], n_save=p["n_save"],
                    tol=self.settings.cg_tolerance,
                    max_iter=self.settings.cg_max_iter)
            self._transient_times = times
            self._transient_T_history = T_hist
            # Фиксируем цветовую шкалу по ГЛОБАЛЬНОМУ диапазону за всё время —
            # иначе каждый кадр нормируется на собственные Tmin/Tmax и
            # рост/падение температуры визуально не виден.
            Tg_min, Tg_max = float(T_hist.min()), float(T_hist.max())
            if p.get("fixed_scale", True):
                self.viz.set_fixed_clim(Tg_min, Tg_max)
            else:
                self.viz.clear_fixed_clim()
            means = T_hist.mean(axis=1)
            dT = float(means[-1] - means[0])
            trend = ("нагрев" if dT > 1e-6 else
                     "остывание" if dT < -1e-6 else "стационар")
            self.result_label.setText(
                f"Готово: {len(times)} снимков за {times[-1]:g} с, "
                f"T={Tg_min:.2f}..{Tg_max:.2f} °C, "
                f"Tmean {means[0]:.2f}→{means[-1]:.2f} °C ({trend})")
            # Включаем плеер.
            self._build_or_show_transient_player()
            # ВАЖНО: показываем НАЧАЛЬНЫЙ кадр (t=0, нагретое состояние),
            # а не последний (остывшее ≈ среда). Иначе кажется, что тело
            # «не нагрето и не остывает» — на самом деле просто сразу
            # показывался конечный остывший кадр.
            self.viz.set_temperature(T_hist[0])
            # Автозапуск анимации остывания/прогрева — пользователь сразу
            # видит динамику, а не статичный кадр.
            self._autostart_transient_animation()
            # Активируем кнопки экспорта (включая отдельный отчёт τ).
            for btn_name in ("btn_vtu", "btn_csv", "btn_report",
                              "btn_treport"):
                if hasattr(self, btn_name):
                    getattr(self, btn_name).setEnabled(True)
            # Если заданы точки наблюдения — показать график T(t).
            if self.problem.observation_points:
                self._show_observation_plot(times, T_hist)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            self.result_label.setText(f"Ошибка: {exc}")
        finally:
            self.run_button.setEnabled(True)

    def _active_material(self):
        """Возвращает Material с rho/cp если активен материал из библиотеки."""
        return None

    # =========================================================================
    # Конвективный теплообмен при обтекании (обдув).
    # =========================================================================
    def _on_forced_convection(self) -> None:
        """Диалог обдува: расчёт Re/Nu/h и (опц.) назначение конвекции на грани."""
        if self.problem.nodes is None:
            QMessageBox.information(self, "Нет сетки",
                                    "Сначала постройте сетку.")
            return
        from fem3d import convection as cv
        # Подсказка по T поверхности: среднее из последнего расчёта, если есть.
        t_hint = (float(self.problem.T.mean())
                  if getattr(self.problem, "T", None) is not None
                  and self.problem.T.size else None)
        dlg = ForcedConvectionDialog(self, T_surface_hint=t_hint)
        if not dlg.exec_():
            return
        p = dlg.params()
        try:
            if p["apply"]:
                res = cv.apply_forced_convection_bc(
                    self.problem, speed=p["speed"], direction=p["direction"],
                    shape=p["shape"], T_inf=p["T_inf"],
                    T_surface=p["T_surface"], orient_weighting=p["orient"])
                # Сохраняем параметры обдува в саму задачу (для вывода/отчёта).
                self.problem.air_flow_enabled = True
                self.problem.air_flow_speed = p["speed"]
                self.problem.air_flow_direction = p["direction"]
                self.problem.air_flow_T_inf = p["T_inf"]
                self.problem.air_flow_shape = p["shape"]
                # Обновляем подсветку ГУ и карточки граней.
                self._refresh_view_from_problem(rebuild_geometry=False)
                if hasattr(self, "_sync_summaries"):
                    self._sync_summaries()
                applied = ("\n\nКонвекция (α = h) назначена на грани. "
                           "Запустите расчёт, чтобы увидеть поле T.")
            else:
                res = cv.analyze_forced_convection(
                    self.problem, speed=p["speed"], direction=p["direction"],
                    shape=p["shape"], T_inf=p["T_inf"],
                    T_surface=p["T_surface"])
                applied = ""
            QMessageBox.information(self, "Конвекция при обтекании",
                                    res.report_text() + applied)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _on_surface_and_nusselt(self) -> None:
        """Полный вывод величин конвекции: обдув, Re/Nu/h, площади, Q, Bi."""
        if self.problem.nodes is None:
            QMessageBox.information(self, "Нет сетки",
                                    "Сначала постройте сетку.")
            return
        from fem3d import convection as cv
        QMessageBox.information(self, "Конвекция: все величины",
                                cv.convection_summary_text(self.problem))

    def _on_add_observation_point(self) -> None:
        """Включить режим установки точки наблюдения кликом в 3D."""
        if self.problem.nodes is None:
            QMessageBox.information(self, "Нет сетки", "Сначала постройте сетку.")
            return
        # Сбрасываем другие pick-режимы и включаем сбор точки.
        self._obs_pick_active = True
        self.viz.set_pick_mode("pick_node")
        self.statusBar().showMessage(
            "Кликните в 3D-виде, чтобы поставить точку наблюдения "
            "(виртуальную термопару). Esc — отмена.", 0)

    def _on_clear_observation_points(self) -> None:
        self.problem.observation_points = []
        self.viz.clear_source_markers()  # переиспользуем маркеры
        self._refresh_observation_markers()
        self.statusBar().showMessage("Точки наблюдения удалены", 3000)

    def _refresh_observation_markers(self) -> None:
        """Показать точки наблюдения зелёными маркерами в 3D."""
        if hasattr(self.viz, "clear_obs_markers"):
            self.viz.clear_obs_markers()
        for i, (x, y, z) in enumerate(self.problem.observation_points):
            if hasattr(self.viz, "add_obs_marker"):
                self.viz.add_obs_marker(x, y, z, i + 1)

    def _show_observation_plot(self, times, T_hist) -> None:
        """Окно с графиком T(t) для всех точек наблюдения (как осциллограф)."""
        from fem3d.postprocess import sample_history_at_points
        pts = self.problem.observation_points
        if not pts:
            return
        series = sample_history_at_points(self.problem.nodes, T_hist, pts)
        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from PyQt5.QtWidgets import QDialog, QVBoxLayout
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"matplotlib: {exc}")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Точки наблюдения — T(t)")
        dlg.resize(820, 540)
        lay = QVBoxLayout(dlg)
        th = current_theme()
        fig = Figure(figsize=(8, 5), facecolor=th.panel)
        canvas = FigureCanvasQTAgg(fig); lay.addWidget(canvas)
        ax = fig.add_subplot(111); ax.set_facecolor(th.panel)
        colors = ["#7a6cf0", "#e8a24e", "#3aa55a", "#e85d4e",
                  "#4e8de8", "#e066b3", "#d4c84a", "#41b8b8"]
        for i in range(series.shape[0]):
            x, y, z = pts[i]
            ax.plot(times, series[i], "-", linewidth=2,
                    color=colors[i % len(colors)],
                    label=f"#{i+1} ({x:.3g}, {y:.3g}, {z:.3g})")
        ax.set_xlabel("Время, с", color=th.text)
        ax.set_ylabel("T, °C", color=th.text)
        ax.set_title("Температура в точках наблюдения", color=th.text, fontsize=11)
        ax.tick_params(colors=th.text)
        for s in ax.spines.values():
            s.set_color(th.text)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        canvas.draw()
        dlg.exec_()

    def _on_whatif_recompute(self, params: dict) -> None:
        """Быстрый пересчёт на грубой сетке для интерактивного исследования.

        Строит небольшой box (8×8×8), применяет сценарий и параметры из
        слайдеров, решает и показывает T_min/T_max/среднее. Поле также
        выводится в 3D-вид, если активна вкладка 3D.
        """
        from fem3d import (Problem, BoundaryCondition, BoxGeometry,
                            BC_DIRICHLET, BC_NEUMANN, BC_ROBIN,
                            FACE_X_MINUS)
        lam = max(params["lambda_"], 1e-3)
        alpha = params["alpha"]
        Q = params["Q"]
        t_inf = params["T_inf"]
        scenario = params["scenario"]

        # Грубая сетка для скорости.
        bcs = {}
        if scenario == "all_conv":
            for f in range(6):
                bcs[f] = BoundaryCondition(type=BC_ROBIN, alpha=alpha, T_inf=t_inf)
        elif scenario == "source_conv":
            for f in range(6):
                bcs[f] = BoundaryCondition(type=BC_ROBIN, alpha=alpha, T_inf=t_inf)
            if Q <= 0:
                Q = 1.0e5  # дать ненулевой источник для наглядности
        elif scenario == "one_hot":
            bcs[FACE_X_MINUS] = BoundaryCondition(type=BC_DIRICHLET, T0=100.0)
            for f in range(1, 6):
                bcs[f] = BoundaryCondition(type=BC_ROBIN, alpha=alpha, T_inf=t_inf)

        p = Problem(geometry=BoxGeometry(Lx=0.1, Ly=0.1, Lz=0.1,
                                          nx=8, ny=8, nz=8),
                    lambda_=lam, Q=Q, bcs=bcs)
        try:
            with CoreBridge() as br:
                p.build_mesh_in_core(br)
                info = p.solve(br, tol=1e-7, max_iter=3000)
            bal = p.energy_balance()
            txt = (f"<b>T:</b> {p.T.min():.2f} … {p.T.max():.2f} °C "
                   f"(перепад {p.T.max()-p.T.min():.2f})  ·  "
                   f"<b>среднее</b> {p.T.mean():.2f} °C<br>"
                   f"λ={lam:.4g}, α={alpha:.4g}, Q={Q:.3g}, T∞={t_inf:.1f}  ·  "
                   f"{info.iterations} итер, "
                   f"{info.time_seconds*1000:.0f} мс")
            if bal:
                txt += f"  ·  энергобаланс {bal['rel_err']*100:.2f}%"
            self.whatif_view.show_result(txt)
            # Показать поле в 3D.
            self._whatif_problem = p
            try:
                from fem3d.mesh import _extract_surface_faces
                bnd = _extract_surface_faces(p.elements)
                self.viz.set_mesh(p.nodes, p.elements, bnd)
                self.viz.set_temperature(p.T)
            except Exception:
                pass
        except Exception as exc:
            self.whatif_view.show_result(f"Ошибка расчёта: {exc}")

    def _build_or_show_transient_player(self) -> None:
        """Создать (или показать) плеер для просмотра T(t) по снимкам."""
        if not hasattr(self, "_transient_player_widget"):
            from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSlider
            w = QWidget()
            lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
            self._tp_play_btn = QPushButton("▶")
            self._tp_play_btn.setMaximumWidth(36)
            self._tp_play_btn.clicked.connect(self._on_transient_play_toggle)
            lay.addWidget(self._tp_play_btn)
            self._tp_slider = QSlider(Qt.Horizontal)
            self._tp_slider.setMinimum(0)
            self._tp_slider.valueChanged.connect(self._on_transient_slider)
            lay.addWidget(self._tp_slider, 1)
            self._tp_label = QLabel("t = 0.000 с")
            self._tp_label.setMinimumWidth(300)
            self._tp_label.setStyleSheet(
                'font-family: "Consolas", "DejaVu Sans Mono", monospace; '
                "font-size: 9pt;")
            lay.addWidget(self._tp_label)
            self._transient_player_widget = w
            self._tp_timer = QTimer(self)
            self._tp_timer.setInterval(120)  # 120 мс = ~8 FPS
            self._tp_timer.timeout.connect(self._on_transient_timer)
            self._tp_playing = False
            # Добавляем плеер под 3D-вид (в layout родителя).
            parent_layout = self.viz.parent().layout()
            if parent_layout is not None:
                parent_layout.addWidget(w)
        # Настройка слайдера. Стартуем с НАЧАЛЬНОГО кадра (t=0), чтобы было
        # видно нагретое тело и последующее остывание, а не конечный кадр.
        n = len(self._transient_times)
        self._tp_slider.setMaximum(n - 1)
        self._tp_slider.setValue(0)
        # Скорость анимации из параметров (кадр/с).
        p = getattr(self, "_transient_params", None) or {}
        fps = max(1, int(p.get("fps", 8)))
        self._tp_timer.setInterval(int(round(1000.0 / fps)))
        self._tp_loop = bool(p.get("loop", False))
        self._transient_player_widget.show()

    def _autostart_transient_animation(self) -> None:
        """Автоматически запустить проигрывание T(t) с начала."""
        if not hasattr(self, "_tp_slider"):
            return
        self._tp_slider.setValue(0)
        if not getattr(self, "_tp_playing", False):
            # Имитируем нажатие ▶: запускаем таймер с кадра 0.
            self._tp_playing = True
            self._tp_play_btn.setText("⏸")
            self._tp_timer.start()

    def _on_transient_slider(self, idx: int) -> None:
        if not hasattr(self, "_transient_T_history") \
                or self._transient_T_history is None:
            return
        T_at = self._transient_T_history[idx]
        t_at = self._transient_times[idx]
        self.viz.set_temperature(T_at)
        self._tp_label.setText(
            f"t = {t_at:.3g} с  ({idx + 1}/{len(self._transient_times)})   "
            f"T: {T_at.min():.1f} / {T_at.mean():.1f} / {T_at.max():.1f} °C")

    def _on_transient_play_toggle(self) -> None:
        self._tp_playing = not self._tp_playing
        self._tp_play_btn.setText("⏸" if self._tp_playing else "▶")
        if self._tp_playing:
            if self._tp_slider.value() >= self._tp_slider.maximum():
                self._tp_slider.setValue(0)
            self._tp_timer.start()
        else:
            self._tp_timer.stop()

    def _on_transient_timer(self) -> None:
        v = self._tp_slider.value()
        if v >= self._tp_slider.maximum():
            if getattr(self, "_tp_loop", False):
                self._tp_slider.setValue(0)
                return
            self._tp_timer.stop()
            self._tp_playing = False
            self._tp_play_btn.setText("▶")
            return
        self._tp_slider.setValue(v + 1)

    def _run_steady(self) -> None:
        # Возврат из нестационарного режима: авто-шкала и скрытый плеер.
        if hasattr(self.viz, "clear_fixed_clim"):
            self.viz.clear_fixed_clim()
        if hasattr(self, "_tp_timer"):
            self._tp_timer.stop()
            self._tp_playing = False
            self._tp_play_btn.setText("▶")
        if hasattr(self, "_transient_player_widget"):
            self._transient_player_widget.hide()

        self.run_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.result_label.setText("Идёт расчёт...")

        self._thread = QThread(self)
        self._worker = SolverWorker(self.problem,
                                     tol=self.settings.cg_tolerance,
                                     max_iter=self.settings.cg_max_iter,
                                     omp_threads=self.settings.omp_threads)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.result_label.setText)
        self._worker.cg_progress.connect(self._on_cg_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_cg_progress(self, iteration: int, residual: float, percent: int) -> None:
        """Слот прогресса CG: обновляет прогресс-бар и текст."""
        self.progress.setValue(percent)
        self.result_label.setText(
            f"CG итерация {iteration}, невязка {residual:.2e}"
        )

    def _on_cancel_solve(self) -> None:
        """Кнопка «Прервать»: посылает запрос на отмену в worker."""
        if self._worker is not None:
            self.cancel_button.setEnabled(False)
            self.result_label.setText("Отмена...")
            self._worker.request_cancel()

    def _on_worker_finished(self, info: SolverInfo) -> None:
        self._refresh_view_from_problem(rebuild_geometry=False)
        self.plots_view.set_results(self.problem.nodes,
                                     self.problem.T, self.problem.flux)
        # Передаём поле потоков в 3D-вид (для стрелок, если включены).
        self.viz.set_flux_field(self.problem.flux)
        Tmin, Tmax = self.problem.temperature_range()
        Tmean = float(self.problem.T.mean()) if self.problem.T is not None else 0.0
        status = (
            f"T = {Tmin:.2f}…{Tmax:.2f}°C   "
            f"среднее {Tmean:.2f}°C   перепад {Tmax - Tmin:.2f}°C   "
            f"итераций: {info.iterations}   "
            f"невязка: {info.residual:.2e}   "
            f"время: {info.time_seconds*1000:.1f} мс   "
            f"{'сошёлся' if info.converged else 'НЕ сошёлся'}"
        )
        # Энергобаланс — если посчитается без ошибок (диагностика качества).
        try:
            bal = self.problem.energy_balance()
            if bal and "rel_err" in bal:
                status += f"   энергобаланс: {bal['rel_err']*100:.2f}%"
        except Exception:
            pass
        # Величины конвекции при обдуве — выводим прямо в статус.
        try:
            if getattr(self.problem, "air_flow_enabled", False):
                from fem3d import convection as cv
                res = cv.analyze_problem_air_flow(self.problem, T_surface=Tmean)
                if res is not None:
                    status += (f"   ·   обдув: Re={res.Re:.2g}, "
                               f"Nu={res.Nu:.0f}, h={res.h:.1f} Вт/(м²·К), "
                               f"Q={res.Q_total:.0f} Вт")
        except Exception:
            pass
        self.result_label.setText(status)
        self.btn_vtu.setEnabled(True); self.btn_csv.setEnabled(True); self.btn_report.setEnabled(True)
        if self.settings.auto_save_calculation:
            self.calc_view.add_record(
                self.problem,
                title=f"λ={self.problem.lambda_:g}, {Tmin:.1f}..{Tmax:.1f}°C")

    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Ошибка расчёта", msg)
        self.result_label.setText("Расчёт прерван.")

    def _on_thread_finished(self) -> None:
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)
        self.run_button.setEnabled(True)
        if self._thread: self._thread.deleteLater()
        if self._worker: self._worker.deleteLater()
        self._thread = None; self._worker = None

    def _on_calc_selected(self, record: CalculationRecord) -> None:
        self.problem = record.problem
        self.plots_view.set_results(self.problem.nodes,
                                     self.problem.T, self.problem.flux)
        self._refresh_view_from_problem(rebuild_geometry=True)
        self.viz.set_flux_field(self.problem.flux)
        self._sync_summaries()

    # =========================================================================
    # Pick mode и 3D-взаимодействие.
    # =========================================================================

    def _set_pick_mode(self, mode: str) -> None:
        # Сбрасываем флаги двухкликовой sphere-машины.
        if mode != "place_source":
            self._sphere_pick_state = None
            self._sphere_center = None
            self.viz.clear_preview_markers()
            self.card_sources.set_pick_point_active(False)
            self.card_sources.set_pick_sphere_active(False)
        # Сбрасываем флаги pick_line при выходе из этого режима.
        if mode != "pick_line":
            self._line_pick_state = None
            self._line_point_a = None
            if hasattr(self, "act_pick_line"):
                self.act_pick_line.setChecked(False)
        if mode == "pick_node":
            self.act_place_source.setChecked(False)
        elif mode == "place_source":
            self.act_pick_node.setChecked(False)
        else:
            self.act_pick_node.setChecked(False)
            self.act_place_source.setChecked(False)
        self.viz.set_pick_mode(mode)

    def _on_node_picked(self, idx: int, x: float, y: float, z: float,
                         T: float) -> None:
        t_str = "—" if T != T else f"{T:.3f} °C"

        # Режим установки точки наблюдения.
        if getattr(self, "_obs_pick_active", False):
            self.problem.observation_points.append((x, y, z))
            self._obs_pick_active = False
            self.viz.set_pick_mode("none")
            self._refresh_observation_markers()
            n = len(self.problem.observation_points)
            self.statusBar().showMessage(
                f"Точка наблюдения #{n} установлена: "
                f"({x:.3g}, {y:.3g}, {z:.3g})", 4000)
            return

        # Режим pick_line: собираем 2 клика.
        if self._line_pick_state == "awaiting_a":
            self._line_point_a = (x, y, z)
            self.viz.clear_preview_markers()
            self.viz.add_preview_marker(x, y, z, color="#3aa55a")
            self._line_pick_state = "awaiting_b"
            self.statusBar().showMessage(
                f"2/2: кликните вторую точку B  (A = ({x:.3g}, {y:.3g}, {z:.3g}))",
                0)
            return
        if self._line_pick_state == "awaiting_b":
            point_b = (x, y, z)
            point_a = self._line_point_a
            self.viz.clear_preview_markers()
            self._line_pick_state = None
            self._line_point_a = None
            # Сброс кнопки/режима viz.
            if hasattr(self, "act_pick_line"):
                self.act_pick_line.setChecked(False)
            self.viz.set_pick_mode("none")
            self.statusBar().clearMessage()
            # Покажем профиль в отдельном окне.
            self._show_line_profile(point_a, point_b)
            return

        # Обычный pick_node — просто статус-строка.
        self.pick_status_label.setText(
            f"Узел #{idx}: ({x:.4g}, {y:.4g}, {z:.4g})  →  T = {t_str}"
        )

    def _show_line_profile(self, point_a, point_b) -> None:
        """Построить и показать график T(x) вдоль отрезка [A, B]."""
        if self.problem.T is None:
            QMessageBox.information(
                self, "Нет данных",
                "Сначала выполните расчёт — для построения профиля нужно поле T.")
            return
        distances, temperatures = compute_temperature_profile(
            self.problem, point_a, point_b, n_samples=120)
        if distances is None:
            QMessageBox.information(self, "Ошибка", "Не удалось построить профиль.")
            return
        # Открываем окно с графиком.
        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from PyQt5.QtWidgets import QDialog, QVBoxLayout
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"matplotlib недоступен: {exc}")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Профиль температуры вдоль линии")
        dlg.resize(800, 520)
        outer = QVBoxLayout(dlg)
        bg = current_theme().panel
        fg = current_theme().text
        fig = Figure(figsize=(8, 5), facecolor=bg)
        canvas = FigureCanvasQTAgg(fig)
        outer.addWidget(canvas)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        ax.plot(distances, temperatures, "-", linewidth=2,
                color=current_theme().accent)
        ax.set_xlabel("Расстояние от точки A, м", color=fg)
        ax.set_ylabel("T, °C", color=fg)
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color(fg)
        a_str = f"({point_a[0]:.4g}, {point_a[1]:.4g}, {point_a[2]:.4g})"
        b_str = f"({point_b[0]:.4g}, {point_b[1]:.4g}, {point_b[2]:.4g})"
        ax.set_title(f"Профиль T между A = {a_str}  и  B = {b_str}",
                     color=fg, fontsize=10)
        ax.grid(alpha=0.3)
        # Подписи Tmin / Tmax.
        i_min = int(np.argmin(temperatures)); i_max = int(np.argmax(temperatures))
        ax.annotate(f"Tmin = {temperatures[i_min]:.2f} °C",
                    xy=(distances[i_min], temperatures[i_min]),
                    xytext=(15, -25), textcoords="offset points",
                    color="#3aa5ff", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#3aa5ff"))
        ax.annotate(f"Tmax = {temperatures[i_max]:.2f} °C",
                    xy=(distances[i_max], temperatures[i_max]),
                    xytext=(15, 15), textcoords="offset points",
                    color="#ff7b3a", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#ff7b3a"))
        canvas.draw()
        dlg.exec_()

    def _on_pick_point_mode(self, active: bool) -> None:
        """Слот переключателя «🖱 +Точка»."""
        if active:
            # Выключаем sphere mode и другие picking-режимы.
            self.card_sources.set_pick_sphere_active(False)
            self._sphere_pick_state = None
            self.viz.clear_preview_markers()
            self.act_pick_node.setChecked(False)
            self.viz.set_pick_mode("place_source")
            self.statusBar().showMessage(
                "Кликните в 3D-виде, чтобы поставить точечный источник.", 0)
        else:
            self.viz.set_pick_mode("none")
            self.statusBar().clearMessage()

    def _on_pick_sphere_mode(self, active: bool) -> None:
        """Слот переключателя «🖱 +Сфера». Реализует state-машину 2 кликов."""
        if active:
            # Выключаем точечный режим.
            self.card_sources.set_pick_point_active(False)
            self.act_pick_node.setChecked(False)
            self._sphere_pick_state = "awaiting_center"
            self._sphere_center = None
            self.viz.clear_preview_markers()
            self.viz.set_pick_mode("place_source")  # сигнал point_clicked
            self.statusBar().showMessage(
                "1/2: кликните в 3D-виде, чтобы задать ЦЕНТР сферы.", 0)
        else:
            self._sphere_pick_state = None
            self._sphere_center = None
            self.viz.clear_preview_markers()
            self.viz.set_pick_mode("none")
            self.statusBar().clearMessage()

    def _on_pick_line_mode(self, active: bool) -> None:
        """Слот режима «Профиль T вдоль линии». Собирает 2 клика."""
        if active:
            if self.problem.T is None:
                QMessageBox.information(
                    self, "Нет данных",
                    "Сначала выполните расчёт — нужно поле T.")
                self.act_pick_line.setChecked(False)
                return
            # Сбрасываем другие режимы.
            self.act_pick_node.setChecked(False)
            self.act_place_source.setChecked(False)
            self.card_sources.set_pick_point_active(False)
            self.card_sources.set_pick_sphere_active(False)
            self._sphere_pick_state = None
            self._line_pick_state = "awaiting_a"
            self._line_point_a = None
            self.viz.clear_preview_markers()
            self.viz.set_pick_mode("pick_line")
            self.statusBar().showMessage(
                "1/2: кликните первую точку A в 3D-виде.", 0)
        else:
            self._line_pick_state = None
            self._line_point_a = None
            self.viz.clear_preview_markers()
            self.viz.set_pick_mode("none")
            self.statusBar().clearMessage()

    def _on_point_clicked(self, x: float, y: float, z: float) -> None:
        """Клик в 3D-виде в режиме place_source.
        В зависимости от того, какая кнопка активна, ставим точку или
        обрабатываем двухкликовое размещение сферы."""
        if self.problem.nodes is None:
            return

        # ----- Режим «сфера двумя кликами» -----
        if self._sphere_pick_state == "awaiting_center":
            self._sphere_center = (x, y, z)
            # Маркер выбранного центра.
            self.viz.clear_preview_markers()
            self.viz.add_preview_marker(x, y, z,
                                         color=current_theme().accent)
            self._sphere_pick_state = "awaiting_radius"
            self.statusBar().showMessage(
                f"2/2: кликните точку на ГРАНИЦЕ сферы "
                f"(центр в ({x:.3g}, {y:.3g}, {z:.3g})).", 0)
            return

        if self._sphere_pick_state == "awaiting_radius":
            cx, cy, cz = self._sphere_center
            radius = float(np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2))
            self.viz.clear_preview_markers()
            self._sphere_pick_state = None
            self.card_sources.set_pick_sphere_active(False)
            self.viz.set_pick_mode("none")
            self.statusBar().clearMessage()
            # Открываем диалог с предзаполнёнными параметрами.
            d = VolumeSourceDialog(self.problem.geometry,
                                    prefill=(cx, cy, cz),
                                    parent=self)
            # У VolumeSourceDialog есть поля cx/cy/cz/r/q — выставим радиус.
            try:
                d.r_spin.setValue(radius)
            except Exception:
                pass
            if d.exec_() == QDialog.Accepted:
                fcx, fcy, fcz, fr, Q0 = d.values()
                self.problem.volume_sources.append(VolumeSource(
                    shape=VOLSRC_SPHERE, params=(fcx, fcy, fcz, fr), Q0=Q0))
                self._sync_summaries()
                self._refresh_source_markers()
            return

        # ----- Режим «точка» (стандартный) -----
        d = PointSourceDialog(self.problem.geometry, prefill=(x, y, z),
                              parent=self)
        if d.exec_() == QDialog.Accepted:
            xx, yy, zz, P = d.values()
            diff = self.problem.nodes - np.array([xx, yy, zz])
            idx = int(np.argmin(np.sum(diff * diff, axis=1)))
            self.problem.point_sources.append(PointSource(node_idx=idx, power=P))
            self._sync_summaries(); self._refresh_source_markers()
            # Если активирована кнопка точечного размещения — выключим,
            # чтобы случайно не наставить десяток источников подряд.
            self.card_sources.set_pick_point_active(False)
            self.viz.set_pick_mode("none")
            self.statusBar().clearMessage()

    def _on_hover(self, x: float, y: float, z: float, T: float) -> None:
        self.pick_status_label.setText(
            f"Курсор: ({x:.4g}, {y:.4g}, {z:.4g})  T ≈ {T:.2f} °C"
        )

    def _on_slice_changed(self, *_args) -> None:
        axis = self.slice_combo.currentData()
        pos = self.slice_slider.value() / 100.0
        self.viz.set_slice(axis, pos)

    # =========================================================================
    # Экспорт.
    # =========================================================================

    def _export_vtu(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт VTU", "result.vtu", "VTK Unstructured (*.vtu)")
        if not path: return
        try:
            export_vtu(self.problem, path)
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт CSV", "result.csv", "CSV (*.csv)")
        if not path: return
        try:
            export_csv(self.problem, path)
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def _export_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Отчёт", "report.txt", "Текстовый файл (*.txt)")
        if not path: return
        try:
            export_report(self.problem, path)
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def _export_transient_report(self) -> None:
        """Отдельный отчёт по нестационарному расчёту (+ CSV истории T(t))."""
        times = getattr(self, "_transient_times", None)
        hist = getattr(self, "_transient_T_history", None)
        if times is None or hist is None:
            QMessageBox.information(
                self, "Нет данных",
                "Сначала выполните нестационарный расчёт "
                "(галочка «τ Нестационарный» + «Запустить расчёт»).")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Отчёт по нестационарному расчёту",
            "transient_report.txt", "Текстовый файл (*.txt)")
        if not path:
            return
        try:
            from fem3d.postprocess import (export_transient_report,
                                            export_transient_history_csv)
            params = getattr(self, "_transient_params", None)
            export_transient_report(self.problem, times, hist, path,
                                    params=params)
            # Рядом — CSV с историей (тот же путь, суффикс _history.csv).
            base, _ext = os.path.splitext(path)
            csv_path = base + "_history.csv"
            export_transient_history_csv(self.problem, times, hist, csv_path)
            self.statusBar().showMessage(
                f"Сохранено: {path} и {csv_path}", 7000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def _export_pdf(self) -> None:
        """Экспорт сводного отчёта в PDF. По возможности — со скриншотом 3D."""
        if self.problem.T is None:
            QMessageBox.information(
                self, "Нет данных", "Сначала выполните расчёт.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Отчёт в PDF", "report.pdf", "PDF (*.pdf)")
        if not path: return
        # Сделаем временный скриншот 3D и приложим к PDF.
        import tempfile
        tmpdir = tempfile.gettempdir()
        screenshot = os.path.join(tmpdir, "fem_heat3d_screenshot.png")
        if not self.viz.screenshot(screenshot):
            screenshot = None
        try:
            export_pdf_report(self.problem, path,
                               screenshot_path=screenshot,
                               author="")
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
        finally:
            if screenshot and os.path.isfile(screenshot):
                try: os.unlink(screenshot)
                except Exception: pass

    def _save_screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить скриншот 3D", "screenshot.png",
            "PNG (*.png);;JPEG (*.jpg)")
        if not path: return
        ok = self.viz.screenshot(path)
        if ok:
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
        else:
            QMessageBox.warning(self, "Скриншот",
                                 "Не удалось сохранить скриншот.")

    # =========================================================================
    # Сохранение / загрузка проектов (.fem3d).
    # =========================================================================

    def _on_new_project(self) -> None:
        if QMessageBox.question(
            self, "Новый проект",
            "Сбросить текущий проект? Несохранённые изменения будут потеряны.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) != QMessageBox.Yes:
            return
        self.problem = Problem()
        self._current_project_path = None
        self._sync_summaries()
        self._refresh_view_from_problem(rebuild_geometry=True)
        self.result_label.setText("Готов к расчёту")
        self.statusBar().showMessage("Новый проект.", 3000)
        self._update_window_title()

    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть проект", "",
            f"Проекты fem3d (*{PROJECT_EXTENSION});;Все файлы (*.*)"
        )
        if not path:
            return
        self._load_project_from_path(path)

    def _load_project_from_path(self, path: str) -> None:
        try:
            problem = load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть проект:\n{exc}")
            return
        self.problem = problem
        self._current_project_path = path

        # Если в проекте есть результаты — но нет сетки в ядре, всё равно
        # восстановим визуализацию из сохранённых nodes/elements.
        self._sync_summaries()
        if self.problem.nodes is not None and self.problem.elements is not None:
            self._refresh_view_from_problem(rebuild_geometry=True)
            if self.problem.T is not None:
                self.plots_view.set_results(self.problem.nodes,
                                             self.problem.T, self.problem.flux)
                Tmin, Tmax = self.problem.temperature_range()
                self.result_label.setText(
                    f"[загружено] Tmin = {Tmin:.2f}°C, Tmax = {Tmax:.2f}°C")
                self.btn_vtu.setEnabled(True)
                self.btn_csv.setEnabled(True)
                self.btn_report.setEnabled(True)
        else:
            # Сетки нет — попробуем построить.
            self._on_generate_mesh()

        self.statusBar().showMessage(f"Открыто: {path}", 5000)
        self._add_recent_file(path)
        self._update_window_title()

    def _on_save_project(self) -> None:
        if not self._current_project_path:
            self._on_save_project_as()
            return
        try:
            save_project(self.problem, self._current_project_path)
            self.statusBar().showMessage(
                f"Сохранено: {self._current_project_path}", 5000)
            self._add_recent_file(self._current_project_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{exc}")

    def _on_save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект",
            self._current_project_path or "project" + PROJECT_EXTENSION,
            f"Проекты fem3d (*{PROJECT_EXTENSION})"
        )
        if not path:
            return
        try:
            save_project(self.problem, path)
            self._current_project_path = path
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
            self._add_recent_file(path)
            self._update_window_title()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{exc}")

    def _update_window_title(self) -> None:
        base = "Программный комплекс МКЭ — расчёт теплопроводности"
        if self._current_project_path:
            name = os.path.basename(self._current_project_path)
            self.setWindowTitle(f"{name} — {base}")
        else:
            self.setWindowTitle(base)

    # =========================================================================
    # Drag-and-drop файлов в окно.
    # =========================================================================

    def dragEnterEvent(self, event):
        # Принимаем перетаскивание любых файлов; фильтр по расширению — в drop.
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        ext = os.path.splitext(path)[1].lower()
        if ext == PROJECT_EXTENSION:
            self._load_project_from_path(path)
        elif ext in SUPPORTED_IMPORT_EXTENSIONS:
            try:
                nodes, tets, bnd_nodes, bnd_face_ids = import_mesh_file(path)
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка импорта", str(exc))
                return
            self._apply_external_mesh(nodes, tets, bnd_nodes, bnd_face_ids)
            self._on_generate_mesh()
        else:
            self.statusBar().showMessage(
                f"Неизвестный тип файла: {ext}", 3000)

    # =========================================================================
    # Темы и настройки.
    # =========================================================================

    def _on_open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        old_theme = self.settings.theme
        dlg.save_to(self.settings)
        idx = self.mode_combo.findData(self.settings.render_mode)
        if idx >= 0: self.mode_combo.setCurrentIndex(idx)
        self.iso_spin.setValue(self.settings.iso_count)
        if self.settings.theme != old_theme:
            self._switch_theme(self.settings.theme)

    def _switch_theme(self, name: str) -> None:
        set_theme(name)
        self.settings.theme = name
        # Обновить чекбоксы в подменю темы.
        for ac in self._theme_group.actions():
            ac.setChecked(ac.data() == name)
        # Применить stylesheet ко всему приложению.
        app = QApplication.instance()
        if app:
            app.setStyleSheet(build_stylesheet())
            app.setPalette(build_palette())
        # Обновить виджеты с динамическими стилями.
        self.apply_theme_to_widgets()

    def apply_theme_to_widgets(self) -> None:
        t = current_theme()
        # 3D-вьюпорт.
        self.viz.set_viewport_background(t.viewport_bg)
        # Вкладка «Графики»: фигура matplotlib в цветах темы.
        if hasattr(self, "plots_view") and hasattr(self.plots_view,
                                                    "apply_theme"):
            self.plots_view.apply_theme()
        # Вкладка «Что будет, если…»: панель результата.
        if hasattr(self, "whatif_view") and hasattr(self.whatif_view,
                                                     "apply_theme"):
            self.whatif_view.apply_theme()
        # Текст-описания (text_dim).
        self.info_label.setStyleSheet(f"color: {t.text_dim}; font-size: 9pt;")
        self.pick_status_label.setStyleSheet(f"color: {t.text_dim}; font-size: 9pt;")
        self.card_geometry.apply_theme()
        self.card_material.apply_theme()
        self.card_regions.apply_theme()
        self.card_sources.apply_theme()
        # Чипы граней (рамка с цветом ГУ).
        for chip in self.face_chips.values():
            chip._refresh()

    def _on_show_help(self) -> None:
        HelpDialog(self).exec_()

    def _on_calibration_test(self) -> None:
        """Проверить точность решателя на 5 физических задачах с
        аналитическими решениями (стенка, конвекция, источник, два материала, поток).
        """
        from PyQt5.QtWidgets import QTextEdit
        dlg = QDialog(self)
        dlg.setWindowTitle("Проверка точности на эталонных задачах")
        dlg.resize(720, 520)
        layout = QVBoxLayout(dlg)
        hint = QLabel(
            "<b>Калибровка решателя</b> — 5 физических задач с известными "
            "аналитическими решениями.<br>"
            "Идёт прогон... подождите."
        )
        hint.setTextFormat(Qt.RichText)
        layout.addWidget(hint)
        report = QTextEdit(); report.setReadOnly(True)
        report.setFontFamily("monospace")
        layout.addWidget(report, 1)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject); btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.show(); QApplication.processEvents()

        # Запускаем верификационные тесты через subprocess.
        import subprocess, sys as _sys, os as _os
        script_path = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "physical_verify.py")
        try:
            res = subprocess.run([_sys.executable, script_path],
                                   capture_output=True, text=True, timeout=120)
            out = res.stdout + (res.stderr if res.returncode else "")
            report.setText(out)
            if "5/5" in out:
                hint.setText("<b>✓ Все 5 эталонных задач пройдены</b> — "
                              "решатель работает корректно.")
            else:
                hint.setText("<b>⚠ Не все тесты пройдены</b> — "
                              "см. вывод ниже.")
        except subprocess.TimeoutExpired:
            report.setText("Таймаут: верификация заняла больше 2 минут.")
        except Exception as exc:
            report.setText(f"Ошибка запуска: {exc}")
        dlg.exec_()

    def _on_show_about(self) -> None:
        QMessageBox.about(
            self, "О программе",
            "<h3>Программный комплекс МКЭ</h3>"
            "<p>Расчёт стационарной теплопроводности методом конечных "
            "элементов на тетраэдрах P1.</p>"
            "<p>Версия 1.4 · Двухуровневая архитектура: C++ ядро + "
            "Python + GUI на PyQt5.</p>"
            f"<p>3D-бэкенд: <b>{self.viz.backend_name}</b></p>"
            f"<p>Тема: <b>{current_theme().title}</b></p>"
        )


# =============================================================================
# Точка входа.
# =============================================================================

def main() -> int:
    app = QApplication(sys.argv)
    set_theme("dark")
    app.setStyleSheet(build_stylesheet())
    app.setPalette(build_palette())

    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
