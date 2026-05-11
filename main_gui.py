# -*- coding: utf-8 -*-
"""
main_gui.py
===========

Главное приложение программного комплекса МКЭ для трёхмерной теплопроводности.

Версия 1.2 — расширенная:
  - вкладочный интерфейс (3D / Графики / Расчёты);
  - 3D-визуализация через PyVista (с резервом на matplotlib);
  - режимы рендера: поверхность / объём / изоповерхности / каркас / сечение;
  - picking узлов с показом T;
  - размещение источников кликом мыши;
  - импорт MSH/VTU/STL/STEP;
  - меню «Справка» и «Настройки»;
  - история выполненных расчётов с возможностью переключения.

Запуск: python main_gui.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, Optional

# -----------------------------------------------------------------------------
try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QPalette
    from PyQt5.QtWidgets import (QAction, QApplication, QComboBox, QDialog,
                                 QDoubleSpinBox, QFileDialog, QFrame,
                                 QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                                 QListWidget, QListWidgetItem, QMainWindow,
                                 QMenuBar, QMessageBox, QProgressBar,
                                 QPushButton, QScrollArea, QSlider, QSpinBox,
                                 QSplitter, QStatusBar, QTabWidget,
                                 QToolButton, QVBoxLayout, QWidget)
except ImportError as exc:
    sys.stderr.write(
        "\nОшибка импорта PyQt5: {}\n"
        "Установите: pip install PyQt5\n".format(exc)
    )
    sys.exit(1)

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fem3d import (BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN, MATERIALS,
                   PRESETS, BoundaryCondition, BoxGeometry, BoxPreset,
                   CoreBridge, CoreError, FACE_NAMES, FACE_X_MINUS, FACE_X_PLUS,
                   FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS,
                   Material, PointSource, Problem, SolverInfo,
                   SUPPORTED_IMPORT_EXTENSIONS, VolumeSource,
                   VOLSRC_BOX, VOLSRC_SPHERE, compute_mesh_info,
                   import_mesh_file,
                   template_all_convection, template_bottom_heat_top_cool,
                   template_reset)
from fem3d.postprocess import export_csv, export_report, export_vtu

from gui import (AppSettings, COLOR_TEXT_DIM, CalculationRecord,
                 CalculationsView, FaceCard, HAS_PYVISTA, HelpDialog,
                 PlotsView, PointSourceDialog, SettingsDialog, SolverWorker,
                 STYLESHEET, VolumeSourceDialog, create_view)


# =============================================================================
# Главное окно.
# =============================================================================

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Программный комплекс МКЭ — расчёт теплопроводности")
        self.setMinimumSize(1280, 820)

        self.problem = Problem()
        self.settings = AppSettings()
        self._thread: Optional[QThread] = None
        self._worker: Optional[SolverWorker] = None
        self._msh_path: Optional[str] = None

        self._build_menus()
        self._build_ui()
        self._refresh_view_from_problem(rebuild_geometry=True)
        self._update_3d_backend_status()

    # =========================================================================
    # Меню.
    # =========================================================================

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        # === Файл ============================================================
        file_menu = menubar.addMenu("&Файл")

        act_import = QAction("&Импортировать сетку...", self)
        act_import.setShortcut("Ctrl+I")
        act_import.triggered.connect(self._on_import_mesh)
        file_menu.addAction(act_import)

        file_menu.addSeparator()

        act_export_vtu = QAction("Экспорт &VTU...", self)
        act_export_vtu.triggered.connect(self._export_vtu)
        file_menu.addAction(act_export_vtu)

        act_export_csv = QAction("Экспорт &CSV...", self)
        act_export_csv.triggered.connect(self._export_csv)
        file_menu.addAction(act_export_csv)

        act_export_report = QAction("Сохранить &отчёт...", self)
        act_export_report.triggered.connect(self._export_report)
        file_menu.addAction(act_export_report)

        file_menu.addSeparator()
        act_quit = QAction("В&ыход", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # === Вид ============================================================
        view_menu = menubar.addMenu("&Вид")

        act_reset_camera = QAction("&Сбросить камеру", self)
        act_reset_camera.setShortcut("Home")
        act_reset_camera.triggered.connect(lambda: self.viz.reset_camera())
        view_menu.addAction(act_reset_camera)

        view_menu.addSeparator()

        act_pick_node = QAction("&Узнать T в точке (клик по узлу)", self)
        act_pick_node.setCheckable(True)
        act_pick_node.triggered.connect(
            lambda checked: self._set_pick_mode("pick_node" if checked else "none"))
        self.act_pick_node = act_pick_node
        view_menu.addAction(act_pick_node)

        act_place_source = QAction("&Поставить источник кликом", self)
        act_place_source.setCheckable(True)
        act_place_source.triggered.connect(
            lambda checked: self._set_pick_mode("place_source" if checked else "none"))
        self.act_place_source = act_place_source
        view_menu.addAction(act_place_source)

        # === Источники ======================================================
        src_menu = menubar.addMenu("&Источники")
        act_add_pt = QAction("Добавить точечный...", self)
        act_add_pt.triggered.connect(self._on_add_point_source_dialog)
        src_menu.addAction(act_add_pt)
        act_add_vol = QAction("Добавить объёмный (сфера)...", self)
        act_add_vol.triggered.connect(self._on_add_volume_source_dialog)
        src_menu.addAction(act_add_vol)
        src_menu.addSeparator()
        act_clear = QAction("Очистить все источники", self)
        act_clear.triggered.connect(self._on_clear_sources)
        src_menu.addAction(act_clear)

        # === Настройки ======================================================
        settings_menu = menubar.addMenu("&Настройки")
        act_settings = QAction("&Параметры...", self)
        act_settings.setShortcut("Ctrl+,")
        act_settings.triggered.connect(self._on_open_settings)
        settings_menu.addAction(act_settings)

        # === Справка ========================================================
        help_menu = menubar.addMenu("Спр&авка")
        act_help = QAction("&Содержание", self)
        act_help.setShortcut("F1")
        act_help.triggered.connect(self._on_show_help)
        help_menu.addAction(act_help)

        act_about = QAction("О программе...", self)
        act_about.triggered.connect(self._on_show_about)
        help_menu.addAction(act_about)

    # =========================================================================
    # Построение UI.
    # =========================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([320, 800, 360])
        root.addWidget(splitter, 1)

        root.addWidget(self._build_bottom_panel(), 0)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    # ---- Левая панель -------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(360)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("<b>Управление моделью</b>")
        title.setStyleSheet("font-size: 11pt;")
        layout.addWidget(title)

        # ----- Геометрия --------------------------------------------------
        geom_box = QGroupBox("Геометрия")
        geom_layout = QVBoxLayout(geom_box)
        geom_layout.setSpacing(6)

        geom_layout.addWidget(QLabel("Тип:"))
        self.geom_type_combo = QComboBox()
        self.geom_type_combo.addItem("Параллелепипед", "box")
        self.geom_type_combo.addItem("Импорт сетки (MSH/VTU/STL/STEP)", "import")
        self.geom_type_combo.currentIndexChanged.connect(self._on_geom_type_changed)
        geom_layout.addWidget(self.geom_type_combo)

        # Параметры параллелепипеда.
        self.box_params_widget = QWidget()
        bp = QVBoxLayout(self.box_params_widget)
        bp.setContentsMargins(0, 0, 0, 0)
        bp.setSpacing(6)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("— Произвольный размер —", None)
        for ps in PRESETS:
            self.preset_combo.addItem(ps.label, ps)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        bp.addWidget(QLabel("Пресет:"))
        bp.addWidget(self.preset_combo)

        size_grid = QGridLayout()
        size_grid.setHorizontalSpacing(6); size_grid.setVerticalSpacing(4)
        for i, lab in enumerate(("X, м:", "Y, м:", "Z, м:")):
            size_grid.addWidget(QLabel(lab), 0, i)
        self.size_x = QDoubleSpinBox(); self.size_x.setRange(1e-4, 100); self.size_x.setDecimals(4); self.size_x.setValue(0.10)
        self.size_y = QDoubleSpinBox(); self.size_y.setRange(1e-4, 100); self.size_y.setDecimals(4); self.size_y.setValue(0.10)
        self.size_z = QDoubleSpinBox(); self.size_z.setRange(1e-4, 100); self.size_z.setDecimals(4); self.size_z.setValue(0.10)
        size_grid.addWidget(self.size_x, 1, 0); size_grid.addWidget(self.size_y, 1, 1); size_grid.addWidget(self.size_z, 1, 2)
        bp.addLayout(size_grid)

        mesh_grid = QGridLayout()
        mesh_grid.setHorizontalSpacing(6); mesh_grid.setVerticalSpacing(4)
        for i, lab in enumerate(("nx:", "ny:", "nz:")):
            mesh_grid.addWidget(QLabel(lab), 0, i)
        self.n_x = QSpinBox(); self.n_x.setRange(2, 200); self.n_x.setValue(15)
        self.n_y = QSpinBox(); self.n_y.setRange(2, 200); self.n_y.setValue(15)
        self.n_z = QSpinBox(); self.n_z.setRange(2, 200); self.n_z.setValue(15)
        mesh_grid.addWidget(self.n_x, 1, 0); mesh_grid.addWidget(self.n_y, 1, 1); mesh_grid.addWidget(self.n_z, 1, 2)
        bp.addLayout(mesh_grid)
        geom_layout.addWidget(self.box_params_widget)

        # Параметры импорта.
        self.import_params_widget = QWidget()
        ip = QVBoxLayout(self.import_params_widget)
        ip.setContentsMargins(0, 0, 0, 0); ip.setSpacing(4)
        self.msh_path_label = QLabel("<i>Файл не выбран</i>")
        self.msh_path_label.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        self.msh_path_label.setWordWrap(True)
        ip.addWidget(self.msh_path_label)
        self.btn_browse_mesh = QPushButton("Выбрать файл...")
        self.btn_browse_mesh.clicked.connect(self._on_import_mesh)
        ip.addWidget(self.btn_browse_mesh)
        geom_layout.addWidget(self.import_params_widget)
        self.import_params_widget.setVisible(False)

        self.gen_button = QPushButton("Сгенерировать сетку")
        self.gen_button.setObjectName("AccentButton")
        self.gen_button.clicked.connect(self._on_generate_mesh)
        geom_layout.addWidget(self.gen_button)
        layout.addWidget(geom_box)

        # ----- Материал ---------------------------------------------------
        mat_box = QGroupBox("Материал")
        mat_layout = QVBoxLayout(mat_box)
        mat_layout.setSpacing(6)
        self.material_combo = QComboBox()
        self.material_combo.addItem("— Произвольный —", None)
        for m in MATERIALS:
            self.material_combo.addItem(f"{m.name}  (λ = {m.lambda_:g})", m)
        for i in range(self.material_combo.count()):
            data = self.material_combo.itemData(i)
            if isinstance(data, Material) and data.name == "Алюминий":
                self.material_combo.setCurrentIndex(i)
                break
        self.material_combo.currentIndexChanged.connect(self._on_material_changed)
        mat_layout.addWidget(self.material_combo)
        param_grid = QGridLayout()
        param_grid.setHorizontalSpacing(6); param_grid.setVerticalSpacing(4)
        param_grid.addWidget(QLabel("λ, Вт/(м·К):"), 0, 0)
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(1e-4, 1e5); self.lambda_spin.setDecimals(4); self.lambda_spin.setValue(237.0)
        self.lambda_spin.valueChanged.connect(self._sync_to_problem)
        param_grid.addWidget(self.lambda_spin, 0, 1)
        param_grid.addWidget(QLabel("Q, Вт/м³:"), 1, 0)
        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(-1e9, 1e9); self.q_spin.setDecimals(2); self.q_spin.setValue(0.0)
        self.q_spin.valueChanged.connect(self._sync_to_problem)
        param_grid.addWidget(self.q_spin, 1, 1)
        mat_layout.addLayout(param_grid)
        layout.addWidget(mat_box)

        # ----- Информация о сетке ----------------------------------------
        info_box = QGroupBox("Информация о сетке")
        info_layout = QVBoxLayout(info_box)
        self.info_label = QLabel("Сетка не построена.")
        self.info_label.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_box)

        # ----- Локальные источники ---------------------------------------
        src_box = QGroupBox("Локальные источники")
        sl = QVBoxLayout(src_box)
        sl.setSpacing(4)
        self.sources_list = QListWidget()
        self.sources_list.setMaximumHeight(120)
        sl.addWidget(self.sources_list)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_add_pt = QPushButton("+ Точка")
        self.btn_add_pt.clicked.connect(self._on_add_point_source_dialog)
        self.btn_add_vol = QPushButton("+ Сфера")
        self.btn_add_vol.clicked.connect(self._on_add_volume_source_dialog)
        self.btn_remove_src = QPushButton("Удалить")
        self.btn_remove_src.clicked.connect(self._on_remove_source)
        for b in (self.btn_add_pt, self.btn_add_vol, self.btn_remove_src):
            btn_row.addWidget(b)
        sl.addLayout(btn_row)
        layout.addWidget(src_box)

        layout.addStretch(1)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return panel

    # ---- Центральная панель: вкладки ---------------------------------------
    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ----- Вкладка 3D ---------------------------------------------------
        tab_3d = QWidget()
        tab_3d_layout = QVBoxLayout(tab_3d)
        tab_3d_layout.setContentsMargins(8, 8, 8, 8)

        # Верхняя строка управления.
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("<b>3D-вид модели</b>"))
        top_row.addStretch(1)
        top_row.addWidget(QLabel("Режим:"))
        self.mode_combo = QComboBox()
        for label, value in [
            ("Поверхность", "surface"),
            ("Объёмный рендер", "volume"),
            ("Изоповерхности", "isosurface"),
            ("Каркас", "wireframe"),
        ]:
            self.mode_combo.addItem(label, value)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        top_row.addWidget(self.mode_combo)
        self.btn_reset_camera = QPushButton("Сбросить вид")
        self.btn_reset_camera.clicked.connect(lambda: self.viz.reset_camera())
        top_row.addWidget(self.btn_reset_camera)
        tab_3d_layout.addLayout(top_row)

        # Само 3D.
        self.viz = create_view(self)
        tab_3d_layout.addWidget(self.viz, 1)
        self.viz.node_picked.connect(self._on_node_picked)
        self.viz.point_clicked.connect(self._on_point_clicked)

        # Нижняя строка: сечение + статус picking.
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Сечение:"))
        self.slice_combo = QComboBox()
        self.slice_combo.addItem("выкл", None)
        for a in ("x", "y", "z"):
            self.slice_combo.addItem(a, a)
        self.slice_combo.currentIndexChanged.connect(self._on_slice_changed)
        bottom.addWidget(self.slice_combo)

        bottom.addWidget(QLabel("Положение:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        bottom.addWidget(self.slice_slider, 1)

        bottom.addWidget(QLabel("Изо:"))
        self.iso_spin = QSpinBox()
        self.iso_spin.setRange(2, 20)
        self.iso_spin.setValue(7)
        self.iso_spin.valueChanged.connect(
            lambda v: self.viz.set_isosurface_count(int(v)))
        bottom.addWidget(self.iso_spin)

        tab_3d_layout.addLayout(bottom)

        self.pick_status_label = QLabel("")
        self.pick_status_label.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        tab_3d_layout.addWidget(self.pick_status_label)

        self.tabs.addTab(tab_3d, "3D-вид")

        # ----- Вкладка Графики ---------------------------------------------
        self.plots_view = PlotsView()
        self.tabs.addTab(self.plots_view, "Графики")

        # ----- Вкладка Расчёты ---------------------------------------------
        self.calc_view = CalculationsView()
        self.calc_view.selected.connect(self._on_calc_selected)
        self.tabs.addTab(self.calc_view, "Расчёты")

        return panel

    # ---- Правая панель: ГУ ----------------------------------------------
    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(420)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("<b>Граничные условия</b>")
        title.setStyleSheet("font-size: 11pt;")
        layout.addWidget(title)

        self.face_cards: Dict[int, FaceCard] = {}
        order = [FACE_Z_PLUS, FACE_Z_MINUS, FACE_X_PLUS, FACE_X_MINUS,
                 FACE_Y_PLUS, FACE_Y_MINUS]
        for fid in order:
            card = FaceCard(fid)
            card.changed.connect(self._on_bc_changed)
            self.face_cards[fid] = card
            layout.addWidget(card)

        tpl_box = QGroupBox("Шаблоны")
        tpl_layout = QVBoxLayout(tpl_box)
        self.template_combo = QComboBox()
        self.template_combo.addItem("— Применить шаблон —", None)
        self.template_combo.addItem("Нагрев снизу + охлаждение сверху",
                                    template_bottom_heat_top_cool)
        self.template_combo.addItem("Все грани — конвекция (α=10, T∞=20)",
                                    template_all_convection)
        self.template_combo.addItem("Сбросить все условия", template_reset)
        self.template_combo.currentIndexChanged.connect(self._on_template)
        tpl_layout.addWidget(self.template_combo)
        layout.addWidget(tpl_box)
        layout.addStretch(1)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return panel

    # ---- Нижняя панель ---------------------------------------------------
    def _build_bottom_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setFixedHeight(86)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        self.run_button = QPushButton("▶  Запустить расчёт")
        self.run_button.setObjectName("RunButton")
        self.run_button.clicked.connect(self._on_run)
        layout.addWidget(self.run_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress, 1)

        self.result_label = QLabel("Готов к расчёту")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label, 2)

        self.btn_vtu = QPushButton("Экспорт .vtu")
        self.btn_vtu.clicked.connect(self._export_vtu); self.btn_vtu.setEnabled(False)
        layout.addWidget(self.btn_vtu)
        self.btn_csv = QPushButton("Экспорт .csv")
        self.btn_csv.clicked.connect(self._export_csv); self.btn_csv.setEnabled(False)
        layout.addWidget(self.btn_csv)
        self.btn_report = QPushButton("Отчёт .txt")
        self.btn_report.clicked.connect(self._export_report); self.btn_report.setEnabled(False)
        layout.addWidget(self.btn_report)
        return panel

    # =========================================================================
    # Слоты управления.
    # =========================================================================

    def _on_preset_changed(self, _idx: int) -> None:
        ps = self.preset_combo.currentData()
        if ps is None:
            return
        self.size_x.setValue(ps.Lx); self.size_y.setValue(ps.Ly); self.size_z.setValue(ps.Lz)
        self.n_x.setValue(ps.nx);    self.n_y.setValue(ps.ny);    self.n_z.setValue(ps.nz)
        self._sync_to_problem()

    def _on_geom_type_changed(self, _idx: int) -> None:
        kind = self.geom_type_combo.currentData()
        is_box = (kind == "box")
        self.box_params_widget.setVisible(is_box)
        self.import_params_widget.setVisible(not is_box)

    def _on_import_mesh(self) -> None:
        exts = " ".join("*" + e for e in SUPPORTED_IMPORT_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл сетки", "",
            f"Сетки ({exts});;Все файлы (*.*)"
        )
        if not path:
            return
        self._msh_path = path
        self.msh_path_label.setText(f"<b>{os.path.basename(path)}</b>")
        # Если пользователь импортирует через меню, переключим тип геометрии.
        idx = self.geom_type_combo.findData("import")
        if idx >= 0:
            self.geom_type_combo.setCurrentIndex(idx)

    def _on_material_changed(self, _idx: int) -> None:
        m = self.material_combo.currentData()
        if isinstance(m, Material):
            self.lambda_spin.setValue(m.lambda_)
        self._sync_to_problem()

    def _on_bc_changed(self, _face_id: int) -> None:
        self._sync_to_problem()

    def _on_template(self, _idx: int) -> None:
        factory = self.template_combo.currentData()
        if factory is None:
            return
        bcs = factory()
        for fid, bc in bcs.items():
            self.face_cards[fid].set_bc(bc)
        self.template_combo.blockSignals(True)
        self.template_combo.setCurrentIndex(0)
        self.template_combo.blockSignals(False)
        self._sync_to_problem()

    def _on_mode_changed(self, _idx: int) -> None:
        self.viz.set_render_mode(self.mode_combo.currentData())

    def _on_slice_changed(self, *_args) -> None:
        axis = self.slice_combo.currentData()
        pos = self.slice_slider.value() / 100.0
        self.viz.set_slice(axis, pos)

    # ---- Pick mode -----------------------------------------------------
    def _set_pick_mode(self, mode: str) -> None:
        # Сделаем галочки взаимоисключающими.
        if mode == "pick_node":
            self.act_place_source.setChecked(False)
        elif mode == "place_source":
            self.act_pick_node.setChecked(False)
        else:
            self.act_pick_node.setChecked(False)
            self.act_place_source.setChecked(False)
        self.viz.set_pick_mode(mode)
        if mode == "pick_node":
            self.statusBar().showMessage("Кликайте по узлам — внизу появятся координаты и температура.", 4000)
        elif mode == "place_source":
            self.statusBar().showMessage("Кликайте в любой точке — будет добавлен точечный источник.", 4000)
        else:
            self.statusBar().clearMessage()

    def _on_node_picked(self, idx: int, x: float, y: float, z: float, T: float) -> None:
        if T != T:  # NaN
            t_str = "—"
        else:
            t_str = f"{T:.3f} °C"
        self.pick_status_label.setText(
            f"Узел #{idx}: ({x:.4g}, {y:.4g}, {z:.4g})  →  T = {t_str}"
        )

    def _on_point_clicked(self, x: float, y: float, z: float) -> None:
        if self.problem.nodes is None:
            return
        # Привязка к ближайшему узлу + диалог мощности.
        self._open_point_source_dialog(prefill=(x, y, z))

    # ---- Action: Generate mesh -----------------------------------------
    def _sync_to_problem(self) -> None:
        if not self.problem.has_external_mesh():
            self.problem.geometry = BoxGeometry(
                Lx=self.size_x.value(), Ly=self.size_y.value(), Lz=self.size_z.value(),
                nx=self.n_x.value(), ny=self.n_y.value(), nz=self.n_z.value(),
            )
        self.problem.lambda_ = self.lambda_spin.value()
        self.problem.Q = self.q_spin.value()
        for fid, card in self.face_cards.items():
            self.problem.bcs[fid] = card.bc

    def _on_generate_mesh(self) -> None:
        kind = self.geom_type_combo.currentData()
        if kind == "import":
            if not self._msh_path:
                QMessageBox.warning(self, "Файл не выбран",
                                    "Сначала выберите файл сетки.")
                return
            try:
                nodes, tets, bnd_nodes, bnd_face_ids = import_mesh_file(self._msh_path)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Ошибка импорта",
                    f"Не удалось загрузить сетку:\n{exc}")
                return
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
        else:
            self.problem.external_nodes = None
            self.problem.external_elements = None
            self.problem.external_bnd_nodes = None
            self.problem.external_bnd_face_ids = None
            self._sync_to_problem()

        try:
            with CoreBridge() as bridge:
                self.problem.build_mesh_in_core(bridge)
                n_nodes = bridge.n_nodes
                n_elems = bridge.n_elements
                n_faces = bridge.n_boundary_faces
        except CoreError as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить сетку:\n{exc}")
            return

        info = compute_mesh_info(self.problem.nodes, n_elems, n_faces)
        self.info_label.setText(
            f"Узлов: <b>{info.n_nodes}</b><br>"
            f"Элементов: <b>{info.n_elements}</b><br>"
            f"Поверхностных граней: <b>{info.n_boundary_faces}</b><br>"
            f"Габариты: {info.bbox_max[0]-info.bbox_min[0]:.4g} × "
            f"{info.bbox_max[1]-info.bbox_min[1]:.4g} × "
            f"{info.bbox_max[2]-info.bbox_min[2]:.4g} м<br>"
            f"~ память: {info.memory_mb:.2f} МБ"
        )
        self._refresh_view_from_problem(rebuild_geometry=True)

    def _refresh_view_from_problem(self, rebuild_geometry: bool) -> None:
        """Перестраивает 3D-вид: либо полностью (после новой сетки),
        либо только обновляет температуру (после расчёта)."""
        if self.problem.nodes is None or self.problem.elements is None:
            return
        if rebuild_geometry:
            bnd = (self.problem.external_bnd_nodes
                   if self.problem.has_external_mesh()
                   and self.problem.external_bnd_nodes is not None
                   else self._compute_box_boundary_triangles())
            self.viz.set_mesh(self.problem.nodes, self.problem.elements, bnd)
            self.viz.clear_source_markers()
            for ps in self.problem.point_sources:
                if ps.node_idx < self.problem.nodes.shape[0]:
                    nx, ny, nz = self.problem.nodes[ps.node_idx]
                    self.viz.add_source_marker(float(nx), float(ny), float(nz),
                                               color="#ffd24a")
            for vs in self.problem.volume_sources:
                if vs.shape == VOLSRC_SPHERE:
                    cx, cy, cz, _r = vs.params[:4]
                    self.viz.add_source_marker(float(cx), float(cy), float(cz),
                                               color="#ff7b3a")
        self.viz.set_temperature(self.problem.T)
        self._refresh_sources_list()

    def _compute_box_boundary_triangles(self) -> np.ndarray:
        """Извлекает поверхностные треугольники из тетраэдральной сетки.

        Используется когда внешняя bnd-информация недоступна (например, при
        генерации Box — мы могли бы взять её из core, но проще извлечь сами)."""
        from fem3d.mesh import _extract_surface_faces
        if self.problem.elements is None:
            return np.empty((0, 3), dtype=np.int32)
        return _extract_surface_faces(self.problem.elements).astype(np.int32)

    # ---- Локальные источники ------------------------------------------
    def _refresh_sources_list(self) -> None:
        self.sources_list.clear()
        for ps in self.problem.point_sources:
            it = QListWidgetItem(f"⊙  {ps.description()}")
            it.setData(Qt.UserRole, ("point", ps))
            self.sources_list.addItem(it)
        for vs in self.problem.volume_sources:
            it = QListWidgetItem(f"◯  {vs.description()}")
            it.setData(Qt.UserRole, ("volume", vs))
            self.sources_list.addItem(it)

    def _on_add_point_source_dialog(self) -> None:
        if self.problem.nodes is None:
            QMessageBox.information(self, "Нет сетки",
                                    "Сначала сгенерируйте сетку.")
            return
        self._open_point_source_dialog(prefill=None)

    def _open_point_source_dialog(self,
                                   prefill: Optional[tuple] = None) -> None:
        dlg = PointSourceDialog(self.problem.geometry, prefill=prefill, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        x, y, z, P = dlg.values()
        diff = self.problem.nodes - np.array([x, y, z])
        idx = int(np.argmin(np.sum(diff * diff, axis=1)))
        ps = PointSource(node_idx=idx, power=P)
        self.problem.point_sources.append(ps)
        nx, ny, nz = self.problem.nodes[idx]
        self.viz.add_source_marker(float(nx), float(ny), float(nz), color="#ffd24a")
        self._refresh_sources_list()

    def _on_add_volume_source_dialog(self) -> None:
        dlg = VolumeSourceDialog(self.problem.geometry, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        cx, cy, cz, r, Q0 = dlg.values()
        self.problem.volume_sources.append(
            VolumeSource(shape=VOLSRC_SPHERE, params=(cx, cy, cz, r), Q0=Q0)
        )
        self.viz.add_source_marker(float(cx), float(cy), float(cz), color="#ff7b3a")
        self._refresh_sources_list()

    def _on_remove_source(self) -> None:
        item = self.sources_list.currentItem()
        if item is None:
            return
        kind, src = item.data(Qt.UserRole)
        if kind == "point":
            self.problem.point_sources.remove(src)
        elif kind == "volume":
            self.problem.volume_sources.remove(src)
        self._refresh_sources_list()
        # Пересоздадим маркеры заново (проще, чем удалять конкретный).
        self.viz.clear_source_markers()
        for ps in self.problem.point_sources:
            if ps.node_idx < self.problem.nodes.shape[0]:
                nx, ny, nz = self.problem.nodes[ps.node_idx]
                self.viz.add_source_marker(float(nx), float(ny), float(nz),
                                           color="#ffd24a")
        for vs in self.problem.volume_sources:
            if vs.shape == VOLSRC_SPHERE:
                cx, cy, cz, _r = vs.params[:4]
                self.viz.add_source_marker(float(cx), float(cy), float(cz),
                                           color="#ff7b3a")

    def _on_clear_sources(self) -> None:
        self.problem.point_sources.clear()
        self.problem.volume_sources.clear()
        self._refresh_sources_list()
        self.viz.clear_source_markers()

    # ---- Расчёт ---------------------------------------------------------
    def _on_run(self) -> None:
        self._sync_to_problem()
        types = {bc.type for bc in self.problem.bcs.values()}
        if BC_DIRICHLET not in types and BC_ROBIN not in types:
            QMessageBox.warning(
                self, "Неполные условия",
                "Хотя бы на одной грани должно быть задано условие Дирихле "
                "или Робена — иначе задача определена не однозначно."
            )
            return
        if self.problem.nodes is None:
            QMessageBox.information(self, "Нет сетки",
                                    "Сначала сгенерируйте сетку.")
            return

        self.run_button.setEnabled(False)
        self.progress.setVisible(True)
        self.result_label.setText("Идёт расчёт...")

        self._thread = QThread(self)
        self._worker = SolverWorker(self.problem,
                                     tol=self.settings.cg_tolerance,
                                     max_iter=self.settings.cg_max_iter,
                                     omp_threads=self.settings.omp_threads)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.result_label.setText)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_worker_finished(self, info: SolverInfo) -> None:
        self._refresh_view_from_problem(rebuild_geometry=False)
        self.plots_view.set_results(self.problem.nodes,
                                     self.problem.T, self.problem.flux)
        Tmin, Tmax = self.problem.temperature_range()
        msg = (f"Tmin = {Tmin:.2f} °C    Tmax = {Tmax:.2f} °C    "
               f"итераций: {info.iterations}    невязка: {info.residual:.2e}    "
               f"время: {info.time_seconds*1000:.1f} мс    "
               f"{'сошёлся' if info.converged else 'НЕ сошёлся'}")
        self.result_label.setText(msg)
        self.btn_vtu.setEnabled(True); self.btn_csv.setEnabled(True); self.btn_report.setEnabled(True)
        if self.settings.auto_save_calculation:
            self.calc_view.add_record(self.problem,
                                       title=f"λ={self.problem.lambda_:g}, "
                                             f"{Tmin:.1f}..{Tmax:.1f} °C")

    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Ошибка расчёта", msg)
        self.result_label.setText("Расчёт прерван.")

    def _on_thread_finished(self) -> None:
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        if self._thread is not None:
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None

    def _on_calc_selected(self, record: CalculationRecord) -> None:
        """Пользователь выбрал запись из истории — загружаем её результаты."""
        self.problem = record.problem
        self.plots_view.set_results(self.problem.nodes,
                                     self.problem.T, self.problem.flux)
        self._refresh_view_from_problem(rebuild_geometry=True)
        info = self.problem.info
        if info is not None:
            Tmin, Tmax = self.problem.temperature_range()
            self.result_label.setText(
                f"[из истории] Tmin = {Tmin:.2f} °C, Tmax = {Tmax:.2f} °C, "
                f"итераций: {info.iterations}"
            )
        self.btn_vtu.setEnabled(True); self.btn_csv.setEnabled(True); self.btn_report.setEnabled(True)

    # ---- Экспорт -----------------------------------------------------------
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
            self, "Сохранить отчёт", "report.txt", "Текстовый файл (*.txt)")
        if not path: return
        try:
            export_report(self.problem, path)
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    # ---- Меню Настройки/Справка -------------------------------------------
    def _on_open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec_() == QDialog.Accepted:
            dlg.save_to(self.settings)
            # Обновим режим рендера по умолчанию.
            idx = self.mode_combo.findData(self.settings.render_mode)
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)
            self.iso_spin.setValue(self.settings.iso_count)

    def _on_show_help(self) -> None:
        HelpDialog(self).exec_()

    def _on_show_about(self) -> None:
        QMessageBox.about(
            self, "О программе",
            "<h3>Программный комплекс МКЭ</h3>"
            "<p>Расчёт стационарной теплопроводности методом конечных "
            "элементов на тетраэдрах P1.</p>"
            "<p>Версия 1.2 · Двухуровневая архитектура: C++ ядро + "
            "Python-управляющий слой + GUI на PyQt5.</p>"
            f"<p>3D backend: <b>{self.viz.backend_name}</b></p>"
        )

    def _update_3d_backend_status(self) -> None:
        if not HAS_PYVISTA:
            self.statusBar().showMessage(
                "Используется matplotlib 3D-вид (для лучшей визуализации "
                "установите: pip install pyvista pyvistaqt).", 8000)


# =============================================================================
# Точка входа.
# =============================================================================

def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    palette = QPalette()
    palette.setColor(QPalette.Window,        QColor("#1f2228"))
    palette.setColor(QPalette.WindowText,    QColor("#dcdee2"))
    palette.setColor(QPalette.Base,          QColor("#1a1d22"))
    palette.setColor(QPalette.AlternateBase, QColor("#2a2e36"))
    palette.setColor(QPalette.Text,          QColor("#dcdee2"))
    palette.setColor(QPalette.Button,        QColor("#3c4049"))
    palette.setColor(QPalette.ButtonText,    QColor("#dcdee2"))
    palette.setColor(QPalette.Highlight,     QColor("#7a6cf0"))
    palette.setColor(QPalette.HighlightedText, QColor("white"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
