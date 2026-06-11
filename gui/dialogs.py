# -*- coding: utf-8 -*-
"""
gui.dialogs — диалоги «Настройки» и «Справка», а также диалоги добавления
локальных источников.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem,
                             QMessageBox, QPushButton, QScrollArea, QSlider,
                             QSpinBox, QStackedWidget, QTabWidget,
                             QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from .theme import current_theme

from fem3d import BoundaryCondition, BoxGeometry


# =============================================================================
# Параметры (настройки) приложения. Хранятся в одном объекте, который
# передаётся между диалогом и главным окном.
# =============================================================================

class AppSettings:
    """Изменяемые параметры приложения."""

    def __init__(self) -> None:
        # Решатель.
        self.cg_tolerance = 1e-8
        self.cg_max_iter = 5000
        self.omp_threads = 0  # 0 = по умолчанию (все доступные)

        # Визуализация.
        self.render_mode = "surface"   # surface / volume / isosurface / wireframe
        self.iso_count = 7
        self.show_axes = True
        self.show_orientation = True

        # Тема приложения: "dark" / "light" / "sepia".
        self.theme = "dark"

        # Прочее.
        self.auto_save_calculation = True


class SettingsDialog(QDialog):
    """Диалог настроек."""

    def __init__(self, settings: AppSettings,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(420)
        self._settings = settings

        outer = QVBoxLayout(self)

        tabs = QTabWidget()
        outer.addWidget(tabs)

        # ----- Вкладка «Решатель» --------------------------------------------
        solver_w = QWidget()
        sf = QFormLayout(solver_w)

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1e-14, 1e-2)
        self.tol_spin.setDecimals(12)
        self.tol_spin.setSingleStep(1e-9)
        self.tol_spin.setValue(settings.cg_tolerance)
        sf.addRow("Относительная норма невязки:", self.tol_spin)

        self.maxiter_spin = QSpinBox()
        self.maxiter_spin.setRange(10, 100000)
        self.maxiter_spin.setValue(settings.cg_max_iter)
        sf.addRow("Максимум итераций CG:", self.maxiter_spin)

        self.omp_spin = QSpinBox()
        self.omp_spin.setRange(0, 256)
        self.omp_spin.setValue(settings.omp_threads)
        self.omp_spin.setSpecialValueText("по умолчанию")
        sf.addRow("Потоки OpenMP:", self.omp_spin)

        tabs.addTab(solver_w, "Решатель")

        # ----- Вкладка «Визуализация» ---------------------------------------
        viz_w = QWidget()
        vf = QFormLayout(viz_w)

        self.mode_combo = QComboBox()
        for label, value in [
            ("Поверхность", "surface"),
            ("Объёмный рендер", "volume"),
            ("Изоповерхности", "isosurface"),
            ("Каркас", "wireframe"),
        ]:
            self.mode_combo.addItem(label, value)
        idx = self.mode_combo.findData(settings.render_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        vf.addRow("Режим рендера по умолчанию:", self.mode_combo)

        self.iso_spin = QSpinBox()
        self.iso_spin.setRange(2, 20)
        self.iso_spin.setValue(settings.iso_count)
        vf.addRow("Число изоповерхностей:", self.iso_spin)

        self.axes_check = QCheckBox()
        self.axes_check.setChecked(settings.show_axes)
        vf.addRow("Показывать оси:", self.axes_check)

        self.orient_check = QCheckBox()
        self.orient_check.setChecked(settings.show_orientation)
        vf.addRow("Индикатор ориентации:", self.orient_check)

        tabs.addTab(viz_w, "Визуализация")

        # ----- Вкладка «Тема» -----------------------------------------------
        from .theme import THEMES
        theme_w = QWidget()
        tf = QFormLayout(theme_w)
        self.theme_combo = QComboBox()
        for key, palette in THEMES.items():
            self.theme_combo.addItem(palette.title, key)
        idx = self.theme_combo.findData(settings.theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        tf.addRow("Цветовая схема:", self.theme_combo)

        theme_hint = QLabel(
            "<b>Тёмная</b> — стандартная контрастная схема.<br>"
            "<b>Светлая</b> — белый фон, для дневной работы.<br>"
            "<b>Бежевая</b> — мягкая тёплая палитра, снижает нагрузку "
            "на глаза при длительной работе."
        )
        theme_hint.setWordWrap(True)
        theme_hint.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        tf.addRow(theme_hint)
        tabs.addTab(theme_w, "Тема")

        # ----- Вкладка «Прочее» ---------------------------------------------
        misc_w = QWidget()
        mf = QFormLayout(misc_w)
        self.auto_save_check = QCheckBox()
        self.auto_save_check.setChecked(settings.auto_save_calculation)
        mf.addRow("Автоматически сохранять в историю расчётов:",
                  self.auto_save_check)
        tabs.addTab(misc_w, "Прочее")

        # ----- Кнопки -------------------------------------------------------
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def save_to(self, settings: AppSettings) -> None:
        settings.cg_tolerance = float(self.tol_spin.value())
        settings.cg_max_iter = int(self.maxiter_spin.value())
        settings.omp_threads = int(self.omp_spin.value())
        settings.render_mode = self.mode_combo.currentData()
        settings.iso_count = int(self.iso_spin.value())
        settings.show_axes = self.axes_check.isChecked()
        settings.show_orientation = self.orient_check.isChecked()
        settings.theme = self.theme_combo.currentData()
        settings.auto_save_calculation = self.auto_save_check.isChecked()


# =============================================================================
# Диалог справки.
# =============================================================================

HELP_HTML = """
<h2>Программный комплекс МКЭ для трёхмерной теплопроводности</h2>
<p style='color:#9aa0a6'>Версия 1.2 · Метод конечных элементов на тетраэдрах P1</p>

<h3>Краткое назначение</h3>
<p>Программа решает <i>стационарное</i> уравнение теплопроводности в трёхмерных
областях. Поддерживаются граничные условия трёх типов: Дирихле (заданная
температура), Нейман (заданный поток или изоляция) и Робен (конвективный
теплообмен), а также объёмные и локальные источники тепла.</p>

<h3>Основные шаги работы</h3>
<ol>
<li><b>Геометрия</b>. На левой панели выберите тип: параметрический параллелепипед
    или импорт из MSH/VTU/STL/STEP. Задайте размеры и плотность сетки.</li>
<li><b>Материал</b>. Выберите из справочника или введите коэффициент
    теплопроводности λ и плотность объёмного источника Q вручную.</li>
<li><b>Локальные источники</b> (опционально). Точечные привязываются к
    ближайшему узлу; объёмные задаются как сферическая подобласть с центром,
    радиусом и плотностью мощности.</li>
<li><b>Граничные условия</b> на правой панели — на каждой из 6 граней.
    Для импортированной геометрии все наружные грани относятся к группе 0.</li>
<li><b>Расчёт</b>. Нажмите кнопку «Запустить расчёт» в нижней панели.</li>
<li><b>Результаты</b>. Переключайтесь между вкладками «3D-вид», «Графики»
    и «Расчёты». Экспорт — в форматы VTU (ParaView), CSV и текстового отчёта.</li>
</ol>

<h3>Управление 3D-видом</h3>
<ul>
<li><b>ЛКМ</b> — вращение модели;</li>
<li><b>СКМ или Shift+ЛКМ</b> — панорамирование;</li>
<li><b>Колесо мыши</b> — масштабирование;</li>
<li>Кнопка <b>«Сбросить вид»</b> в правой части — установить камеру по габариту;</li>
<li><b>Режим «Узнать T в точке»</b>: клик по узлу — внизу появятся координаты и температура;</li>
<li><b>Режим «Поставить источник»</b>: клик в любой точке — открывается диалог
    параметров точечного источника, привязанного к ближайшему узлу.</li>
</ul>

<h3>Режимы отображения</h3>
<ul>
<li><b>Поверхность</b> — цвет температуры на наружной поверхности тела;</li>
<li><b>Объёмный рендер</b> — полупрозрачное температурное поле во всём объёме
    (рекомендуется при наличии PyVista);</li>
<li><b>Изоповерхности</b> — линии равной температуры;</li>
<li><b>Каркас</b> — рёбра сетки без заливки;</li>
<li><b>Сечение</b> — карта температуры в плоскости, ползунок задаёт положение.</li>
</ul>

<h3>Граничные условия</h3>
<table border='1' cellpadding='4' style='border-collapse:collapse'>
<tr><th>Тип</th><th>Параметры</th><th>Физика</th></tr>
<tr><td>Дирихле</td><td>T₀, °C</td><td>заданная температура поверхности</td></tr>
<tr><td>Нейман</td><td>q, Вт/м²</td><td>заданный тепловой поток (q=0 → изоляция)</td></tr>
<tr><td>Робен</td><td>α, T∞</td><td>конвекция: −λ ∂T/∂n = α(T − T∞)</td></tr>
</table>

<h3>Поддерживаемые форматы импорта</h3>
<ul>
<li><b>.msh</b> — Gmsh, объёмные тетраэдры (требует <code>meshio</code>);</li>
<li><b>.vtu, .vtk</b> — VTK Unstructured Grid с тетраэдрами;</li>
<li><b>.stl</b> — поверхностная сетка (требует <code>gmsh</code> для тетраэдризации);</li>
<li><b>.step, .stp</b> — CAD-геометрия (требует <code>gmsh</code>).</li>
</ul>

<h3>Корректность задачи</h3>
<p>Для уникального решения хотя бы на одной грани должно быть задано условие
Дирихле или Робена — иначе система имеет ненулевое ядро (любая константа
является решением).</p>

<h3>Верификация</h3>
<p>Программа поставляется с четырьмя верификационными тестами (T1–T4),
запускаемыми командой <code>python -m fem3d.verify</code>. Тесты включают
аналитические задачи и проверку эмпирического порядка сходимости в L²
(теоретическое значение для P1 — ровно 2).</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.resize(720, 600)
        outer = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(HELP_HTML)
        from .theme import current_theme as _ct_help
        _thh = _ct_help()
        browser.setStyleSheet(
            f"background-color: {_thh.input_bg}; color: {_thh.text};")
        browser.setOpenExternalLinks(True)
        outer.addWidget(browser)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        outer.addWidget(btns)


# =============================================================================
# Диалоги добавления источников (раньше были в main_gui.py).
# =============================================================================

class PointSourceDialog(QDialog):
    """Диалог ввода параметров точечного источника."""

    def __init__(self, geom: BoxGeometry,
                 prefill: Optional[Tuple[float, float, float]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Точечный источник")
        self.setMinimumWidth(320)
        form = QFormLayout(self)

        # Координаты — либо центр габарита, либо предзаполненные (например,
        # из клика мыши в 3D-виде).
        cx = prefill[0] if prefill else geom.Lx / 2
        cy = prefill[1] if prefill else geom.Ly / 2
        cz = prefill[2] if prefill else geom.Lz / 2

        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-1e5, 1e5); self.x_spin.setDecimals(5); self.x_spin.setValue(cx)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-1e5, 1e5); self.y_spin.setDecimals(5); self.y_spin.setValue(cy)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-1e5, 1e5); self.z_spin.setDecimals(5); self.z_spin.setValue(cz)
        form.addRow("x, м:", self.x_spin)
        form.addRow("y, м:", self.y_spin)
        form.addRow("z, м:", self.z_spin)

        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(-1.0e6, 1.0e6); self.p_spin.setDecimals(2)
        self.p_spin.setValue(10.0)
        self.p_spin.setSuffix(" Вт")
        self.p_spin.setToolTip(
            "Полная тепловая мощность точечного источника.\n"
            "Типичные значения [Вт]:\n"
            "  Резистор/диод ≈ 0.1 – 5\n"
            "  Светодиод ≈ 0.5 – 50\n"
            "  CPU/GPU ≈ 5 – 300\n"
            "  Нагревательный элемент ≈ 100 – 5000")
        form.addRow("Мощность P:", self.p_spin)

        hint = QLabel("Источник будет привязан к ближайшему узлу сетки.\n"
                      "P > 0 — нагрев, P < 0 — отвод тепла.")
        hint.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        hint.setWordWrap(True)
        form.addRow(hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def values(self) -> Tuple[float, float, float, float]:
        return (self.x_spin.value(), self.y_spin.value(),
                self.z_spin.value(), self.p_spin.value())


class VolumeSourceDialog(QDialog):
    """Диалог ввода параметров объёмного источника (сфера)."""

    def __init__(self, geom: BoxGeometry,
                 prefill: Optional[Tuple[float, float, float]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Объёмный источник (сфера)")
        self.setMinimumWidth(320)
        form = QFormLayout(self)

        cx = prefill[0] if prefill else geom.Lx / 2
        cy = prefill[1] if prefill else geom.Ly / 2
        cz = prefill[2] if prefill else geom.Lz / 2

        self.cx_spin = QDoubleSpinBox()
        self.cx_spin.setRange(-1e5, 1e5); self.cx_spin.setDecimals(5); self.cx_spin.setValue(cx)
        self.cy_spin = QDoubleSpinBox()
        self.cy_spin.setRange(-1e5, 1e5); self.cy_spin.setDecimals(5); self.cy_spin.setValue(cy)
        self.cz_spin = QDoubleSpinBox()
        self.cz_spin.setRange(-1e5, 1e5); self.cz_spin.setDecimals(5); self.cz_spin.setValue(cz)
        form.addRow("Центр x, м:", self.cx_spin)
        form.addRow("Центр y, м:", self.cy_spin)
        form.addRow("Центр z, м:", self.cz_spin)

        self.r_spin = QDoubleSpinBox()
        self.r_spin.setRange(1e-5, 1e5); self.r_spin.setDecimals(5)
        self.r_spin.setValue(min(geom.Lx, geom.Ly, geom.Lz) / 5)
        form.addRow("Радиус, м:", self.r_spin)

        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(-1.0e10, 1.0e10); self.q_spin.setDecimals(0)
        self.q_spin.setValue(1.0e6)
        self.q_spin.setSuffix(" Вт/м³")
        self.q_spin.setToolTip(
            "Объёмная плотность тепловой мощности в сфере.\n"
            "Типичные значения [Вт/м³]:\n"
            "  Тепло от батареи отопления ≈ 10³ – 10⁴\n"
            "  CPU ≈ 10⁶ – 10⁷\n"
            "  Тепловыделение от химической реакции ≈ 10⁷ – 10⁹")
        form.addRow("Плотность Q₀:", self.q_spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def values(self) -> Tuple[float, float, float, float, float]:
        return (self.cx_spin.value(), self.cy_spin.value(),
                self.cz_spin.value(), self.r_spin.value(),
                self.q_spin.value())


# =============================================================================
# Диалог редактирования одного региона материала.
# =============================================================================

from PyQt5.QtWidgets import (QColorDialog, QHeaderView, QPushButton,
                              QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QColor


class MaterialRegionDialog(QDialog):
    """Диалог параметров одного региона материала: λ, Q, форма (box/sphere)
    с её размерами, цвет."""

    DEFAULT_COLORS = ["#f0a030", "#3aa55a", "#3a78d0", "#d05050",
                       "#a06cf0", "#e0c040", "#50bfa0", "#c46868"]

    def __init__(self, geom: BoxGeometry,
                 region=None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Регион материала")
        self.setMinimumWidth(400)
        from fem3d import REGION_BOX, REGION_SPHERE
        self._REGION_BOX = REGION_BOX
        self._REGION_SPHERE = REGION_SPHERE

        outer = QVBoxLayout(self)
        form = QFormLayout()

        from PyQt5.QtWidgets import QLineEdit
        self.name_edit = QLineEdit()
        self.name_edit.setText(region.name if region else "Регион")
        form.addRow("Имя:", self.name_edit)

        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(1e-4, 1e5)
        self.lambda_spin.setDecimals(4)
        self.lambda_spin.setValue(region.lambda_ if region else 237.0)
        self.lambda_spin.setSuffix(" Вт/(м·К)")
        form.addRow("λ:", self.lambda_spin)

        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(-1e9, 1e9)
        self.q_spin.setDecimals(2)
        self.q_spin.setValue(region.Q if region else 0.0)
        self.q_spin.setSuffix(" Вт/м³")
        form.addRow("Q:", self.q_spin)

        # Тип формы.
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("Параллелепипед (box)", REGION_BOX)
        self.shape_combo.addItem("Сфера", REGION_SPHERE)
        if region and region.shape == REGION_SPHERE:
            self.shape_combo.setCurrentIndex(1)
        self.shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        form.addRow("Форма:", self.shape_combo)

        # Контейнер для параметров box.
        self.box_widget = QWidget()
        bg = QFormLayout(self.box_widget)
        bg.setContentsMargins(0, 0, 0, 0)
        self.box_xmin = QDoubleSpinBox(); self.box_xmin.setRange(-1e5, 1e5); self.box_xmin.setDecimals(5)
        self.box_xmax = QDoubleSpinBox(); self.box_xmax.setRange(-1e5, 1e5); self.box_xmax.setDecimals(5)
        self.box_ymin = QDoubleSpinBox(); self.box_ymin.setRange(-1e5, 1e5); self.box_ymin.setDecimals(5)
        self.box_ymax = QDoubleSpinBox(); self.box_ymax.setRange(-1e5, 1e5); self.box_ymax.setDecimals(5)
        self.box_zmin = QDoubleSpinBox(); self.box_zmin.setRange(-1e5, 1e5); self.box_zmin.setDecimals(5)
        self.box_zmax = QDoubleSpinBox(); self.box_zmax.setRange(-1e5, 1e5); self.box_zmax.setDecimals(5)
        bg.addRow("x min, max:", self._pair(self.box_xmin, self.box_xmax))
        bg.addRow("y min, max:", self._pair(self.box_ymin, self.box_ymax))
        bg.addRow("z min, max:", self._pair(self.box_zmin, self.box_zmax))
        # Заполнение значений.
        if region and region.shape == REGION_BOX and len(region.params) == 6:
            xmin, xmax, ymin, ymax, zmin, zmax = region.params
        else:
            xmin, xmax = 0.0, geom.Lx / 2
            ymin, ymax = 0.0, geom.Ly
            zmin, zmax = 0.0, geom.Lz
        self.box_xmin.setValue(xmin); self.box_xmax.setValue(xmax)
        self.box_ymin.setValue(ymin); self.box_ymax.setValue(ymax)
        self.box_zmin.setValue(zmin); self.box_zmax.setValue(zmax)
        form.addRow(self.box_widget)

        # Контейнер для параметров sphere.
        self.sph_widget = QWidget()
        sg = QFormLayout(self.sph_widget)
        sg.setContentsMargins(0, 0, 0, 0)
        self.sph_cx = QDoubleSpinBox(); self.sph_cx.setRange(-1e5, 1e5); self.sph_cx.setDecimals(5)
        self.sph_cy = QDoubleSpinBox(); self.sph_cy.setRange(-1e5, 1e5); self.sph_cy.setDecimals(5)
        self.sph_cz = QDoubleSpinBox(); self.sph_cz.setRange(-1e5, 1e5); self.sph_cz.setDecimals(5)
        self.sph_r = QDoubleSpinBox(); self.sph_r.setRange(1e-5, 1e5); self.sph_r.setDecimals(5)
        sg.addRow("Центр x:", self.sph_cx)
        sg.addRow("Центр y:", self.sph_cy)
        sg.addRow("Центр z:", self.sph_cz)
        sg.addRow("Радиус:", self.sph_r)
        if region and region.shape == REGION_SPHERE and len(region.params) == 4:
            cx, cy, cz, r = region.params
        else:
            cx, cy, cz = geom.Lx / 2, geom.Ly / 2, geom.Lz / 2
            r = min(geom.Lx, geom.Ly, geom.Lz) / 4
        self.sph_cx.setValue(cx); self.sph_cy.setValue(cy)
        self.sph_cz.setValue(cz); self.sph_r.setValue(r)
        form.addRow(self.sph_widget)

        # Цвет.
        self.color_btn = QPushButton()
        self._color = region.color if region else self.DEFAULT_COLORS[0]
        self._update_color_button()
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow("Цвет:", self.color_btn)

        outer.addLayout(form)

        hint = QLabel(
            "Регион применяется к тетраэдрам, центроиды которых попадают "
            "в указанную область. Если узлов в области нет — регион не "
            "будет иметь эффекта."
        )
        hint.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        self._on_shape_changed()

    def _pair(self, w1, w2):
        c = QWidget()
        h = QHBoxLayout(c)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w1); h.addWidget(w2)
        return c

    def _on_shape_changed(self):
        is_box = (self.shape_combo.currentData() == self._REGION_BOX)
        self.box_widget.setVisible(is_box)
        self.sph_widget.setVisible(not is_box)

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self, "Цвет региона")
        if c.isValid():
            self._color = c.name()
            self._update_color_button()

    def _update_color_button(self):
        self.color_btn.setStyleSheet(
            f"background-color: {self._color}; color: white; min-width: 80px;"
        )
        self.color_btn.setText(self._color)

    def to_region(self):
        """Возвращает MaterialRegion с введёнными параметрами."""
        from fem3d import MaterialRegion
        shape = self.shape_combo.currentData()
        if shape == self._REGION_BOX:
            params = (self.box_xmin.value(), self.box_xmax.value(),
                      self.box_ymin.value(), self.box_ymax.value(),
                      self.box_zmin.value(), self.box_zmax.value())
        else:
            params = (self.sph_cx.value(), self.sph_cy.value(),
                      self.sph_cz.value(), self.sph_r.value())
        return MaterialRegion(
            name=self.name_edit.text() or "Регион",
            lambda_=float(self.lambda_spin.value()),
            Q=float(self.q_spin.value()),
            shape=shape, params=params, color=self._color,
        )


# =============================================================================
# Диалог-редактор СПИСКА регионов: таблица + кнопки добавить/изменить/удалить.
# =============================================================================

class MaterialRegionsDialog(QDialog):
    """Редактор списка регионов материалов."""

    def __init__(self, regions: list, geom: BoxGeometry,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Регионы материалов")
        self.resize(700, 460)
        self._regions = [r for r in regions]  # копия списка
        self._geom = geom

        outer = QVBoxLayout(self)

        info = QLabel(
            "Регионы материалов позволяют задать <b>разные λ и Q в разных "
            "частях детали</b>. Регионы накладываются последовательно: "
            "тетраэдр получает материал последнего региона, который его "
            "захватил."
        )
        info.setStyleSheet(f"color: {current_theme().text_dim};")
        info.setWordWrap(True)
        outer.addWidget(info)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["", "Имя", "λ, Вт/(м·К)", "Q, Вт/м³", "Форма"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 28)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemDoubleClicked.connect(self._on_edit)
        outer.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("+ Добавить")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_edit = QPushButton("Изменить...")
        self.btn_edit.clicked.connect(self._on_edit)
        self.btn_remove = QPushButton("Удалить")
        self.btn_remove.clicked.connect(self._on_remove)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        bottom = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bottom.accepted.connect(self.accept)
        bottom.rejected.connect(self.reject)
        outer.addWidget(bottom)

        self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self._regions))
        for row, r in enumerate(self._regions):
            color_item = QTableWidgetItem("")
            color_item.setBackground(QColor(r.color))
            self.table.setItem(row, 0, color_item)
            self.table.setItem(row, 1, QTableWidgetItem(r.name))
            self.table.setItem(row, 2, QTableWidgetItem(f"{r.lambda_:g}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{r.Q:g}"))
            self.table.setItem(row, 4, QTableWidgetItem(r.shape))

    def _on_add(self):
        dlg = MaterialRegionDialog(self._geom, region=None, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._regions.append(dlg.to_region())
            self._refresh_table()

    def _on_edit(self):
        row = self.table.currentRow()
        if row < 0:
            return
        dlg = MaterialRegionDialog(
            self._geom, region=self._regions[row], parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._regions[row] = dlg.to_region()
            self._refresh_table()

    def _on_remove(self):
        row = self.table.currentRow()
        if row < 0:
            return
        del self._regions[row]
        self._refresh_table()

    def regions(self):
        return list(self._regions)


# =============================================================================
# Компактный диалог редактирования геометрии — заменяет длинную панель.
# =============================================================================

class GeometryDialog(QDialog):
    """Диалог компактного выбора геометрии: тип (box/shape/import) + параметры."""

    def __init__(self, problem,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Геометрия")
        self.resize(420, 380)
        self._problem = problem
        self._import_path: Optional[str] = None

        from fem3d import PRESETS, SHAPE_PRESETS

        outer = QVBoxLayout(self)
        form = QFormLayout()

        self.type_combo = QComboBox()
        self.type_combo.addItem("Параллелепипед", "box")
        self.type_combo.addItem("Сложная фигура (пресет)", "shape")
        self.type_combo.addItem("Импорт сетки", "import")
        self.type_combo.currentIndexChanged.connect(self._on_type)
        form.addRow("Тип:", self.type_combo)

        # ----- Box -----
        self.box_widget = QWidget()
        bw = QFormLayout(self.box_widget)
        bw.setContentsMargins(0, 0, 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("— Произвольно —", None)
        for ps in PRESETS:
            self.preset_combo.addItem(ps.label, ps)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        bw.addRow("Пресет:", self.preset_combo)
        self.size_x = QDoubleSpinBox(); self.size_x.setRange(1e-4, 100); self.size_x.setDecimals(4)
        self.size_y = QDoubleSpinBox(); self.size_y.setRange(1e-4, 100); self.size_y.setDecimals(4)
        self.size_z = QDoubleSpinBox(); self.size_z.setRange(1e-4, 100); self.size_z.setDecimals(4)
        self.n_x = QSpinBox(); self.n_x.setRange(2, 200)
        self.n_y = QSpinBox(); self.n_y.setRange(2, 200)
        self.n_z = QSpinBox(); self.n_z.setRange(2, 200)
        size_w = QWidget(); sl = QHBoxLayout(size_w); sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("X:")); sl.addWidget(self.size_x)
        sl.addWidget(QLabel("Y:")); sl.addWidget(self.size_y)
        sl.addWidget(QLabel("Z:")); sl.addWidget(self.size_z)
        bw.addRow("Размеры, м:", size_w)
        n_w = QWidget(); nl = QHBoxLayout(n_w); nl.setContentsMargins(0, 0, 0, 0)
        nl.addWidget(QLabel("nx:")); nl.addWidget(self.n_x)
        nl.addWidget(QLabel("ny:")); nl.addWidget(self.n_y)
        nl.addWidget(QLabel("nz:")); nl.addWidget(self.n_z)
        bw.addRow("Разбиение:", n_w)

        g = problem.geometry
        self.size_x.setValue(g.Lx); self.size_y.setValue(g.Ly); self.size_z.setValue(g.Lz)
        self.n_x.setValue(g.nx); self.n_y.setValue(g.ny); self.n_z.setValue(g.nz)
        form.addRow(self.box_widget)

        # ----- Shape -----
        self.shape_widget = QWidget()
        sw = QFormLayout(self.shape_widget)
        sw.setContentsMargins(0, 0, 0, 0)
        self.shape_combo = QComboBox()
        for sps in SHAPE_PRESETS:
            self.shape_combo.addItem(sps.label, sps)
        sw.addRow("Фигура:", self.shape_combo)
        self.shape_desc = QLabel("")
        self.shape_desc.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        self.shape_desc.setWordWrap(True)
        sw.addRow(self.shape_desc)
        self.shape_combo.currentIndexChanged.connect(self._on_shape)
        if SHAPE_PRESETS:
            self.shape_desc.setText(SHAPE_PRESETS[0].description)

        # Слайдер плотности сетки для фигур.
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setMinimum(5)   # 0.5
        self.density_slider.setMaximum(30)  # 3.0
        self.density_slider.setValue(10)    # 1.0
        self.density_slider.setTickInterval(5)
        self.density_label = QLabel("Плотность сетки: ×1.0 (стандарт)")
        self.density_label.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        def _upd_density(v):
            d = v / 10.0
            tag = ("грубая" if d < 0.8 else
                   "стандарт" if d <= 1.2 else
                   "детальная" if d <= 2.0 else "очень детальная")
            self.density_label.setText(f"Плотность сетки: ×{d:.1f} ({tag})")
        self.density_slider.valueChanged.connect(_upd_density)
        sw.addRow(self.density_label)
        sw.addRow(self.density_slider)
        form.addRow(self.shape_widget)

        # ----- Import -----
        self.import_widget = QWidget()
        iw = QFormLayout(self.import_widget)
        iw.setContentsMargins(0, 0, 0, 0)
        self.import_label = QLabel("<i>Файл не выбран</i>")
        self.import_label.setStyleSheet(f"color: {current_theme().text_dim};")
        self.import_label.setWordWrap(True)
        iw.addRow(self.import_label)
        self.btn_browse = QPushButton("Выбрать файл (MSH/VTU/STL/STEP)...")
        self.btn_browse.clicked.connect(self._on_browse)
        iw.addRow(self.btn_browse)
        form.addRow(self.import_widget)

        outer.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        self.shape_widget.setVisible(False)
        self.import_widget.setVisible(False)

    def _on_type(self, _idx):
        kind = self.type_combo.currentData()
        self.box_widget.setVisible(kind == "box")
        self.shape_widget.setVisible(kind == "shape")
        self.import_widget.setVisible(kind == "import")

    def _on_preset(self, _idx):
        ps = self.preset_combo.currentData()
        if ps is None:
            return
        self.size_x.setValue(ps.Lx); self.size_y.setValue(ps.Ly); self.size_z.setValue(ps.Lz)
        self.n_x.setValue(ps.nx); self.n_y.setValue(ps.ny); self.n_z.setValue(ps.nz)

    def _on_shape(self, _idx):
        sps = self.shape_combo.currentData()
        if sps is not None:
            self.shape_desc.setText(sps.description)

    def _on_browse(self):
        from fem3d import SUPPORTED_IMPORT_EXTENSIONS
        exts = " ".join("*" + e for e in SUPPORTED_IMPORT_EXTENSIONS)
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл сетки", "",
            f"Сетки ({exts});;Все файлы (*.*)"
        )
        if path:
            self._import_path = path
            import os
            self.import_label.setText(f"<b>{os.path.basename(path)}</b>")

    def result_kind(self) -> str:
        return self.type_combo.currentData()

    def box_params(self):
        return (self.size_x.value(), self.size_y.value(), self.size_z.value(),
                int(self.n_x.value()), int(self.n_y.value()), int(self.n_z.value()))

    def shape_preset(self):
        return self.shape_combo.currentData()

    def shape_density(self) -> float:
        return self.density_slider.value() / 10.0

    def import_path(self):
        return self._import_path


# =============================================================================
# Компактный диалог редактирования материала (один глобальный).
# =============================================================================

class MaterialDialog(QDialog):
    """Глобальный материал — выбор из библиотеки (с категориями) или ручной ввод.
    Открывает MaterialLibraryDialog для создания/удаления пользовательских материалов."""

    def __init__(self, problem, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Материал")
        self.setMinimumWidth(440)
        from fem3d import all_materials, Material

        outer = QVBoxLayout(self)
        form = QFormLayout()

        # Категория.
        self._all_mats = all_materials()
        categories = list(dict.fromkeys(m.category for m in self._all_mats))
        self.category_combo = QComboBox()
        self.category_combo.addItem("— Все —", None)
        for cat in categories:
            self.category_combo.addItem(cat, cat)
        self.category_combo.currentIndexChanged.connect(self._on_category)
        form.addRow("Категория:", self.category_combo)

        # Материал.
        self.material_combo = QComboBox()
        self._selected_material = None
        self._refill_material_combo(None)
        self.material_combo.currentIndexChanged.connect(self._on_select)
        form.addRow("Материал:", self.material_combo)

        # Параметры.
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(1e-4, 1e5); self.lambda_spin.setDecimals(4)
        self.lambda_spin.setValue(problem.lambda_)
        self.lambda_spin.setSuffix(" Вт/(м·К)")
        self.lambda_spin.setToolTip(
            "Коэффициент теплопроводности.\n"
            "Типичные значения [Вт/(м·К)]:\n"
            "  Серебро 429, медь 401, алюминий 237, сталь 55,\n"
            "  стекло 1.0, бетон 1.5, дерево 0.15, минвата 0.045")
        form.addRow("λ (теплопроводность):", self.lambda_spin)

        self.rho_spin = QDoubleSpinBox()
        self.rho_spin.setRange(0, 1e6); self.rho_spin.setDecimals(1)
        self.rho_spin.setSuffix(" кг/м³")
        self.rho_spin.setToolTip(
            "Плотность материала.\n"
            "Типичные значения [кг/м³]:\n"
            "  Сталь 7850, алюминий 2700, стекло 2500,\n"
            "  бетон 2400, дерево 720, воздух 1.2")
        form.addRow("ρ (плотность):", self.rho_spin)

        self.cp_spin = QDoubleSpinBox()
        self.cp_spin.setRange(0, 1e5); self.cp_spin.setDecimals(1)
        self.cp_spin.setSuffix(" Дж/(кг·К)")
        self.cp_spin.setToolTip(
            "Удельная теплоёмкость.\n"
            "Типичные значения [Дж/(кг·К)]:\n"
            "  Вода 4186, дерево 2400, бетон 880,\n"
            "  алюминий 900, сталь 490, медь 385")
        form.addRow("c_p (теплоёмкость):", self.cp_spin)

        self.emissivity_spin = QDoubleSpinBox()
        self.emissivity_spin.setRange(0.0, 1.0); self.emissivity_spin.setDecimals(2)
        self.emissivity_spin.setSingleStep(0.05)
        self.emissivity_spin.setToolTip(
            "Степень черноты ε ∈ [0, 1].\n"
            "  Полированный металл ≈ 0.02 – 0.10\n"
            "  Окисленный металл ≈ 0.20 – 0.80\n"
            "  Стекло, керамика, краска ≈ 0.85 – 0.95\n"
            "Используется для излучения и теплообмена.")
        form.addRow("ε (степень черноты):", self.emissivity_spin)

        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(-1e9, 1e9); self.q_spin.setDecimals(2)
        self.q_spin.setValue(problem.Q)
        self.q_spin.setSuffix(" Вт/м³")
        self.q_spin.setToolTip(
            "Объёмная плотность мощности тепловыделения.\n"
            "Например, для CPU ~ 10⁶ – 10⁷ Вт/м³.\n"
            "Положительное значение — тело греется.")
        form.addRow("Q (объёмный источник):", self.q_spin)

        outer.addLayout(form)

        # Кнопка библиотеки.
        from PyQt5.QtWidgets import QPushButton
        lib_row = QHBoxLayout()
        self.btn_library = QPushButton("📚 Управление библиотекой материалов…")
        self.btn_library.clicked.connect(self._on_open_library)
        lib_row.addWidget(self.btn_library)
        outer.addLayout(lib_row)

        hint = QLabel(
            "Параметры ρ, c_p, ε используются для расчёта тепловой массы "
            "(нестационар), излучения и более полного баланса. Для статического "
            "расчёта обязательно только λ.<br><br>"
            "Чтобы задать <b>разные материалы в разных частях</b> детали, "
            "используйте «Регионы материалов»."
        )
        hint.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def _refill_material_combo(self, category):
        self.material_combo.blockSignals(True)
        self.material_combo.clear()
        self.material_combo.addItem("— Произвольный —", None)
        for m in self._all_mats:
            if category is not None and m.category != category:
                continue
            self.material_combo.addItem(
                f"{m.name}  (λ = {m.lambda_:g})", m)
        self.material_combo.blockSignals(False)

    def _on_category(self, _idx):
        cat = self.category_combo.currentData()
        self._refill_material_combo(cat)

    def _on_select(self, _idx):
        from fem3d import Material
        m = self.material_combo.currentData()
        if isinstance(m, Material):
            # Сохраняем выбранный объект, чтобы сохранить имя, категорию,
            # анизотропию и т.п. при ОК.
            self._selected_material = m
            # Изотропная часть.
            self.lambda_spin.setValue(m.effective_lambda())
            self.rho_spin.setValue(m.rho)
            self.cp_spin.setValue(m.cp)
            self.emissivity_spin.setValue(m.emissivity)
        else:
            self._selected_material = None

    def _on_open_library(self):
        dlg = MaterialLibraryDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            # Перезагружаем список — могли добавиться пользовательские.
            from fem3d import all_materials
            self._all_mats = all_materials()
            self._refill_material_combo(self.category_combo.currentData())

    def values(self) -> Tuple[float, float]:
        """Совместимость: (λ, Q)."""
        return float(self.lambda_spin.value()), float(self.q_spin.value())

    def selected_material(self):
        """Полный объект Material с актуальными значениями полей.

        Если пользователь не выбирал материал из библиотеки, создаёт
        новый Material из значений спинбоксов («Произвольный»).
        Используется в _on_edit_material() для сохранения ВСЕХ свойств
        в Problem (включая ρ, c_p, ε для нестационарного режима).
        """
        from fem3d import Material
        # Базируемся на выбранном из библиотеки (для сохранения name и категории),
        # либо создаём новый.
        sel = getattr(self, "_selected_material", None)
        if sel is None:
            return Material(
                name="Произвольный",
                category="Пользовательский",
                lambda_=float(self.lambda_spin.value()),
                rho=float(self.rho_spin.value()),
                cp=float(self.cp_spin.value()),
                emissivity=float(self.emissivity_spin.value()),
            )
        # Возвращаем копию с актуальными значениями (вдруг отредактировал
        # после выбора).
        return Material(
            name=sel.name,
            category=sel.category,
            lambda_=float(self.lambda_spin.value()),
            rho=float(self.rho_spin.value()),
            cp=float(self.cp_spin.value()),
            emissivity=float(self.emissivity_spin.value()),
            is_anisotropic=getattr(sel, "is_anisotropic", False),
            lambda_x=getattr(sel, "lambda_x", 0.0),
            lambda_y=getattr(sel, "lambda_y", 0.0),
            lambda_z=getattr(sel, "lambda_z", 0.0),
        )


class MaterialLibraryDialog(QDialog):
    """Управление библиотекой материалов: добавить / редактировать / удалить
    пользовательские материалы. Встроенные нельзя редактировать."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Библиотека материалов")
        self.resize(780, 480)

        outer = QVBoxLayout(self)

        info = QLabel(
            "Встроенные материалы помечены 🔒 (изменить нельзя). "
            "Пользовательские можно создать, изменить или удалить — они "
            "сохраняются в ~/.fem_heat3d_user_materials.json")
        info.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        info.setWordWrap(True)
        outer.addWidget(info)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["", "Имя", "λ", "ρ", "c_p", "ε", "Категория"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 30)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        outer.addWidget(self.table, 1)

        row = QHBoxLayout()
        from PyQt5.QtWidgets import QPushButton
        self.btn_add = QPushButton("+ Добавить материал")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_edit = QPushButton("Изменить...")
        self.btn_edit.clicked.connect(self._on_edit)
        self.btn_remove = QPushButton("Удалить")
        self.btn_remove.clicked.connect(self._on_remove)
        row.addWidget(self.btn_add); row.addWidget(self.btn_edit)
        row.addWidget(self.btn_remove); row.addStretch(1)
        outer.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        btns.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        outer.addWidget(btns)

        self._refresh()

    def _refresh(self):
        from fem3d import MATERIALS, load_user_materials
        builtin = list(MATERIALS)
        user = load_user_materials()
        self._builtin_count = len(builtin)
        all_mats = builtin + user
        self.table.setRowCount(len(all_mats))
        for r, m in enumerate(all_mats):
            lock = "🔒" if r < self._builtin_count else "✏"
            self.table.setItem(r, 0, QTableWidgetItem(lock))
            self.table.setItem(r, 1, QTableWidgetItem(m.name))
            self.table.setItem(r, 2, QTableWidgetItem(f"{m.lambda_:g}"))
            self.table.setItem(r, 3, QTableWidgetItem(f"{m.rho:g}"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{m.cp:g}"))
            self.table.setItem(r, 5, QTableWidgetItem(f"{m.emissivity:.2f}"))
            self.table.setItem(r, 6, QTableWidgetItem(m.category))

    def _on_add(self):
        dlg = MaterialEditorDialog(material=None, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._save_user_material(dlg.material(), add=True)
            self._refresh()

    def _on_edit(self):
        row = self.table.currentRow()
        if row < 0 or row < self._builtin_count:
            return  # встроенные не редактируем
        from fem3d import load_user_materials
        user = load_user_materials()
        user_idx = row - self._builtin_count
        dlg = MaterialEditorDialog(material=user[user_idx], parent=self)
        if dlg.exec_() == QDialog.Accepted:
            new_mat = dlg.material()
            user[user_idx] = new_mat
            from fem3d import save_user_materials
            save_user_materials(user)
            self._refresh()

    def _on_remove(self):
        row = self.table.currentRow()
        if row < 0 or row < self._builtin_count:
            return
        from fem3d import load_user_materials, save_user_materials
        user = load_user_materials()
        del user[row - self._builtin_count]
        save_user_materials(user)
        self._refresh()

    def _save_user_material(self, mat, add: bool):
        from fem3d import load_user_materials, save_user_materials
        user = load_user_materials()
        user.append(mat)
        save_user_materials(user)


class MaterialEditorDialog(QDialog):
    """Редактор одного материала."""

    def __init__(self, material=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Создать материал" if material is None
                            else "Редактировать материал")
        self.setMinimumWidth(420)

        from PyQt5.QtWidgets import QLineEdit
        outer = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setText(material.name if material else "")
        form.addRow("Имя:", self.name_edit)

        self.category_edit = QLineEdit()
        self.category_edit.setText(material.category if material else "Пользовательские")
        form.addRow("Категория:", self.category_edit)

        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(1e-4, 1e5); self.lambda_spin.setDecimals(4)
        self.lambda_spin.setValue(material.lambda_ if material else 50.0)
        self.lambda_spin.setSuffix(" Вт/(м·К)")
        form.addRow("λ (теплопроводность):", self.lambda_spin)

        self.rho_spin = QDoubleSpinBox()
        self.rho_spin.setRange(0, 1e6); self.rho_spin.setDecimals(1)
        self.rho_spin.setValue(material.rho if material else 0)
        self.rho_spin.setSuffix(" кг/м³")
        form.addRow("ρ (плотность):", self.rho_spin)

        self.cp_spin = QDoubleSpinBox()
        self.cp_spin.setRange(0, 1e5); self.cp_spin.setDecimals(1)
        self.cp_spin.setValue(material.cp if material else 0)
        self.cp_spin.setSuffix(" Дж/(кг·К)")
        form.addRow("c_p (теплоёмкость):", self.cp_spin)

        self.emissivity_spin = QDoubleSpinBox()
        self.emissivity_spin.setRange(0.0, 1.0); self.emissivity_spin.setDecimals(2)
        self.emissivity_spin.setSingleStep(0.05)
        self.emissivity_spin.setValue(material.emissivity if material else 0.85)
        form.addRow("ε (степень черноты):", self.emissivity_spin)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(-1.0, 1.0); self.beta_spin.setDecimals(6)
        self.beta_spin.setValue(material.lambda_temp_coef if material else 0.0)
        self.beta_spin.setSuffix(" 1/К")
        form.addRow("β (температ. коэф. λ):", self.beta_spin)

        # Анизотропия.
        from PyQt5.QtWidgets import QCheckBox
        self.aniso_check = QCheckBox(
            "Анизотропный материал (разные λ по осям X/Y/Z)")
        self.aniso_check.setChecked(material.is_anisotropic if material else False)
        self.aniso_check.setToolTip(
            "Включить тензорную теплопроводность.\n"
            "Сборка матрицы:\n"
            "  K^e_ij = (λ_x·b_i·b_j + λ_y·c_i·c_j + λ_z·d_i·d_j) / (36V)\n"
            "Поток q вычисляется покомпонентно: q_i = −λ_i ∂T/∂x_i.")
        form.addRow(self.aniso_check)

        # Контейнер с тремя λ.
        self.aniso_widget = QWidget()
        aniso_form = QFormLayout(self.aniso_widget)
        aniso_form.setContentsMargins(20, 0, 0, 0)

        # Пресеты известных анизотропных материалов.
        aniso_preset_combo = QComboBox()
        ANISO_PRESETS = [
            ("— Пресет —",              None),
            ("Изотропный (λ_x = λ_y = λ_z)", (50.0, 50.0, 50.0)),
            ("Графит (вдоль слоёв)",    (200.0, 200.0, 10.0)),
            ("CFRP — углепластик",      (10.0, 0.8, 0.8)),
            ("Дерево (вдоль волокон)",  (0.40, 0.17, 0.17)),
            ("PCB FR-4",                 (0.8, 0.8, 0.3)),
            ("Слоистая керамика",        (30.0, 30.0, 3.0)),
            ("Армированный композит",   (50.0, 5.0, 5.0)),
        ]
        for label, vals in ANISO_PRESETS:
            aniso_preset_combo.addItem(label, vals)
        aniso_preset_combo.setToolTip(
            "Готовые значения для типичных анизотропных материалов.\n"
            "Например, графит проводит тепло в 20 раз лучше вдоль слоёв.")
        aniso_form.addRow("Пресет:", aniso_preset_combo)

        self.lam_x_spin = QDoubleSpinBox()
        self.lam_x_spin.setRange(1e-4, 1e5); self.lam_x_spin.setDecimals(4)
        self.lam_x_spin.setSuffix(" Вт/(м·К)")
        self.lam_x_spin.setValue(material.lambda_x if (material and material.is_anisotropic)
                                  else (material.lambda_ if material else 50.0))
        aniso_form.addRow("λ по X:", self.lam_x_spin)
        self.lam_y_spin = QDoubleSpinBox()
        self.lam_y_spin.setRange(1e-4, 1e5); self.lam_y_spin.setDecimals(4)
        self.lam_y_spin.setSuffix(" Вт/(м·К)")
        self.lam_y_spin.setValue(material.lambda_y if (material and material.is_anisotropic)
                                  else (material.lambda_ if material else 50.0))
        aniso_form.addRow("λ по Y:", self.lam_y_spin)
        self.lam_z_spin = QDoubleSpinBox()
        self.lam_z_spin.setRange(1e-4, 1e5); self.lam_z_spin.setDecimals(4)
        self.lam_z_spin.setSuffix(" Вт/(м·К)")
        self.lam_z_spin.setValue(material.lambda_z if (material and material.is_anisotropic)
                                  else (material.lambda_ if material else 50.0))
        aniso_form.addRow("λ по Z:", self.lam_z_spin)

        # Индикатор степени анизотропии.
        self.aniso_indicator = QLabel("")
        self.aniso_indicator.setStyleSheet("color: #b89c40; font-style: italic;")
        aniso_form.addRow("", self.aniso_indicator)

        def _update_indicator():
            x = self.lam_x_spin.value(); y = self.lam_y_spin.value(); z = self.lam_z_spin.value()
            lmax, lmin = max(x, y, z), min(x, y, z)
            ratio = lmax / lmin if lmin > 0 else float("inf")
            if ratio < 1.05:
                txt = "Почти изотропный (λ_max/λ_min < 1.05)"
            elif ratio < 2.0:
                txt = f"Слабая анизотропия (λ_max/λ_min = {ratio:.2f})"
            elif ratio < 20.0:
                txt = f"Заметная анизотропия (λ_max/λ_min = {ratio:.1f})"
            else:
                txt = f"Сильная анизотропия (λ_max/λ_min = {ratio:.0f})"
            self.aniso_indicator.setText(txt)
        _update_indicator()
        for sp in (self.lam_x_spin, self.lam_y_spin, self.lam_z_spin):
            sp.valueChanged.connect(lambda _v: _update_indicator())

        def _on_aniso_preset(_idx):
            vals = aniso_preset_combo.currentData()
            if vals is not None:
                self.lam_x_spin.setValue(vals[0])
                self.lam_y_spin.setValue(vals[1])
                self.lam_z_spin.setValue(vals[2])
        aniso_preset_combo.currentIndexChanged.connect(_on_aniso_preset)

        form.addRow(self.aniso_widget)
        self.aniso_widget.setVisible(self.aniso_check.isChecked())
        self.aniso_check.toggled.connect(self.aniso_widget.setVisible)

        outer.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def material(self):
        from fem3d import Material
        is_aniso = self.aniso_check.isChecked()
        return Material(
            name=self.name_edit.text() or "Без названия",
            lambda_=float(self.lambda_spin.value()),
            rho=float(self.rho_spin.value()),
            cp=float(self.cp_spin.value()),
            emissivity=float(self.emissivity_spin.value()),
            lambda_temp_coef=float(self.beta_spin.value()),
            category=self.category_edit.text() or "Пользовательские",
            is_anisotropic=is_aniso,
            lambda_x=float(self.lam_x_spin.value()),
            lambda_y=float(self.lam_y_spin.value()),
            lambda_z=float(self.lam_z_spin.value()),
        )


# =============================================================================
# Компактный диалог редактирования граничных условий — таблица 6 граней.
# =============================================================================

class BoundaryConditionsDialog(QDialog):
    """Диалог граничных условий: вкладки + список граней с одним редактором.

    Структура (вместо сплошной «простыни» из карточек):
      • вкладка «Грани» — слева список 6 граней с краткой сводкой текущего
        условия, справа редактор выбранной грани (тип + параметры + формула);
      • вкладка «Обдув» — вынужденная конвекция по скорости потока;
      • вкладка «Погружение» — частичное погружение детали в жидкость.
    Шаблоны сценариев и кнопка «Применить ко всем граням» — над списком.
    """

    # Типы условий с физическими названиями (для GUI).
    BC_KIND_NONE       = "none"
    BC_KIND_DIRICHLET  = "dirichlet"     # заданная температура
    BC_KIND_INSULATED  = "insulated"     # изоляция (Нейман с q=0)
    BC_KIND_HEAT_FLUX  = "heat_flux"     # тепловой поток (Нейман с q≠0)
    BC_KIND_CONVECTION = "convection"    # конвекция (Робен)
    BC_KIND_RADIATION  = "radiation"     # излучение (Стефан-Больцман)

    KIND_ORDER = (BC_KIND_NONE, BC_KIND_DIRICHLET, BC_KIND_INSULATED,
                  BC_KIND_HEAT_FLUX, BC_KIND_CONVECTION, BC_KIND_RADIATION)

    # Описания каждого типа — (название, пояснение, формула).
    BC_KIND_INFO = {
        BC_KIND_NONE: (
            "Не задано",
            "Грань без условия. Эквивалентно изоляции в стационаре, но лучше "
            "задать явно для ясности.",
            ""
        ),
        BC_KIND_DIRICHLET: (
            "Заданная температура",
            "На этой грани температура фиксирована и равна T₀. "
            "Используется когда грань контактирует с термостатом, "
            "массивной деталью при известной температуре, или когда задана "
            "наблюдаемая температура поверхности.",
            "T = T₀"
        ),
        BC_KIND_INSULATED: (
            "Изоляция (адиабатическая грань)",
            "Через грань тепло не проходит. Используется для границ "
            "симметрии или хорошо изолированных стенок.",
            "∂T/∂n = 0"
        ),
        BC_KIND_HEAT_FLUX: (
            "Заданный тепловой поток",
            "Через грань течёт заданный поток. q > 0 = тело нагревается "
            "(тепло ВХОДИТ); q < 0 = охлаждается. Типичные значения: "
            "солнечный поток 800 Вт/м², нагрев плитой 5000–50000 Вт/м².",
            "−λ ∂T/∂n = −q   (q > 0 — нагрев)"
        ),
        BC_KIND_CONVECTION: (
            "Конвекция (Ньютон)",
            "Грань охлаждается / нагревается окружающей средой. "
            "α — коэффициент теплоотдачи, T∞ — температура среды. "
            "Типичные α: воздух 5–25, воздух с вентилятором 30–200, "
            "вода 500–10000 Вт/(м²·К).",
            "−λ ∂T/∂n = α (T − T∞)"
        ),
        BC_KIND_RADIATION: (
            "Излучение (Стефан-Больцман)",
            "Тепло уходит излучением в окружающую среду с температурой T_окр. "
            "ε — степень черноты поверхности (полированный металл ≈ 0.05; "
            "оксидированный/окрашенный ≈ 0.85–0.95). "
            "Учитывается только тепловое излучение, без учёта окружающих тел.",
            "−λ ∂T/∂n = ε σ (T⁴ − T_окр⁴)"
        ),
    }

    # Короткие подписи для сводки в списке граней.
    KIND_SHORT = {
        BC_KIND_NONE:       "не задано",
        BC_KIND_DIRICHLET:  "T₀",
        BC_KIND_INSULATED:  "изоляция",
        BC_KIND_HEAT_FLUX:  "поток q",
        BC_KIND_CONVECTION: "конвекция",
        BC_KIND_RADIATION:  "излучение",
    }

    def __init__(self, problem, parent: Optional[QWidget] = None,
                 focus_face: int = 0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Граничные условия")
        self.resize(820, 560)
        from fem3d import (FACE_NAMES, HEATING_TEMPLATES, CONVECTION_PRESETS)

        self._problem = problem
        self._face_names = FACE_NAMES
        self._presets = CONVECTION_PRESETS

        # --- Модель: состояние каждой грани (тип + все параметры). ----------
        self._state = {}
        for fid in range(6):
            bc = problem.bcs[fid]
            self._state[fid] = {
                "kind":  self._bc_to_kind(bc),
                "T0":    float(bc.T0),
                "q":     float(bc.q0) if bc.q0 != 0 else 1000.0,
                "alpha": float(bc.alpha) if bc.alpha > 0 else 10.0,
                "tinf":  float(bc.T_inf),
                "eps":   float(getattr(bc, "emissivity", 0.85)),
                "tenv":  float(bc.T_inf),
            }
        self._current_fid = max(0, min(5, int(focus_face)))
        self._loading = False     # защита от записи при загрузке редактора

        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self._build_faces_tab(), "Грани")
        tabs.addTab(self._build_air_flow_tab(), "Обдув")
        tabs.addTab(self._build_immersion_tab(), "Погружение")
        outer.addWidget(tabs, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        # Стартовое состояние.
        self.face_list.setCurrentRow(self._current_fid)
        self._load_editor(self._current_fid)
        self._refresh_all_summaries()

    # =========================================================================
    # Вкладка «Грани»: список + редактор.
    # =========================================================================
    def _build_faces_tab(self) -> QWidget:
        from PyQt5.QtWidgets import QListWidget, QSplitter

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(8)

        # --- Верхняя строка: шаблон + «применить ко всем». -------------------
        from fem3d import HEATING_TEMPLATES
        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Шаблон сценария:</b>"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(260)
        self.template_combo.addItem("— Применить шаблон —", None)
        for label, factory in HEATING_TEMPLATES:
            self.template_combo.addItem(label, factory)
        self.template_combo.currentIndexChanged.connect(self._on_template)
        top.addWidget(self.template_combo, 1)

        self.apply_all_btn = QPushButton("Применить ко всем граням")
        self.apply_all_btn.setToolTip(
            "Скопировать условие текущей грани на все шесть граней.")
        self.apply_all_btn.clicked.connect(self._on_apply_to_all)
        top.addWidget(self.apply_all_btn)
        lay.addLayout(top)

        # --- Слева список граней, справа редактор. ---------------------------
        split = QSplitter(Qt.Horizontal)

        self.face_list = QListWidget()
        self.face_list.setMinimumWidth(250)
        self.face_list.setAlternatingRowColors(True)
        for fid in range(6):
            self.face_list.addItem("")   # текст заполняется в _refresh_summary
        self.face_list.currentRowChanged.connect(self._on_face_changed)
        split.addWidget(self.face_list)

        split.addWidget(self._build_editor_panel())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)
        return page

    def _build_editor_panel(self) -> QWidget:
        from PyQt5.QtWidgets import QStackedWidget

        panel = QFrame(); panel.setObjectName("Card")
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 10, 12, 10); v.setSpacing(8)

        self.editor_title = QLabel("")
        self.editor_title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        v.addWidget(self.editor_title)

        row = QHBoxLayout()
        row.addWidget(QLabel("Тип условия:"))
        self.kind_combo = QComboBox()
        self.kind_combo.setMinimumWidth(240)
        for kind in self.KIND_ORDER:
            self.kind_combo.addItem(self.BC_KIND_INFO[kind][0], kind)
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        row.addWidget(self.kind_combo, 1)
        v.addLayout(row)

        # Формула — отдельно и заметно.
        self.formula_label = QLabel("")
        self.formula_label.setStyleSheet(
            'font-family: "Cambria Math", "DejaVu Serif", serif; '
            "font-size: 11pt; padding: 2px 0;")
        v.addWidget(self.formula_label)

        # Пояснение.
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        v.addWidget(self.hint_label)

        # --- Параметры под каждый тип (стек). --------------------------------
        self.stack = QStackedWidget()

        # 0: None.
        w0 = QWidget(); f0 = QFormLayout(w0); f0.setContentsMargins(0, 4, 0, 0)
        f0.addRow(QLabel("<i>Параметров нет</i>"))
        self.stack.addWidget(w0)

        # 1: Dirichlet — T₀.
        w1 = QWidget(); f1 = QFormLayout(w1); f1.setContentsMargins(0, 4, 0, 0)
        self.sp_T0 = QDoubleSpinBox()
        self.sp_T0.setRange(-273, 5000); self.sp_T0.setDecimals(2)
        self.sp_T0.setSuffix(" °C")
        self.sp_T0.setToolTip("Температура в градусах Цельсия.\n"
                               "Диапазон: −273 ... +5000 °C")
        f1.addRow("Температура T₀:", self.sp_T0)
        self.stack.addWidget(w1)

        # 2: Insulated.
        w2 = QWidget(); f2 = QFormLayout(w2); f2.setContentsMargins(0, 4, 0, 0)
        f2.addRow(QLabel("<i>Параметров нет (∂T/∂n = 0)</i>"))
        self.stack.addWidget(w2)

        # 3: Heat flux — q.
        w3 = QWidget(); f3 = QFormLayout(w3); f3.setContentsMargins(0, 4, 0, 0)
        self.sp_q = QDoubleSpinBox()
        self.sp_q.setRange(-1e9, 1e9); self.sp_q.setDecimals(2)
        self.sp_q.setSuffix(" Вт/м²")
        self.sp_q.setToolTip("Плотность теплового потока через грань.\n"
                              "Положительное q — тело НАГРЕВАЕТСЯ, "
                              "отрицательное — охлаждается.\n"
                              "Типичные значения [Вт/м²]:\n"
                              "  Солнечный поток ≈ 800–1000\n"
                              "  Электрическая плита ≈ 10 000–50 000\n"
                              "  Импульсный лазер ≈ 10⁵–10⁸")
        f3.addRow("Поток q (+нагрев, −отвод):", self.sp_q)
        self.stack.addWidget(w3)

        # 4: Convection — пресет + α + T∞.
        w4 = QWidget(); f4 = QFormLayout(w4); f4.setContentsMargins(0, 4, 0, 0)
        self.preset_combo = QComboBox()
        for label, alpha_v, tinf_v in self._presets:
            self.preset_combo.addItem(label, (alpha_v, tinf_v))
        self.preset_combo.setToolTip(
            "Готовые значения α и T∞ для типичных сред.")
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        f4.addRow("Пресет:", self.preset_combo)
        self.sp_alpha = QDoubleSpinBox()
        self.sp_alpha.setRange(0, 1e6); self.sp_alpha.setDecimals(2)
        self.sp_alpha.setSuffix(" Вт/(м²·К)")
        self.sp_alpha.setToolTip(
            "Коэффициент теплоотдачи α.\n"
            "Закон Ньютона: q = α (T_поверхности − T∞).\n"
            "Типичные значения [Вт/(м²·К)]:\n"
            "  Воздух (свободная конвекция) ≈ 5–25\n"
            "  Воздух с вентилятором ≈ 25–250\n"
            "  Вода (свободная) ≈ 500–1000\n"
            "  Вода с насосом ≈ 1000–15000\n"
            "  Кипение ≈ 2500–25000\n"
            "  Конденсация пара ≈ 5000–100 000")
        f4.addRow("Коэф. теплоотдачи α:", self.sp_alpha)
        self.sp_tinf = QDoubleSpinBox()
        self.sp_tinf.setRange(-273, 5000); self.sp_tinf.setDecimals(2)
        self.sp_tinf.setSuffix(" °C")
        self.sp_tinf.setToolTip(
            "Температура окружающей среды (вдали от поверхности).")
        f4.addRow("Температура среды T∞:", self.sp_tinf)
        self.stack.addWidget(w4)

        # 5: Radiation — ε + T_окр.
        w5 = QWidget(); f5 = QFormLayout(w5); f5.setContentsMargins(0, 4, 0, 0)
        self.sp_eps = QDoubleSpinBox()
        self.sp_eps.setRange(0.0, 1.0); self.sp_eps.setSingleStep(0.05)
        self.sp_eps.setDecimals(2)
        self.sp_eps.setToolTip(
            "Степень черноты поверхности ε ∈ [0, 1].\n"
            "  Полированный металл: 0.02 – 0.10\n"
            "  Окислённый/окрашенный: 0.20 – 0.95\n"
            "  Чёрное тело (абсолютное): 1.00")
        f5.addRow("Степень черноты ε (0..1):", self.sp_eps)
        self.sp_tenv = QDoubleSpinBox()
        self.sp_tenv.setRange(-273, 5000); self.sp_tenv.setDecimals(2)
        self.sp_tenv.setSuffix(" °C")
        self.sp_tenv.setToolTip("Температура окружающего пространства, °C.\n"
                                 "Применяется в формуле Стефана-Больцмана:\n"
                                 "q = ε σ (T⁴ − T_окр⁴)")
        f5.addRow("Температура окружения T_окр:", self.sp_tenv)
        self.stack.addWidget(w5)

        v.addWidget(self.stack)
        v.addStretch(1)

        # Запись значений редактора в модель «на лету».
        self.sp_T0.valueChanged.connect(
            lambda val: self._write_state("T0", val))
        self.sp_q.valueChanged.connect(
            lambda val: self._write_state("q", val))
        self.sp_alpha.valueChanged.connect(
            lambda val: self._write_state("alpha", val))
        self.sp_tinf.valueChanged.connect(
            lambda val: self._write_state("tinf", val))
        self.sp_eps.valueChanged.connect(
            lambda val: self._write_state("eps", val))
        self.sp_tenv.valueChanged.connect(
            lambda val: self._write_state("tenv", val))
        return panel

    # --- Модель ↔ редактор ----------------------------------------------------

    def _write_state(self, key: str, value: float) -> None:
        if self._loading:
            return
        self._state[self._current_fid][key] = float(value)
        self._refresh_summary(self._current_fid)

    def _on_kind_changed(self, _idx: int) -> None:
        kind = self.kind_combo.currentData()
        idx = self.KIND_ORDER.index(kind)
        self.stack.setCurrentIndex(idx)
        info = self.BC_KIND_INFO[kind]
        self.formula_label.setText(f"<b>{info[2]}</b>" if info[2] else "")
        self.hint_label.setText(info[1])
        if not self._loading:
            self._state[self._current_fid]["kind"] = kind
            self._refresh_summary(self._current_fid)

    def _on_preset(self, _i: int) -> None:
        data = self.preset_combo.currentData()
        if data and data[0] is not None:
            self.sp_alpha.setValue(data[0])
            self.sp_tinf.setValue(data[1])

    def _on_face_changed(self, row: int) -> None:
        if row < 0:
            return
        self._current_fid = row
        self._load_editor(row)

    def _load_editor(self, fid: int) -> None:
        st = self._state[fid]
        self._loading = True
        try:
            self.editor_title.setText(f"Грань {self._face_names[fid]}")
            self.kind_combo.setCurrentIndex(self.KIND_ORDER.index(st["kind"]))
            self._on_kind_changed(0)   # обновить стек/формулу/подсказку
            self.sp_T0.setValue(st["T0"])
            self.sp_q.setValue(st["q"])
            self.sp_alpha.setValue(st["alpha"])
            self.sp_tinf.setValue(st["tinf"])
            self.sp_eps.setValue(st["eps"])
            self.sp_tenv.setValue(st["tenv"])
        finally:
            self._loading = False

    def _summary_text(self, fid: int) -> str:
        st = self._state[fid]
        k = st["kind"]
        name = self._face_names[fid]
        if k == self.BC_KIND_DIRICHLET:
            detail = f"T₀ = {st['T0']:g} °C"
        elif k == self.BC_KIND_HEAT_FLUX:
            detail = f"q = {st['q']:g} Вт/м²"
        elif k == self.BC_KIND_CONVECTION:
            detail = f"α = {st['alpha']:g}, T∞ = {st['tinf']:g} °C"
        elif k == self.BC_KIND_RADIATION:
            detail = f"ε = {st['eps']:g}, T_окр = {st['tenv']:g} °C"
        elif k == self.BC_KIND_INSULATED:
            detail = "∂T/∂n = 0"
        else:
            detail = "—"
        return f"{name}   |   {self.KIND_SHORT[k]}: {detail}"

    def _refresh_summary(self, fid: int) -> None:
        item = self.face_list.item(fid)
        if item is not None:
            item.setText(self._summary_text(fid))

    def _refresh_all_summaries(self) -> None:
        for fid in range(6):
            self._refresh_summary(fid)

    def _on_apply_to_all(self) -> None:
        cur = dict(self._state[self._current_fid])
        for fid in range(6):
            self._state[fid] = dict(cur)
        self._refresh_all_summaries()

    # =========================================================================
    # Вкладка «Обдув».
    # =========================================================================
    def _build_air_flow_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(8)

        intro = QLabel(
            "Вынужденная конвекция при обтекании фигуры потоком воздуха. "
            "По скорости U вычисляются Re, Nu и коэффициент теплоотдачи h "
            "по размерам РЕАЛЬНОЙ геометрии, после чего конвекция "
            "назначается на все шесть граней.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {current_theme().text_dim};")
        lay.addWidget(intro)

        self.flow_enable = QCheckBox(
            "Задавать конвекцию на гранях через обдув (а не вручную α)")
        self.flow_enable.setChecked(
            bool(getattr(self._problem, "air_flow_enabled", False)))
        lay.addWidget(self.flow_enable)

        form = QFormLayout()
        self.flow_speed = QDoubleSpinBox()
        self.flow_speed.setRange(0.0, 500.0); self.flow_speed.setDecimals(2)
        self.flow_speed.setSuffix(" м/с")
        self.flow_speed.setValue(float(getattr(self._problem,
                                               "air_flow_speed", 0.0)) or 5.0)
        form.addRow("Скорость U:", self.flow_speed)

        self.flow_dir = QComboBox()
        self._flow_dirs = [("+X", "+x"), ("−X", "-x"), ("+Y", "+y"),
                           ("−Y", "-y"), ("+Z", "+z"), ("−Z", "-z")]
        for label, _ in self._flow_dirs:
            self.flow_dir.addItem(label)
        cur_dir = getattr(self._problem, "air_flow_direction", "+x")
        for i, (_, code) in enumerate(self._flow_dirs):
            if code == cur_dir:
                self.flow_dir.setCurrentIndex(i)
        form.addRow("Направление:", self.flow_dir)

        self.flow_tinf = QDoubleSpinBox()
        self.flow_tinf.setRange(-273.0, 2000.0); self.flow_tinf.setDecimals(1)
        self.flow_tinf.setSuffix(" °C")
        self.flow_tinf.setValue(float(getattr(self._problem,
                                              "air_flow_T_inf", 20.0)))
        form.addRow("Температура воздуха T∞:", self.flow_tinf)
        lay.addLayout(form)

        self.flow_apply_btn = QPushButton(
            "Рассчитать h и применить ко всем граням")
        self.flow_apply_btn.clicked.connect(self._on_apply_air_flow)
        lay.addWidget(self.flow_apply_btn)

        self.flow_result = QLabel(
            "Введите скорость и нажмите «Рассчитать h…». "
            "Re, Nu и h считаются по размерам самой фигуры.")
        self.flow_result.setWordWrap(True)
        self.flow_result.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        lay.addWidget(self.flow_result)
        lay.addStretch(1)
        return page

    def _on_apply_air_flow(self):
        """Посчитать h по обдуву и проставить конвекцию на все грани."""
        from fem3d import convection as cv
        if self._problem.nodes is None:
            self.flow_result.setText("Сначала постройте сетку фигуры.")
            return
        speed = float(self.flow_speed.value())
        direction = self._flow_dirs[self.flow_dir.currentIndex()][1]
        T_inf = float(self.flow_tinf.value())
        self._problem.air_flow_enabled = True
        self._problem.air_flow_speed = speed
        self._problem.air_flow_direction = direction
        self._problem.air_flow_T_inf = T_inf
        T_surface = (float(self._problem.T.mean())
                     if getattr(self._problem, "T", None) is not None
                     and self._problem.T.size else None)
        try:
            res = cv.analyze_problem_air_flow(self._problem,
                                              T_surface=T_surface)
        except Exception as exc:
            self.flow_result.setText(f"Ошибка: {exc}")
            return
        if res is None:
            self.flow_result.setText("Задайте скорость > 0.")
            return
        # Конвекция α=h, T∞ — во ВСЕ грани (в модель, не в виджеты).
        for fid in range(6):
            st = self._state[fid]
            st["kind"] = self.BC_KIND_CONVECTION
            st["alpha"] = float(res.h)
            st["tinf"] = T_inf
        self._refresh_all_summaries()
        self._load_editor(self._current_fid)
        self.flow_enable.setChecked(True)
        self.flow_result.setText(
            f"Re = {res.Re:.3g},  Nu = {res.Nu:.1f},  "
            f"h = {res.h:.2f} Вт/(м²·К)  ({res.regime}).  "
            f"Назначено на все 6 граней. A_полн = {res.total_area:.4g} м², "
            f"Q ≈ {res.Q_total:.4g} Вт.")

    def air_flow_result(self) -> dict:
        """Параметры обдува для сохранения в задачу."""
        return {
            "enabled": bool(self.flow_enable.isChecked()),
            "speed": float(self.flow_speed.value()),
            "direction": self._flow_dirs[self.flow_dir.currentIndex()][1],
            "T_inf": float(self.flow_tinf.value()),
        }

    # =========================================================================
    # Вкладка «Погружение».
    # =========================================================================
    def _build_immersion_tab(self) -> QWidget:
        from fem3d import BC_DIRICHLET, BC_ROBIN
        imm = getattr(self._problem, "immersion", None)
        wb = imm.wetted_bc if imm else None

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(8)

        intro = QLabel(
            "Деталь частично опущена в жидкость. Линия воды режет стенки: "
            "фасетки ниже неё получают ГУ воды, выше — ГУ, заданные на "
            "вкладке «Грани». Кипяток ⇒ поверхность ≈ T воды, поэтому "
            "обоснован Дирихле; h воды важен только для некипящей ванны.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {current_theme().text_dim};")
        lay.addWidget(intro)

        self.imm_enable = QCheckBox(
            "Деталь частично погружена в жидкость")
        self.imm_enable.setChecked(bool(imm.enabled) if imm else False)
        lay.addWidget(self.imm_enable)

        form = QFormLayout()
        self.imm_axis = QComboBox()
        for label, code in [("X", 0), ("Y", 1), ("Z (вертикаль)", 2)]:
            self.imm_axis.addItem(label, code)
        self.imm_axis.setCurrentIndex(imm.axis if imm else 2)
        form.addRow("Ось погружения:", self.imm_axis)

        self.imm_side = QComboBox()
        for label, code in [("нижний конец", 0), ("верхний конец", 1)]:
            self.imm_side.addItem(label, code)
        self.imm_side.setCurrentIndex(imm.side if imm else 0)
        form.addRow("В воде:", self.imm_side)

        self.imm_level = QDoubleSpinBox()
        self.imm_level.setRange(0.0, 100.0); self.imm_level.setDecimals(4)
        self.imm_level.setSingleStep(0.005); self.imm_level.setSuffix(" м")
        self.imm_level.setValue(float(imm.level) if imm else 0.05)
        form.addRow("Линия воды (глубина):", self.imm_level)

        self.imm_kind = QComboBox()
        self.imm_kind.addItem("Дирихле T (кипяток, рекоменд.)", "dirichlet")
        self.imm_kind.addItem("Робин: h + T воды", "robin")
        self.imm_kind.setCurrentIndex(
            1 if (wb is not None and wb.type == BC_ROBIN) else 0)
        form.addRow("Смоченный поясок:", self.imm_kind)

        self.imm_twater = QDoubleSpinBox()
        self.imm_twater.setRange(-273.0, 2000.0); self.imm_twater.setDecimals(1)
        self.imm_twater.setSuffix(" °C")
        self.imm_twater.setValue(
            float(wb.T0) if (wb and wb.type == BC_DIRICHLET)
            else (float(wb.T_inf) if wb else 100.0))
        form.addRow("Температура воды:", self.imm_twater)

        self.imm_hwater = QDoubleSpinBox()
        self.imm_hwater.setRange(0.0, 1e6); self.imm_hwater.setDecimals(0)
        self.imm_hwater.setSuffix(" Вт/(м²·К)")
        self.imm_hwater.setValue(
            float(wb.alpha) if (wb and wb.type == BC_ROBIN and wb.alpha > 0)
            else 10000.0)
        form.addRow("h воды:", self.imm_hwater)
        lay.addLayout(form)
        lay.addStretch(1)

        def _sync_kind():
            self.imm_hwater.setEnabled(
                self.imm_kind.currentData() == "robin")
        self.imm_kind.currentIndexChanged.connect(lambda _i: _sync_kind())
        _sync_kind()
        return page

    def immersion_result(self):
        """Объект Immersion для записи в задачу."""
        from fem3d import BC_DIRICHLET, BC_ROBIN
        from fem3d.problem import Immersion, BoundaryCondition
        if self.imm_kind.currentData() == "robin":
            wetted = BoundaryCondition(type=BC_ROBIN,
                                       alpha=float(self.imm_hwater.value()),
                                       T_inf=float(self.imm_twater.value()))
        else:
            wetted = BoundaryCondition(type=BC_DIRICHLET,
                                       T0=float(self.imm_twater.value()))
        return Immersion(
            enabled=bool(self.imm_enable.isChecked()),
            axis=int(self.imm_axis.currentData()),
            level=float(self.imm_level.value()),
            side=int(self.imm_side.currentData()),
            wetted_bc=wetted,
        )

    # =========================================================================
    # Шаблоны и преобразования.
    # =========================================================================
    def _bc_to_kind(self, bc) -> str:
        from fem3d import (BC_NONE, BC_DIRICHLET, BC_NEUMANN, BC_ROBIN,
                            BC_RADIATION)
        if bc.type == BC_NONE: return self.BC_KIND_NONE
        if bc.type == BC_DIRICHLET: return self.BC_KIND_DIRICHLET
        if bc.type == BC_NEUMANN:
            return (self.BC_KIND_INSULATED if abs(bc.q0) < 1e-15
                    else self.BC_KIND_HEAT_FLUX)
        if bc.type == BC_ROBIN: return self.BC_KIND_CONVECTION
        if bc.type == BC_RADIATION: return self.BC_KIND_RADIATION
        return self.BC_KIND_NONE

    def _on_template(self, _idx):
        factory = self.template_combo.currentData()
        if factory is None:
            return
        new_bcs = factory()
        for fid, bc in new_bcs.items():
            st = self._state[fid]
            st["kind"] = self._bc_to_kind(bc)
            st["T0"] = float(bc.T0)
            if bc.q0 != 0:
                st["q"] = float(bc.q0)
            if bc.alpha > 0:
                st["alpha"] = float(bc.alpha)
            st["tinf"] = float(bc.T_inf)
            st["tenv"] = float(bc.T_inf)
        self._refresh_all_summaries()
        self._load_editor(self._current_fid)
        self.template_combo.blockSignals(True)
        self.template_combo.setCurrentIndex(0)
        self.template_combo.blockSignals(False)

    def result_bcs(self) -> dict:
        from fem3d import (BoundaryCondition, BC_NONE, BC_DIRICHLET,
                            BC_NEUMANN, BC_ROBIN, BC_RADIATION)
        kind_to_bc_type = {
            self.BC_KIND_NONE: BC_NONE,
            self.BC_KIND_DIRICHLET: BC_DIRICHLET,
            self.BC_KIND_INSULATED: BC_NEUMANN,
            self.BC_KIND_HEAT_FLUX: BC_NEUMANN,
            self.BC_KIND_CONVECTION: BC_ROBIN,
            self.BC_KIND_RADIATION: BC_RADIATION,
        }
        result = {}
        for fid in range(6):
            st = self._state[fid]
            kind = st["kind"]
            bc = BoundaryCondition(type=kind_to_bc_type[kind])
            if kind == self.BC_KIND_DIRICHLET:
                bc.T0 = st["T0"]
            elif kind == self.BC_KIND_INSULATED:
                bc.q0 = 0.0
            elif kind == self.BC_KIND_HEAT_FLUX:
                bc.q0 = st["q"]
            elif kind == self.BC_KIND_CONVECTION:
                bc.alpha = st["alpha"]
                bc.T_inf = st["tinf"]
            elif kind == self.BC_KIND_RADIATION:
                bc.emissivity = st["eps"]
                bc.T_inf = st["tenv"]
            result[fid] = bc
        return result


# =============================================================================
# Диалог параметров нестационарной задачи.
# =============================================================================

class TransientParamsDialog(QDialog):
    """Параметры нестационарного расчёта.

    Базовые: t_end, dt, T_init, n_save.
    Теплофизика: rho, cp (предзаполняются из задачи, можно переопределить).
    Отображение: fps анимации, фиксированная цветовая шкала, автоповтор.
    Диалог показывает живую сводку: a = lambda/(rho*cp), tau = L^2/a,
    число шагов, Fo — и умеет подобрать dt и t_end по tau.
    """

    def __init__(self, parent=None, problem=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры нестационарной задачи")
        self.setModal(True)
        self.resize(480, 560)
        self._problem = problem

        layout = QVBoxLayout(self)

        hint = QLabel(
            "Нестационарная теплопроводность:\n"
            "  ρ·c_p · ∂T/∂t = ∇·(λ∇T) + Q\n"
            "Неявная схема Эйлера (безусловно устойчива, точность O(Δt)).\n"
            "Характерное время: τ ≈ ρ·c_p·L²/λ — тепловая инерция тела."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {current_theme().text_dim}; padding: 4px;")
        layout.addWidget(hint)

        form = QFormLayout()

        # --- Интегрирование по времени -----------------------------------
        self.t_end_spin = QDoubleSpinBox()
        self.t_end_spin.setRange(1e-6, 1e8)
        self.t_end_spin.setDecimals(3)
        self.t_end_spin.setValue(3600.0)
        self.t_end_spin.setSuffix(" с")
        self.t_end_spin.setToolTip(
            "Финальное физическое время моделирования.\n"
            "Эталонно: 3–5 характерных времён τ "
            "для практически полного выхода в стационар.")
        form.addRow("Конечное время t_end:", self.t_end_spin)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(1e-9, 1e6)
        self.dt_spin.setDecimals(6)
        self.dt_spin.setValue(60.0)
        self.dt_spin.setSuffix(" с")
        self.dt_spin.setToolTip(
            "Шаг по времени Δt.\n"
            "Рекомендация: Δt ≈ τ/30…τ/50 для гладкой динамики.\n"
            "Неявная схема Эйлера устойчива при любом Δt, "
            "но точность ~O(Δt).")
        form.addRow("Шаг по времени Δt:", self.dt_spin)

        self.t_init_spin = QDoubleSpinBox()
        self.t_init_spin.setRange(-273.0, 5000.0)
        self.t_init_spin.setDecimals(2)
        self.t_init_spin.setValue(100.0)
        self.t_init_spin.setSuffix(" °C")
        self.t_init_spin.setToolTip(
            "Начальная температура во ВСЕХ узлах тела при t=0.\n"
            "Чтобы увидеть динамику (нагрев/остывание), она должна\n"
            "ОТЛИЧАТЬСЯ от температуры среды T∞ / стенки T₀ в ГУ.\n"
            "Например: T₀=100 °C при конвекции с T∞=20 °C → остывание.")
        form.addRow("Начальная T₀:", self.t_init_spin)

        self.n_save_spin = QSpinBox()
        self.n_save_spin.setRange(2, 1000)
        self.n_save_spin.setValue(50)
        self.n_save_spin.setToolTip(
            "Число сохранённых снимков для проигрывателя анимации.\n"
            "Снимки распределены равномерно во времени.\n"
            "Не может превышать число шагов t_end/Δt + 1 — лишние\n"
            "снимки автоматически отбрасываются.")
        form.addRow("Снимков для анимации:", self.n_save_spin)

        # --- Теплофизические свойства -------------------------------------
        rho0, cp0 = 2700.0, 900.0
        if problem is not None:
            if getattr(problem, "rho", 0) > 0:
                rho0 = float(problem.rho)
            if getattr(problem, "cp", 0) > 0:
                cp0 = float(problem.cp)

        self.rho_spin = QDoubleSpinBox()
        self.rho_spin.setRange(0.1, 50000.0)
        self.rho_spin.setDecimals(1)
        self.rho_spin.setValue(rho0)
        self.rho_spin.setSuffix(" кг/м³")
        self.rho_spin.setToolTip("Плотность материала ρ.")
        form.addRow("Плотность ρ:", self.rho_spin)

        self.cp_spin = QDoubleSpinBox()
        self.cp_spin.setRange(1.0, 100000.0)
        self.cp_spin.setDecimals(1)
        self.cp_spin.setValue(cp0)
        self.cp_spin.setSuffix(" Дж/(кг·К)")
        self.cp_spin.setToolTip("Удельная теплоёмкость c_p.")
        form.addRow("Теплоёмкость c_p:", self.cp_spin)

        # --- Отображение ----------------------------------------------------
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(8)
        self.fps_spin.setSuffix(" кадр/с")
        self.fps_spin.setToolTip("Скорость воспроизведения анимации T(t).")
        form.addRow("Скорость анимации:", self.fps_spin)

        self.fixed_scale_check = QCheckBox(
            "Единая цветовая шкала для всех кадров")
        self.fixed_scale_check.setChecked(True)
        self.fixed_scale_check.setToolTip(
            "Шкала цветов фиксируется по глобальным Tmin/Tmax за всё время.\n"
            "Так рост и падение температуры видны корректно: тело реально\n"
            "«остывает» в синий или «греется» в красный.\n"
            "Без фиксации каждый кадр нормируется на свой диапазон и\n"
            "динамика визуально пропадает.")
        form.addRow("", self.fixed_scale_check)

        self.loop_check = QCheckBox("Зациклить воспроизведение")
        self.loop_check.setChecked(False)
        form.addRow("", self.loop_check)

        layout.addLayout(form)

        # --- Кнопка автоподбора и живая сводка ------------------------------
        self.auto_btn = QPushButton("Подобрать Δt и t_end по τ")
        self.auto_btn.setToolTip(
            "Δt = τ/40, t_end = 4·τ — стандартный выбор:\n"
            "плавная анимация и практически полный выход в стационар.")
        self.auto_btn.clicked.connect(self._on_auto_pick)
        layout.addWidget(self.auto_btn)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            'font-family: "Consolas", "DejaVu Sans Mono", monospace; '
            f"font-size: 9pt; color: {current_theme().text_dim}; padding: 4px;")
        layout.addWidget(self.info_label)

        for w in (self.t_end_spin, self.dt_spin, self.rho_spin, self.cp_spin):
            w.valueChanged.connect(self._update_info)
        self.n_save_spin.valueChanged.connect(self._update_info)
        self._update_info()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- Вспомогательное ------------------------------------------------------

    def _char_length(self) -> float:
        """Характерный размер L: минимальный габарит тела, м."""
        p = self._problem
        if p is not None and getattr(p, "geometry", None) is not None:
            g = p.geometry
            dims = [d for d in (g.Lx, g.Ly, g.Lz) if d and d > 0]
            if dims:
                return float(min(dims))
        return 0.1

    def _lambda(self) -> float:
        p = self._problem
        if p is not None and getattr(p, "lambda_", 0) > 0:
            return float(p.lambda_)
        return 200.0

    def _tau(self) -> float:
        """Характерное время τ = ρ·c_p·L²/λ, с."""
        rho = self.rho_spin.value()
        cp = self.cp_spin.value()
        lam = self._lambda()
        L = self._char_length()
        return rho * cp * L * L / max(lam, 1e-12)

    def _on_auto_pick(self) -> None:
        tau = self._tau()
        self.t_end_spin.setValue(max(4.0 * tau, 1e-3))
        self.dt_spin.setValue(max(tau / 40.0, 1e-6))
        self._update_info()

    def _update_info(self) -> None:
        rho = self.rho_spin.value()
        cp = self.cp_spin.value()
        lam = self._lambda()
        L = self._char_length()
        a = lam / max(rho * cp, 1e-12)            # температуропроводность
        tau = self._tau()
        t_end = self.t_end_spin.value()
        dt = self.dt_spin.value()
        steps = int(np.ceil(t_end / max(dt, 1e-12)))
        Fo = a * t_end / max(L * L, 1e-18)        # число Фурье
        n_save = self.n_save_spin.value()
        lines = [
            f"λ = {lam:g} Вт/(м·К),  L = {L:g} м",
            f"a = λ/(ρ·c_p) = {a:.3e} м²/с",
            f"τ = L²/a = {tau:.4g} с  (≈ {tau/60.0:.3g} мин)",
            f"Шагов по времени: {steps},  Fo = a·t_end/L² = {Fo:.3g}",
        ]
        warn = []
        if steps + 1 < n_save:
            warn.append(f"снимков ({n_save}) больше, чем шагов+1 "
                        f"({steps + 1}) — будет сохранено {steps + 1}")
        if dt > tau and tau > 0:
            warn.append("Δt > τ: динамика будет «перешагнута», "
                        "уменьшите Δt")
        if t_end < 0.5 * tau and tau > 0:
            warn.append("t_end < τ/2: процесс не успеет развиться")
        if steps > 200000:
            warn.append(f"очень много шагов ({steps}) — расчёт будет долгим")
        if warn:
            lines.append("⚠ " + "; ".join(warn))
            self.info_label.setStyleSheet(
                'font-family: "Consolas", monospace; font-size: 9pt; '
                "color: #e8a24e; padding: 4px;")
        else:
            self.info_label.setStyleSheet(
                'font-family: "Consolas", monospace; font-size: 9pt; '
                f"color: {current_theme().text_dim}; padding: 4px;")
        self.info_label.setText("\n".join(lines))

    def params(self) -> dict:
        return {
            "t_end":       float(self.t_end_spin.value()),
            "dt":          float(self.dt_spin.value()),
            "T_init":      float(self.t_init_spin.value()),
            "n_save":      int(self.n_save_spin.value()),
            "rho":         float(self.rho_spin.value()),
            "cp":          float(self.cp_spin.value()),
            "fps":         int(self.fps_spin.value()),
            "fixed_scale": bool(self.fixed_scale_check.isChecked()),
            "loop":        bool(self.loop_check.isChecked()),
        }


# =============================================================================
# Галерея шаблонов граничных условий с миниатюрами.
# =============================================================================

def _make_template_icon_svg(category: str) -> str:
    """Вернуть SVG-миниатюру схемы куба для категории шаблона.
    Грани раскрашены по смыслу: красный=горячо, синий=холод/конвекция,
    оранжевый=поток, серый=изоляция, розовый=излучение."""
    # Цвета.
    HOT = "#e85d4e"; COLD = "#4e8de8"; FLUX = "#e8a24e"
    INS = "#6b7280"; RAD = "#e066b3"; NEU = "#8a9099"
    # Базовая изометрическая проекция куба (3 видимые грани: top, left, right).
    def cube(top, left, right):
        return f'''<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <polygon points="50,12 84,30 50,48 16,30" fill="{top}" stroke="#1a1d22" stroke-width="1.2"/>
  <polygon points="16,30 50,48 50,86 16,68" fill="{left}" stroke="#1a1d22" stroke-width="1.2"/>
  <polygon points="84,30 50,48 50,86 84,68" fill="{right}" stroke="#1a1d22" stroke-width="1.2"/>
</svg>'''
    schemes = {
        "bottom_hot":  cube(COLD, HOT, HOT),
        "top_hot":     cube(HOT, COLD, COLD),
        "all_conv":    cube(COLD, COLD, COLD),
        "one_hot":     cube(NEU, HOT, NEU),
        "two_walls":   cube(NEU, HOT, COLD),
        "flux_bottom": cube(NEU, FLUX, FLUX),
        "flux_top":    cube(FLUX, COLD, COLD),
        "cpu":         cube(COLD, FLUX, FLUX),
        "insulated":   cube(INS, INS, INS),
        "exchanger":   cube(COLD, HOT, NEU),
        "radiation":   cube(RAD, HOT, RAD),
        "reset":       cube("#2a2d33", "#2a2d33", "#2a2d33"),
    }
    return schemes.get(category, cube(NEU, NEU, NEU))


class TemplateGalleryDialog(QDialog):
    """Галерея сценариев ГУ с миниатюрами вместо длинного комбобокса."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор сценария нагрева")
        self.setModal(True)
        self.resize(720, 560)
        self._chosen = None

        outer = QVBoxLayout(self)
        hint = QLabel("Выберите готовый сценарий граничных условий. "
                      "Цвет граней: <span style='color:#e85d4e'>красный — "
                      "нагрев</span>, <span style='color:#4e8de8'>синий — "
                      "конвекция/холод</span>, <span style='color:#e8a24e'>"
                      "оранжевый — поток</span>, <span style='color:#e066b3'>"
                      "розовый — излучение</span>, серый — изоляция.")
        hint.setWordWrap(True); hint.setTextFormat(Qt.RichText)
        from .theme import current_theme as _ct
        hint.setStyleSheet(
            f"color: {_ct().text_dim}; font-size: 9pt; padding: 4px;")
        outer.addWidget(hint)

        from fem3d import HEATING_TEMPLATES_FULL
        try:
            from PyQt5.QtSvg import QSvgWidget
            _has_svg = True
        except ImportError:
            _has_svg = False

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setSpacing(8)

        COLS = 3
        for i, (label, factory, desc, cat) in enumerate(HEATING_TEMPLATES_FULL):
            card = QFrame()
            card.setObjectName("Card")
            _th = _ct()
            card.setStyleSheet(
                f"QFrame#Card {{ border: 1px solid {_th.border}; "
                f"border-radius: 6px; background: {_th.panel}; }} "
                f"QFrame#Card:hover {{ border-color: {_th.accent}; }}")
            cl = QVBoxLayout(card); cl.setContentsMargins(8, 8, 8, 8); cl.setSpacing(4)

            row = QHBoxLayout(); row.addStretch(1)
            if _has_svg:
                try:
                    svg = QSvgWidget()
                    svg.load(bytearray(_make_template_icon_svg(cat),
                                        encoding="utf-8"))
                    svg.setFixedSize(72, 72)
                    row.addWidget(svg)
                except Exception:
                    row.addWidget(self._fallback_icon(cat))
            else:
                row.addWidget(self._fallback_icon(cat))
            row.addStretch(1)
            cl.addLayout(row)

            name = QLabel(f"<b>{label}</b>")
            name.setWordWrap(True); name.setTextFormat(Qt.RichText)
            name.setStyleSheet("font-size: 9pt;")
            cl.addWidget(name)
            d = QLabel(desc); d.setWordWrap(True)
            d.setStyleSheet(f"color: {_th.text_dim}; font-size: 8pt;")
            cl.addWidget(d, 1)

            btn = QPushButton("Применить")
            btn.clicked.connect(lambda _c, f=factory: self._choose(f))
            cl.addWidget(btn)

            grid.addWidget(card, i // COLS, i % COLS)

        scroll.setWidget(grid_host)
        outer.addWidget(scroll, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Cancel)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def _fallback_icon(self, category: str) -> QLabel:
        """Простая цветная иконка, если QtSvg недоступен: три полосы
        (верх/лево/право), окрашенные по схеме грани."""
        HOT = "#e85d4e"; COLD = "#4e8de8"; FLUX = "#e8a24e"
        INS = "#6b7280"; RAD = "#e066b3"; NEU = "#8a9099"
        schemes = {
            "bottom_hot": (COLD, HOT, HOT), "top_hot": (HOT, COLD, COLD),
            "all_conv": (COLD, COLD, COLD), "one_hot": (NEU, HOT, NEU),
            "two_walls": (NEU, HOT, COLD), "flux_bottom": (NEU, FLUX, FLUX),
            "flux_top": (FLUX, COLD, COLD), "cpu": (COLD, FLUX, FLUX),
            "insulated": (INS, INS, INS), "exchanger": (COLD, HOT, NEU),
            "radiation": (RAD, HOT, RAD), "reset": ("#2a2d33",)*3,
        }
        t, l, r = schemes.get(category, (NEU, NEU, NEU))
        lbl = QLabel()
        lbl.setFixedSize(72, 72)
        lbl.setStyleSheet(
            f"border-radius: 6px; "
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1, "
            f"stop:0 {t}, stop:0.5 {l}, stop:1 {r});")
        return lbl

    def _choose(self, factory):
        self._chosen = factory
        self.accept()

    def chosen_factory(self):
        return self._chosen


# =============================================================================
# Диалог: конвективный теплообмен при обтекании потоком.
# =============================================================================
class ForcedConvectionDialog(QDialog):
    """Параметры обдува: скорость, направление, форма тела, T среды.

    По нажатию OK возвращает params(); вызывающий код считает Re/Nu/h и
    (опционально) назначает конвекцию на грани.
    """

    DIRECTIONS = [("+X", "+x"), ("−X", "-x"), ("+Y", "+y"),
                  ("−Y", "-y"), ("+Z", "+z"), ("−Z", "-z")]

    def __init__(self, parent=None, T_surface_hint: float = None):
        super().__init__(parent)
        from fem3d import convection as cv
        self._cv = cv
        self.setWindowTitle("Конвективный теплообмен при обтекании")
        self.setModal(True)
        self.resize(520, 480)

        lay = QVBoxLayout(self)
        hint = QLabel(
            "Вынужденная конвекция при обтекании тела потоком воздуха.\n"
            "Цепочка расчёта:  U, L → Re = U·L/ν → Nu (корреляция формы)\n"
            "→ h = Nu·λ_возд/L → ГУ конвекции (α = h) → поток Q = h·A·ΔT.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {current_theme().text_dim}; padding: 4px;")
        lay.addWidget(hint)

        form = QFormLayout()

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.0, 500.0)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(5.0)
        self.speed_spin.setSuffix(" м/с")
        self.speed_spin.setToolTip("Скорость набегающего потока воздуха U.")
        form.addRow("Скорость потока U:", self.speed_spin)

        self.dir_combo = QComboBox()
        for label, _ in self.DIRECTIONS:
            self.dir_combo.addItem(label)
        self.dir_combo.setToolTip("Направление потока вдоль выбранной оси.")
        form.addRow("Направление потока:", self.dir_combo)

        self.shape_combo = QComboBox()
        self._shapes = [
            (cv.SHAPE_PLATE, cv.SHAPE_NAMES[cv.SHAPE_PLATE]),
            (cv.SHAPE_CYLINDER, cv.SHAPE_NAMES[cv.SHAPE_CYLINDER]),
            (cv.SHAPE_SPHERE, cv.SHAPE_NAMES[cv.SHAPE_SPHERE]),
            (cv.SHAPE_CUBE, cv.SHAPE_NAMES[cv.SHAPE_CUBE]),
        ]
        for _, name in self._shapes:
            self.shape_combo.addItem(name)
        self.shape_combo.setToolTip(
            "Форма тела определяет эмпирическую корреляцию для Nu.")
        form.addRow("Форма тела:", self.shape_combo)

        self.tinf_spin = QDoubleSpinBox()
        self.tinf_spin.setRange(-273.0, 2000.0)
        self.tinf_spin.setDecimals(1)
        self.tinf_spin.setValue(20.0)
        self.tinf_spin.setSuffix(" °C")
        self.tinf_spin.setToolTip("Температура набегающего потока (среды) T∞.")
        form.addRow("Температура среды T∞:", self.tinf_spin)

        self.tsurf_spin = QDoubleSpinBox()
        self.tsurf_spin.setRange(-273.0, 5000.0)
        self.tsurf_spin.setDecimals(1)
        self.tsurf_spin.setValue(float(T_surface_hint)
                                 if T_surface_hint is not None else 80.0)
        self.tsurf_spin.setSuffix(" °C")
        self.tsurf_spin.setToolTip(
            "Оценка средней температуры поверхности (для свойств плёнки\n"
            "и оценки потока Q). После расчёта можно уточнить.")
        form.addRow("Оценка T поверхности:", self.tsurf_spin)

        self.apply_check = QCheckBox(
            "Назначить рассчитанный h как конвекцию на все грани")
        self.apply_check.setChecked(True)
        form.addRow("", self.apply_check)

        self.orient_check = QCheckBox(
            "Учитывать ориентацию граней (наветр.×1, бок.×0.7, подветр.×0.5)")
        self.orient_check.setChecked(False)
        form.addRow("", self.orient_check)

        lay.addLayout(form)

        self.report = QTextBrowser()
        self.report.setMinimumHeight(150)
        self.report.setStyleSheet("font-family: monospace; font-size: 11px;")
        lay.addWidget(self.report)

        btns = QHBoxLayout()
        self.calc_btn = QPushButton("Рассчитать (предпросмотр)")
        self.calc_btn.clicked.connect(self._preview)
        btns.addWidget(self.calc_btn)
        lay.addLayout(btns)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._problem = parent.problem if parent is not None else None

    def _preview(self):
        """Посчитать и показать отчёт без изменения задачи."""
        if self._problem is None or self._problem.nodes is None:
            self.report.setPlainText("Сначала постройте сетку.")
            return
        p = self.params()
        try:
            res = self._cv.analyze_forced_convection(
                self._problem, speed=p["speed"], direction=p["direction"],
                shape=p["shape"], T_inf=p["T_inf"], T_surface=p["T_surface"])
            self.report.setPlainText(res.report_text())
        except Exception as exc:
            self.report.setPlainText(f"Ошибка: {exc}")

    def params(self) -> dict:
        return {
            "speed":     float(self.speed_spin.value()),
            "direction": self.DIRECTIONS[self.dir_combo.currentIndex()][1],
            "shape":     self._shapes[self.shape_combo.currentIndex()][0],
            "T_inf":     float(self.tinf_spin.value()),
            "T_surface": float(self.tsurf_spin.value()),
            "apply":     bool(self.apply_check.isChecked()),
            "orient":    bool(self.orient_check.isChecked()),
        }
