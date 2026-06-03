# -*- coding: utf-8 -*-
"""
gui.dialogs — диалоги «Настройки» и «Справка», а также диалоги добавления
локальных источников.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem,
                             QMessageBox, QPushButton, QScrollArea, QSlider,
                             QSpinBox, QStackedWidget, QTabWidget,
                             QTextBrowser, QToolButton, QVBoxLayout, QWidget)

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
        theme_hint.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
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
        browser.setStyleSheet("background-color: #1f2228; color: #dcdee2;")
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
        hint.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
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
        hint.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
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
        info.setStyleSheet("color: #9aa0a6;")
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
        self.shape_desc.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
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
        self.density_label.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
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
        self.import_label.setStyleSheet("color: #9aa0a6;")
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
        hint.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
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
        info.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
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
    """Понятный диалог граничных условий с физическими терминами.

    Для каждой грани:
      • выбор типа из 5 физических вариантов с пояснением;
      • поля параметров, скрываются/показываются в зависимости от типа;
      • пресеты конвекции для быстрого подбора α и T_среды;
      • подсказки с типичными значениями и формулами.
    """

    # Типы условий с физическими названиями (для GUI).
    BC_KIND_NONE       = "none"
    BC_KIND_DIRICHLET  = "dirichlet"     # заданная температура
    BC_KIND_INSULATED  = "insulated"     # изоляция (Нейман с q=0)
    BC_KIND_HEAT_FLUX  = "heat_flux"     # тепловой поток (Нейман с q≠0)
    BC_KIND_CONVECTION = "convection"    # конвекция (Робен)
    BC_KIND_RADIATION  = "radiation"     # излучение (Стефан-Больцман)

    # Описания каждого типа — для подсказки.
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

    def __init__(self, problem, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Граничные условия")
        self.resize(720, 560)
        from fem3d import (FACE_NAMES, BC_NONE, BC_DIRICHLET, BC_NEUMANN,
                            BC_ROBIN, BC_RADIATION, HEATING_TEMPLATES,
                            CONVECTION_PRESETS)

        self._problem = problem
        self._bcs = {fid: BoundaryCondition(
            type=problem.bcs[fid].type, T0=problem.bcs[fid].T0,
            q0=problem.bcs[fid].q0, alpha=problem.bcs[fid].alpha,
            T_inf=problem.bcs[fid].T_inf,
            emissivity=getattr(problem.bcs[fid], 'emissivity', 0.85))
            for fid in range(6)}
        self._face_widgets = {}   # fid → словарь с виджетами

        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        # ---- Шапка с шаблонами и пресетами ----
        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Шаблон сценария:</b>"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(280)
        self.template_combo.addItem("— Применить шаблон —", None)
        for label, factory in HEATING_TEMPLATES:
            self.template_combo.addItem(label, factory)
        self.template_combo.currentIndexChanged.connect(self._on_template)
        top.addWidget(self.template_combo, 1)
        outer.addLayout(top)

        # ---- Обдув основной фигуры потоком воздуха ----
        outer.addWidget(self._build_air_flow_group())

        # ---- Список карточек граней ----
        from PyQt5.QtWidgets import QScrollArea, QStackedWidget
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        list_w = QWidget()
        list_lay = QVBoxLayout(list_w)
        list_lay.setContentsMargins(0, 0, 0, 0); list_lay.setSpacing(6)
        for fid in range(6):
            list_lay.addWidget(self._build_face_card(fid, FACE_NAMES[fid],
                                                      CONVECTION_PRESETS))
        list_lay.addStretch(1)
        scroll.setWidget(list_w)
        outer.addWidget(scroll, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def _build_face_card(self, fid: int, face_name: str,
                          convection_presets) -> QFrame:
        """Карточка одной грани со всеми параметрами."""
        from PyQt5.QtWidgets import QFrame, QStackedWidget
        card = QFrame(); card.setObjectName("Card")
        outer = QVBoxLayout(card); outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        # Заголовок: имя грани + выпадающий список типа.
        head = QHBoxLayout()
        head.addWidget(QLabel(f"<b>{face_name}</b>"))
        head.addStretch(0)
        head.addWidget(QLabel("Тип условия:"))
        kind_combo = QComboBox()
        kind_combo.setMinimumWidth(220)
        for kind in (self.BC_KIND_NONE, self.BC_KIND_DIRICHLET,
                     self.BC_KIND_INSULATED, self.BC_KIND_HEAT_FLUX,
                     self.BC_KIND_CONVECTION, self.BC_KIND_RADIATION):
            kind_combo.addItem(self.BC_KIND_INFO[kind][0], kind)
        # Установим текущий тип на основе bc.
        kind_combo.setCurrentIndex(
            list(self.BC_KIND_INFO.keys()).index(self._bc_to_kind(self._bcs[fid])))
        head.addWidget(kind_combo)
        outer.addLayout(head)

        # Подсказка.
        hint = QLabel("")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
        outer.addWidget(hint)

        # Stacked widget с параметрами под каждый тип.
        stack = QStackedWidget()
        # Индекс совпадает с порядком в BC_KIND_INFO.

        # 0: None — пустой.
        stack.addWidget(QWidget())
        # 1: Dirichlet — T₀.
        w_d = QWidget(); fl_d = QFormLayout(w_d); fl_d.setContentsMargins(0, 0, 0, 0)
        sp_T0 = QDoubleSpinBox(); sp_T0.setRange(-273, 5000); sp_T0.setSuffix(" °C")
        sp_T0.setValue(self._bcs[fid].T0); sp_T0.setDecimals(2)
        sp_T0.setToolTip("Температура в градусах Цельсия.\n"
                          "Диапазон: −273 ... +5000 °C")
        fl_d.addRow("Температура T₀:", sp_T0)
        stack.addWidget(w_d)
        # 2: Insulated — нет параметров.
        w_ins = QWidget(); fl_ins = QFormLayout(w_ins); fl_ins.setContentsMargins(0, 0, 0, 0)
        fl_ins.addRow(QLabel("<i>Параметров нет (∂T/∂n = 0)</i>"))
        stack.addWidget(w_ins)
        # 3: Heat flux — q.
        w_q = QWidget(); fl_q = QFormLayout(w_q); fl_q.setContentsMargins(0, 0, 0, 0)
        sp_q = QDoubleSpinBox(); sp_q.setRange(-1e9, 1e9); sp_q.setSuffix(" Вт/м²")
        sp_q.setValue(self._bcs[fid].q0 if self._bcs[fid].q0 != 0 else 1000.0)
        sp_q.setDecimals(2)
        sp_q.setToolTip("Плотность теплового потока через грань.\n"
                         "Положительное q — тело НАГРЕВАЕТСЯ, "
                         "отрицательное — охлаждается.\n"
                         "Типичные значения [Вт/м²]:\n"
                         "  Солнечный поток ≈ 800–1000\n"
                         "  Электрическая плита ≈ 10 000–50 000\n"
                         "  Импульсный лазер ≈ 10⁵–10⁸")
        fl_q.addRow("Поток q  (+нагрев, −отвод):", sp_q)
        stack.addWidget(w_q)
        # 4: Convection — пресет + α + T∞.
        w_c = QWidget(); fl_c = QFormLayout(w_c); fl_c.setContentsMargins(0, 0, 0, 0)
        preset_combo = QComboBox()
        for label, alpha_v, tinf_v in convection_presets:
            preset_combo.addItem(label, (alpha_v, tinf_v))
        preset_combo.setToolTip("Готовые значения α и T∞ для типичных сред.")
        fl_c.addRow("Пресет:", preset_combo)
        sp_alpha = QDoubleSpinBox(); sp_alpha.setRange(0, 1e6)
        sp_alpha.setSuffix(" Вт/(м²·К)"); sp_alpha.setDecimals(2)
        sp_alpha.setValue(self._bcs[fid].alpha
                          if self._bcs[fid].alpha > 0 else 10.0)
        sp_alpha.setToolTip(
            "Коэффициент теплоотдачи α.\n"
            "Закон Ньютона: q = α (T_поверхности − T∞).\n"
            "Типичные значения [Вт/(м²·К)]:\n"
            "  Воздух (свободная конвекция) ≈ 5–25\n"
            "  Воздух с вентилятором ≈ 25–250\n"
            "  Вода (свободная) ≈ 500–1000\n"
            "  Вода с насосом ≈ 1000–15000\n"
            "  Кипение ≈ 2500–25000\n"
            "  Конденсация пара ≈ 5000–100 000")
        fl_c.addRow("Коэф. теплоотдачи α:", sp_alpha)
        sp_tinf = QDoubleSpinBox(); sp_tinf.setRange(-273, 5000)
        sp_tinf.setSuffix(" °C"); sp_tinf.setDecimals(2)
        sp_tinf.setValue(self._bcs[fid].T_inf)
        sp_tinf.setToolTip("Температура окружающей среды (вдали от поверхности).")
        fl_c.addRow("Температура среды T∞:", sp_tinf)

        def _on_preset(_i):
            data = preset_combo.currentData()
            if data and data[0] is not None:
                sp_alpha.setValue(data[0])
                sp_tinf.setValue(data[1])
        preset_combo.currentIndexChanged.connect(_on_preset)
        stack.addWidget(w_c)
        # 5: Radiation — ε + T_окр.
        w_r = QWidget(); fl_r = QFormLayout(w_r); fl_r.setContentsMargins(0, 0, 0, 0)
        sp_eps = QDoubleSpinBox(); sp_eps.setRange(0.0, 1.0); sp_eps.setSingleStep(0.05)
        sp_eps.setDecimals(2)
        sp_eps.setValue(self._bcs[fid].emissivity)
        sp_eps.setToolTip(
            "Степень черноты поверхности ε ∈ [0, 1].\n"
            "  Полированный металл: 0.02 – 0.10\n"
            "  Окислённый/окрашенный: 0.20 – 0.95\n"
            "  Чёрное тело (абсолютное): 1.00")
        fl_r.addRow("Степень черноты ε (0..1):", sp_eps)
        sp_tenv = QDoubleSpinBox(); sp_tenv.setRange(-273, 5000)
        sp_tenv.setSuffix(" °C"); sp_tenv.setDecimals(2)
        sp_tenv.setValue(self._bcs[fid].T_inf)
        sp_tenv.setToolTip("Температура окружающего пространства, °C.\n"
                            "Применяется в формуле Стефана-Больцмана:\n"
                            "q = ε σ (T⁴ − T_окр⁴)")
        fl_r.addRow("Температура окружения T_окр:", sp_tenv)
        stack.addWidget(w_r)

        outer.addWidget(stack)

        def _update_hint_and_stack(_i):
            kind = kind_combo.currentData()
            idx = list(self.BC_KIND_INFO.keys()).index(kind)
            stack.setCurrentIndex(idx)
            info = self.BC_KIND_INFO[kind]
            text = info[1]
            if info[2]:
                text += f"<br><b>{info[2]}</b>"
            hint.setText(text)
        kind_combo.currentIndexChanged.connect(_update_hint_and_stack)
        _update_hint_and_stack(0)

        # Сохраняем виджеты для извлечения значений.
        self._face_widgets[fid] = {
            "kind": kind_combo,
            "T0": sp_T0, "q": sp_q, "alpha": sp_alpha, "tinf": sp_tinf,
            "eps": sp_eps, "tenv": sp_tenv, "preset": preset_combo,
        }
        return card

    def _build_air_flow_group(self):
        """Группа обдува: скорость+направление потока → h на все грани фигуры.

        Характерный размер и площадь берутся из РЕАЛЬНОЙ геометрии фигуры,
        поэтому условие применяется именно к основному телу, а не к
        абстрактной форме."""
        from PyQt5.QtWidgets import QGroupBox
        box = QGroupBox("Обдув фигуры потоком воздуха (вынужденная конвекция)")
        lay = QVBoxLayout(box)

        self.flow_enable = QCheckBox(
            "Задавать конвекцию на гранях через обдув (а не вручную α)")
        self.flow_enable.setChecked(
            bool(getattr(self._problem, "air_flow_enabled", False)))
        lay.addWidget(self.flow_enable)

        row = QHBoxLayout()
        row.addWidget(QLabel("Скорость U:"))
        self.flow_speed = QDoubleSpinBox()
        self.flow_speed.setRange(0.0, 500.0); self.flow_speed.setDecimals(2)
        self.flow_speed.setSuffix(" м/с")
        self.flow_speed.setValue(float(getattr(self._problem,
                                               "air_flow_speed", 0.0)) or 5.0)
        row.addWidget(self.flow_speed)

        row.addWidget(QLabel("Направление:"))
        self.flow_dir = QComboBox()
        self._flow_dirs = [("+X", "+x"), ("−X", "-x"), ("+Y", "+y"),
                           ("−Y", "-y"), ("+Z", "+z"), ("−Z", "-z")]
        for label, _ in self._flow_dirs:
            self.flow_dir.addItem(label)
        cur_dir = getattr(self._problem, "air_flow_direction", "+x")
        for i, (_, code) in enumerate(self._flow_dirs):
            if code == cur_dir:
                self.flow_dir.setCurrentIndex(i)
        row.addWidget(self.flow_dir)

        row.addWidget(QLabel("T∞:"))
        self.flow_tinf = QDoubleSpinBox()
        self.flow_tinf.setRange(-273.0, 2000.0); self.flow_tinf.setDecimals(1)
        self.flow_tinf.setSuffix(" °C")
        self.flow_tinf.setValue(float(getattr(self._problem,
                                              "air_flow_T_inf", 20.0)))
        row.addWidget(self.flow_tinf)
        lay.addLayout(row)

        btn_row = QHBoxLayout()
        self.flow_apply_btn = QPushButton("Рассчитать h и применить ко всем граням")
        self.flow_apply_btn.clicked.connect(self._on_apply_air_flow)
        btn_row.addWidget(self.flow_apply_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self.flow_result = QLabel(
            "Введите скорость и нажмите «Рассчитать h…». "
            "Re, Nu и h считаются по размерам самой фигуры.")
        self.flow_result.setWordWrap(True)
        self.flow_result.setStyleSheet("color:#9aa0a6; font-size: 9pt;")
        lay.addWidget(self.flow_result)
        return box

    def _on_apply_air_flow(self):
        """Посчитать h по обдуву и проставить конвекцию на все грани."""
        from fem3d import convection as cv
        if self._problem.nodes is None:
            self.flow_result.setText("Сначала постройте сетку фигуры.")
            return
        # Временно записываем параметры обдува в задачу для расчёта по фигуре.
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
            res = cv.analyze_problem_air_flow(self._problem, T_surface=T_surface)
        except Exception as exc:
            self.flow_result.setText(f"Ошибка: {exc}")
            return
        if res is None:
            self.flow_result.setText("Задайте скорость > 0.")
            return
        # Проставляем конвекцию α=h, T∞ на все грани (в виджеты).
        idx_conv = list(self.BC_KIND_INFO.keys()).index(self.BC_KIND_CONVECTION)
        for fid in range(6):
            w = self._face_widgets[fid]
            w["kind"].setCurrentIndex(idx_conv)
            w["alpha"].setValue(res.h)
            w["tinf"].setValue(T_inf)
        self.flow_enable.setChecked(True)
        self.flow_result.setText(
            f"Re = {res.Re:.3g},  Nu = {res.Nu:.1f},  "
            f"h = {res.h:.2f} Вт/(м²·К)  ({res.regime}).  "
            f"Назначено на все 6 граней. A_полн = {res.total_area:.4g} м², "
            f"Q ≈ {res.Q_total:.4g} Вт.")

    def air_flow_result(self) -> dict:
        """Параметры обдува для сохранения в задачу (читается вызывающим кодом)."""
        return {
            "enabled": bool(self.flow_enable.isChecked()),
            "speed": float(self.flow_speed.value()),
            "direction": self._flow_dirs[self.flow_dir.currentIndex()][1],
            "T_inf": float(self.flow_tinf.value()),
        }

    def _bc_to_kind(self, bc) -> str:
        from fem3d import (BC_NONE, BC_DIRICHLET, BC_NEUMANN, BC_ROBIN,
                            BC_RADIATION)
        if bc.type == BC_NONE: return self.BC_KIND_NONE
        if bc.type == BC_DIRICHLET: return self.BC_KIND_DIRICHLET
        if bc.type == BC_NEUMANN:
            return self.BC_KIND_INSULATED if abs(bc.q0) < 1e-15 else self.BC_KIND_HEAT_FLUX
        if bc.type == BC_ROBIN: return self.BC_KIND_CONVECTION
        if bc.type == BC_RADIATION: return self.BC_KIND_RADIATION
        return self.BC_KIND_NONE

    def _on_template(self, _idx):
        factory = self.template_combo.currentData()
        if factory is None:
            return
        new_bcs = factory()
        for fid, bc in new_bcs.items():
            w = self._face_widgets[fid]
            kind = self._bc_to_kind(bc)
            # Установить тип.
            idx = list(self.BC_KIND_INFO.keys()).index(kind)
            w["kind"].setCurrentIndex(idx)
            # Установить значения.
            w["T0"].setValue(bc.T0)
            w["q"].setValue(bc.q0)
            w["alpha"].setValue(bc.alpha)
            w["tinf"].setValue(bc.T_inf)
            w["tenv"].setValue(bc.T_inf)
        self.template_combo.blockSignals(True)
        self.template_combo.setCurrentIndex(0)
        self.template_combo.blockSignals(False)

    def result_bcs(self) -> dict:
        from fem3d import (BoundaryCondition, BC_NONE, BC_DIRICHLET,
                            BC_NEUMANN, BC_ROBIN, BC_RADIATION)
        result = {}
        kind_to_bc_type = {
            self.BC_KIND_NONE: BC_NONE,
            self.BC_KIND_DIRICHLET: BC_DIRICHLET,
            self.BC_KIND_INSULATED: BC_NEUMANN,
            self.BC_KIND_HEAT_FLUX: BC_NEUMANN,
            self.BC_KIND_CONVECTION: BC_ROBIN,
            self.BC_KIND_RADIATION: BC_RADIATION,
        }
        for fid in range(6):
            w = self._face_widgets[fid]
            kind = w["kind"].currentData()
            bc_type = kind_to_bc_type[kind]
            bc = BoundaryCondition(type=bc_type)
            if kind == self.BC_KIND_DIRICHLET:
                bc.T0 = w["T0"].value()
            elif kind == self.BC_KIND_INSULATED:
                bc.q0 = 0.0
            elif kind == self.BC_KIND_HEAT_FLUX:
                bc.q0 = w["q"].value()
            elif kind == self.BC_KIND_CONVECTION:
                bc.alpha = w["alpha"].value()
                bc.T_inf = w["tinf"].value()
            elif kind == self.BC_KIND_RADIATION:
                bc.emissivity = w["eps"].value()
                bc.T_inf = w["tenv"].value()
            result[fid] = bc
        return result


# =============================================================================
# Диалог параметров нестационарной задачи.
# =============================================================================

class TransientParamsDialog(QDialog):
    """Параметры нестационарного расчёта: t_end, dt, T_init, n_save."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры нестационарной задачи")
        self.setModal(True)
        self.resize(420, 280)

        layout = QVBoxLayout(self)

        hint = QLabel(
            "Нестационарная теплопроводность:\n"
            "  ρ·c_p · ∂T/∂t = ∇·(λ∇T) + Q\n"
            "Неявная схема Эйлера на каждом шаге Δt.\n"
            "Характерное время: τ ≈ ρ·c_p·L²/λ (тепловая инерция)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9ca0aa; padding: 4px;")
        layout.addWidget(hint)

        form = QFormLayout()

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
            "Рекомендация: Δt ≈ τ/30 для гладкой динамики.\n"
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
            "ОТЛИЧАТЬСЯ от температуры среды T∞ в условиях конвекции.\n"
            "Например: T₀=100 °C при конвекции с T∞=20 °C → остывание.")
        form.addRow("Начальная T₀:", self.t_init_spin)

        self.n_save_spin = QSpinBox()
        self.n_save_spin.setRange(2, 1000)
        self.n_save_spin.setValue(50)
        self.n_save_spin.setToolTip(
            "Число сохранённых снимков для проигрывателя анимации.\n"
            "Снимки распределены равномерно во времени.")
        form.addRow("Снимков для анимации:", self.n_save_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "t_end":  float(self.t_end_spin.value()),
            "dt":     float(self.dt_spin.value()),
            "T_init": float(self.t_init_spin.value()),
            "n_save": int(self.n_save_spin.value()),
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
        hint.setStyleSheet("color: #b8bcc4; font-size: 9pt; padding: 4px;")
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
            card.setStyleSheet(
                "QFrame#Card { border: 1px solid #3c4049; border-radius: 6px; "
                "background: #23262c; } QFrame#Card:hover { border-color: #7a6cf0; }")
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
            d.setStyleSheet("color: #9aa0a6; font-size: 8pt;")
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
        hint.setStyleSheet("color:#9ca0aa; padding:4px;")
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
