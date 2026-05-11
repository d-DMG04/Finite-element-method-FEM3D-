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
                             QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                             QSpinBox, QTabWidget, QTextBrowser, QVBoxLayout,
                             QWidget)

from fem3d import BoxGeometry


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
        form.addRow("Плотность Q₀:", self.q_spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def values(self) -> Tuple[float, float, float, float, float]:
        return (self.cx_spin.value(), self.cy_spin.value(),
                self.cz_spin.value(), self.r_spin.value(),
                self.q_spin.value())
