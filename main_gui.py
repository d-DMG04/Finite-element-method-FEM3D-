# -*- coding: utf-8 -*-
"""
main_gui.py
===========

Главное приложение программного комплекса МКЭ для трёхмерной
теплопроводности. Реализует графический интерфейс по требованию раздела 1.6.4
ТЗ: четыре функциональные зоны (левая, центр, правая, нижняя) на русском
языке, минимальный размер окна 1200×800 пикселей.

Запуск:
    python main_gui.py

Зависимости: PyQt5, NumPy, matplotlib (опционально meshio для VTU).
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, Optional

# -----------------------------------------------------------------------------
# Импорты PyQt5. Если библиотека не установлена, выводим понятное сообщение.
# -----------------------------------------------------------------------------
try:
    from PyQt5.QtCore import (QObject, QPoint, QRect, QSize, Qt, QThread,
                              pyqtSignal)
    from PyQt5.QtGui import (QBrush, QColor, QFont, QIcon, QPainter, QPainterPath,
                             QPalette, QPen, QPolygon)
    from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox,
                                 QDialog, QDialogButtonBox, QDoubleSpinBox,
                                 QFileDialog, QFormLayout, QFrame,
                                 QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                                 QListWidget, QListWidgetItem, QMainWindow,
                                 QMessageBox, QProgressBar,
                                 QPushButton, QScrollArea, QSizePolicy,
                                 QSpinBox, QSplitter, QStatusBar, QStyleOption,
                                 QStyle, QToolButton, QVBoxLayout, QWidget)
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "\nОшибка импорта PyQt5: {}\n"
        "Установите PyQt5 командой: pip install PyQt5\n".format(exc)
    )
    sys.exit(1)

import numpy as np

# -----------------------------------------------------------------------------
# Импорт пакета fem3d из локального каталога.
# -----------------------------------------------------------------------------
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fem3d import (BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN, MATERIALS,
                   PRESETS, BoundaryCondition, BoxGeometry, BoxPreset,
                   CoreBridge, CoreError, FACE_NAMES, FACE_X_MINUS, FACE_X_PLUS,
                   FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS,
                   Material, PointSource, Problem, SolverInfo, VolumeSource,
                   VOLSRC_BOX, VOLSRC_SPHERE, compute_mesh_info,
                   template_all_convection, template_bottom_heat_top_cool,
                   template_reset)
from fem3d.postprocess import (export_csv, export_report, export_vtu,
                               slice_by_plane)


# =============================================================================
# Цветовая схема (раздел 1.6.4 ТЗ + раздел 3.3.7 ПЗ).
# =============================================================================

COLOR_BG_DARK   = "#1f2228"
COLOR_PANEL     = "#2a2e36"
COLOR_TEXT      = "#dcdee2"
COLOR_TEXT_DIM  = "#9aa0a6"
COLOR_ACCENT    = "#7a6cf0"   # фиолетовый — акценты
COLOR_RUN       = "#3aa55a"   # зелёный — главная кнопка
COLOR_DIRICHLET = "#d05050"   # красный
COLOR_ROBIN     = "#3a78d0"   # синий
COLOR_NEUMANN   = "#7a7e88"   # серый
COLOR_NONE      = "#3c4049"   # тёмный нейтральный

BC_COLORS = {
    BC_NONE:      COLOR_NONE,
    BC_DIRICHLET: COLOR_DIRICHLET,
    BC_NEUMANN:   COLOR_NEUMANN,
    BC_ROBIN:     COLOR_ROBIN,
}


# =============================================================================
# Глобальный stylesheet — тёмная тема (раздел 3.3.8 ПЗ).
# =============================================================================

STYLESHEET = f"""
QWidget {{
    background-color: {COLOR_BG_DARK};
    color: {COLOR_TEXT};
    font-family: "Segoe UI", "Helvetica", "Arial", sans-serif;
    font-size: 10pt;
}}
QFrame#Panel {{
    background-color: {COLOR_PANEL};
    border-radius: 6px;
}}
QGroupBox {{
    border: 1px solid #3c4049;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 14px;
    color: {COLOR_TEXT_DIM};
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
}}
QLabel {{
    background-color: transparent;
}}
QPushButton {{
    background-color: #3c4049;
    color: {COLOR_TEXT};
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
}}
QPushButton:hover {{
    background-color: #4a4f5a;
}}
QPushButton:pressed {{
    background-color: #2f333b;
}}
QPushButton#AccentButton {{
    background-color: {COLOR_ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton#AccentButton:hover {{
    background-color: #8b7eff;
}}
QPushButton#RunButton {{
    background-color: {COLOR_RUN};
    color: white;
    font-weight: 600;
    padding: 8px 18px;
}}
QPushButton#RunButton:hover {{
    background-color: #4ec070;
}}
QPushButton#RunButton:disabled {{
    background-color: #2e4a37;
    color: #6f8a76;
}}
QComboBox, QDoubleSpinBox, QSpinBox {{
    background-color: #1a1d22;
    color: {COLOR_TEXT};
    border: 1px solid #3c4049;
    border-radius: 3px;
    padding: 3px 6px;
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
    border-color: #5a606b;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: #1a1d22;
    color: {COLOR_TEXT};
    selection-background-color: {COLOR_ACCENT};
}}
QProgressBar {{
    background-color: #1a1d22;
    border: 1px solid #3c4049;
    border-radius: 3px;
    text-align: center;
    color: {COLOR_TEXT};
}}
QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 2px;
}}
QScrollArea {{
    border: none;
}}
QSplitter::handle {{
    background-color: #1a1d22;
    width: 3px;
}}
QStatusBar {{
    background-color: #1a1d22;
    color: {COLOR_TEXT_DIM};
}}
"""


# =============================================================================
# Карточка одной грани в правой панели.
# =============================================================================

class FaceCard(QFrame):
    """
    Виджет одной грани: тип ГУ + параметры. Цветной индикатор слева.
    """
    changed = pyqtSignal(int)  # face_id

    def __init__(self, face_id: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.face_id = face_id
        self.bc = BoundaryCondition()
        self.setObjectName("Panel")
        self.setProperty("class", "FaceCard")
        self._build_ui()
        self._update_indicator()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 8, 0)
        outer.setSpacing(8)

        # Цветной индикатор слева.
        self.indicator = QFrame()
        self.indicator.setFixedWidth(6)
        self.indicator.setStyleSheet(f"background-color: {COLOR_NONE};"
                                     "border-top-left-radius: 6px;"
                                     "border-bottom-left-radius: 6px;")
        outer.addWidget(self.indicator)

        body = QVBoxLayout()
        body.setContentsMargins(8, 8, 8, 8)
        body.setSpacing(4)

        # Заголовок: имя грани.
        self.title_label = QLabel(f"<b>{FACE_NAMES[self.face_id]}</b>")
        body.addWidget(self.title_label)

        # Описание текущего ГУ.
        self.desc_label = QLabel(self.bc.description())
        self.desc_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 9pt;")
        body.addWidget(self.desc_label)

        # Тип ГУ.
        type_row = QHBoxLayout()
        type_row.setSpacing(6)
        type_row.addWidget(QLabel("Тип:"))
        self.type_combo = QComboBox()
        self.type_combo.addItem("Не задано",        BC_NONE)
        self.type_combo.addItem("Нагрев (Дирихле)", BC_DIRICHLET)
        self.type_combo.addItem("Изоляция (Нейман)", BC_NEUMANN)
        self.type_combo.addItem("Конвекция (Робен)", BC_ROBIN)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self.type_combo, 1)
        body.addLayout(type_row)

        # Параметры (создаются всегда, видны по типу).
        self.params_widget = QWidget()
        params_layout = QGridLayout(self.params_widget)
        params_layout.setContentsMargins(0, 4, 0, 0)
        params_layout.setHorizontalSpacing(6)
        params_layout.setVerticalSpacing(4)

        self.t0_label = QLabel("T₀, °C:")
        self.t0_spin = QDoubleSpinBox()
        self.t0_spin.setRange(-273.0, 5000.0)
        self.t0_spin.setDecimals(2)
        self.t0_spin.setValue(0.0)
        self.t0_spin.valueChanged.connect(self._collect_and_emit)
        params_layout.addWidget(self.t0_label, 0, 0)
        params_layout.addWidget(self.t0_spin,  0, 1)

        self.q0_label = QLabel("q, Вт/м²:")
        self.q0_spin = QDoubleSpinBox()
        self.q0_spin.setRange(-1e9, 1e9)
        self.q0_spin.setDecimals(2)
        self.q0_spin.setValue(0.0)
        self.q0_spin.valueChanged.connect(self._collect_and_emit)
        params_layout.addWidget(self.q0_label, 1, 0)
        params_layout.addWidget(self.q0_spin,  1, 1)

        self.alpha_label = QLabel("α, Вт/(м²·К):")
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1e6)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setValue(25.0)
        self.alpha_spin.valueChanged.connect(self._collect_and_emit)
        params_layout.addWidget(self.alpha_label, 2, 0)
        params_layout.addWidget(self.alpha_spin,  2, 1)

        self.tinf_label = QLabel("T∞, °C:")
        self.tinf_spin = QDoubleSpinBox()
        self.tinf_spin.setRange(-273.0, 5000.0)
        self.tinf_spin.setDecimals(2)
        self.tinf_spin.setValue(20.0)
        self.tinf_spin.valueChanged.connect(self._collect_and_emit)
        params_layout.addWidget(self.tinf_label, 3, 0)
        params_layout.addWidget(self.tinf_spin,  3, 1)

        body.addWidget(self.params_widget)
        outer.addLayout(body, 1)
        self._refresh_visibility()

    # -------------------------------------------------------------------------
    def _on_type_changed(self, _idx: int) -> None:
        self._refresh_visibility()
        self._collect_and_emit()

    def _refresh_visibility(self) -> None:
        bc_type = self.type_combo.currentData()
        # T0 показываем для Дирихле; q0 — для Неймана; alpha и Tinf — для Робена.
        for w in (self.t0_label, self.t0_spin):
            w.setVisible(bc_type == BC_DIRICHLET)
        for w in (self.q0_label, self.q0_spin):
            w.setVisible(bc_type == BC_NEUMANN)
        for w in (self.alpha_label, self.alpha_spin,
                  self.tinf_label, self.tinf_spin):
            w.setVisible(bc_type == BC_ROBIN)

    def _collect_and_emit(self) -> None:
        bc_type = self.type_combo.currentData()
        self.bc = BoundaryCondition(
            type=int(bc_type),
            T0=self.t0_spin.value(),
            q0=self.q0_spin.value(),
            alpha=self.alpha_spin.value(),
            T_inf=self.tinf_spin.value(),
        )
        self.desc_label.setText(self.bc.description())
        self._update_indicator()
        self.changed.emit(self.face_id)

    def _update_indicator(self) -> None:
        color = BC_COLORS.get(self.bc.type, COLOR_NONE)
        self.indicator.setStyleSheet(
            f"background-color: {color};"
            "border-top-left-radius: 6px;"
            "border-bottom-left-radius: 6px;"
        )

    def set_bc(self, bc: BoundaryCondition) -> None:
        """Программная установка ГУ (например, из шаблона)."""
        # Блокируем сигналы, чтобы не сгенерировать лишние changed.
        self.type_combo.blockSignals(True)
        self.t0_spin.blockSignals(True)
        self.q0_spin.blockSignals(True)
        self.alpha_spin.blockSignals(True)
        self.tinf_spin.blockSignals(True)

        # Устанавливаем тип в комбо.
        for i in range(self.type_combo.count()):
            if int(self.type_combo.itemData(i)) == int(bc.type):
                self.type_combo.setCurrentIndex(i)
                break
        self.t0_spin.setValue(bc.T0)
        self.q0_spin.setValue(bc.q0)
        self.alpha_spin.setValue(bc.alpha)
        self.tinf_spin.setValue(bc.T_inf)

        self.type_combo.blockSignals(False)
        self.t0_spin.blockSignals(False)
        self.q0_spin.blockSignals(False)
        self.alpha_spin.blockSignals(False)
        self.tinf_spin.blockSignals(False)

        self.bc = bc
        self.desc_label.setText(bc.description())
        self._refresh_visibility()
        self._update_indicator()


# =============================================================================
# Поток-воркер для выполнения расчёта.
# =============================================================================

class SolverWorker(QObject):
    """
    Запускает полный цикл расчёта в отдельном потоке (раздел 3.3.6 ПЗ).
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # SolverInfo + Problem (через self.problem)
    error = pyqtSignal(str)

    def __init__(self, problem: Problem, tol: float, max_iter: int) -> None:
        super().__init__()
        self.problem = problem
        self.tol = tol
        self.max_iter = max_iter

    def run(self) -> None:
        try:
            self.progress.emit("Сборка матрицы...")
            with CoreBridge() as bridge:
                self.problem.build_mesh_in_core(bridge)
                self.progress.emit(
                    f"Решение СЛАУ ({self.problem.nodes.shape[0]} узлов)..."
                )
                info = self.problem.solve(bridge, tol=self.tol,
                                          max_iter=self.max_iter)
                self.progress.emit("Готово.")
                self.finished.emit(info)
        except CoreError as exc:
            self.error.emit(f"Ошибка ядра: {exc}")
        except Exception as exc:  # pragma: no cover
            tb = traceback.format_exc()
            self.error.emit(f"Внутренняя ошибка:\n{exc}\n\n{tb}")


# =============================================================================
# Центральная зона — 3D-вид и температурная карта (на QPainter).
# =============================================================================

class CentralView(QWidget):
    """
    Простой 3D-просмотр модели с цветовой подсветкой граней и режимом
    температурной карты на сечении. Реализован вручную через QPainter
    с проекционными матрицами — без OpenGL/VTK, что соответствует решению
    раздела 3.3.3 ПЗ.
    """
    MODE_MODEL       = "Модель"
    MODE_TEMPERATURE = "Температура"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(500, 400)
        self.setObjectName("Panel")
        self.setMouseTracking(True)

        self._geometry: Optional[BoxGeometry] = None
        self._bcs: Dict[int, BoundaryCondition] = {f: BoundaryCondition() for f in range(6)}
        self._problem: Optional[Problem] = None
        self._mode = self.MODE_MODEL
        self._slice_axis = "z"
        self._slice_pos_norm = 0.5  # положение сечения в долях габарита

        # Параметры камеры.
        self._yaw = -0.6     # поворот вокруг вертикали
        self._pitch = 0.5    # наклон
        self._dist = 2.5
        self._last_mouse: Optional[QPoint] = None

        # Лейбл подсказки.
        self._hint = ""

    # -------------------------------------------------------------------------
    def set_geometry(self, geom: BoxGeometry) -> None:
        self._geometry = geom
        self.update()

    def set_bcs(self, bcs: Dict[int, BoundaryCondition]) -> None:
        self._bcs = bcs
        self.update()

    def set_problem(self, problem: Optional[Problem]) -> None:
        self._problem = problem
        if problem is not None and problem.T is not None:
            self._mode = self.MODE_TEMPERATURE
        else:
            self._mode = self.MODE_MODEL
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.update()

    def set_slice_axis(self, axis: str) -> None:
        self._slice_axis = axis.lower()
        self.update()

    def set_slice_position(self, pos_norm: float) -> None:
        self._slice_pos_norm = max(0.0, min(1.0, pos_norm))
        self.update()

    # =========================================================================
    # Простейшая 3D-математика.
    # =========================================================================

    def _project(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """
        Проекция точки в координаты экрана (px, py) и глубину для сортировки.
        Камера смотрит на центр габарита из расстояния dist.
        """
        if self._geometry is None:
            return (0.0, 0.0, 0.0)
        g = self._geometry
        cx, cy, cz = g.Lx / 2, g.Ly / 2, g.Lz / 2
        # Перевод в систему координат камеры.
        x0, y0, z0 = x - cx, y - cy, z - cz

        # Поворот вокруг вертикали (yaw).
        cy_, sy_ = np.cos(self._yaw), np.sin(self._yaw)
        x1 =  cy_ * x0 + sy_ * y0
        y1 = -sy_ * x0 + cy_ * y0
        z1 = z0

        # Поворот вокруг горизонтальной оси (pitch).
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        x2 = x1
        y2 =  cp * y1 + sp * z1
        z2 = -sp * y1 + cp * z1

        # Перспективная проекция.
        scale = max(g.Lx, g.Ly, g.Lz)
        focal = self._dist * scale
        z_cam = focal - x2  # «вглубь» по +x камеры
        if z_cam <= 1e-6:
            z_cam = 1e-6
        f = focal / z_cam

        w, h = self.width(), self.height()
        px = w / 2 + f * y2 * 0.5 * min(w, h) / scale
        py = h / 2 - f * z2 * 0.5 * min(w, h) / scale
        return px, py, z_cam

    # =========================================================================
    # Покраска.
    # =========================================================================

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(COLOR_PANEL))

        if self._geometry is None:
            p.setPen(QColor(COLOR_TEXT_DIM))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Сгенерируйте сетку для отображения модели.")
            return

        # 1) Подложка: рисуем 6 граней параллелепипеда с подсветкой по ГУ.
        self._paint_box(p)

        # 2) Если есть результаты — нарисовать срез температуры.
        if (self._mode == self.MODE_TEMPERATURE and self._problem is not None
                and self._problem.T is not None):
            self._paint_temperature_slice(p)
            self._paint_legend(p)
            self._paint_hot_spot(p)

        # 3) Подписи режима и сечения.
        self._paint_overlay(p)

    # -------------------------------------------------------------------------
    def _paint_box(self, p: QPainter) -> None:
        g = self._geometry
        # 8 вершин куба.
        verts = np.array([
            [0,    0,    0   ],
            [g.Lx, 0,    0   ],
            [g.Lx, g.Ly, 0   ],
            [0,    g.Ly, 0   ],
            [0,    0,    g.Lz],
            [g.Lx, 0,    g.Lz],
            [g.Lx, g.Ly, g.Lz],
            [0,    g.Ly, g.Lz],
        ])
        # 6 граней (4 вершины каждая) + face_id.
        faces = [
            ([0, 3, 7, 4], FACE_X_MINUS),  # x = 0
            ([1, 2, 6, 5], FACE_X_PLUS),   # x = Lx
            ([0, 1, 5, 4], FACE_Y_MINUS),  # y = 0
            ([2, 3, 7, 6], FACE_Y_PLUS),   # y = Ly
            ([0, 1, 2, 3], FACE_Z_MINUS),  # z = 0
            ([4, 5, 6, 7], FACE_Z_PLUS),   # z = Lz
        ]

        # Сортируем по глубине (z_cam в центре грани) — рисуем дальние первыми.
        face_data = []
        for vidx, fid in faces:
            poly = []
            zsum = 0.0
            for v in vidx:
                x, y, z = verts[v]
                px, py, zc = self._project(x, y, z)
                poly.append((px, py))
                zsum += zc
            zavg = zsum / len(vidx)
            face_data.append((zavg, poly, fid))
        face_data.sort(key=lambda t: -t[0])  # дальние first

        # Пропускаем самую переднюю грань (это срезанная грань — для просмотра внутрь),
        # если в режиме температуры со срезом по соответствующей оси.
        # Простая стратегия: всегда рисуем все грани полупрозрачно.

        for zavg, poly, fid in face_data:
            color = QColor(BC_COLORS.get(self._bcs[fid].type, COLOR_NONE))
            color.setAlphaF(0.55)
            p.setBrush(QBrush(color))
            p.setPen(QPen(QColor(COLOR_TEXT_DIM), 1))
            qpoly = QPolygon([QPoint(int(round(x)), int(round(y))) for x, y in poly])
            p.drawPolygon(qpoly)

    # -------------------------------------------------------------------------
    def _paint_temperature_slice(self, p: QPainter) -> None:
        """Рисует точечную цветовую карту узлов, попавших в полосу среза."""
        prob = self._problem
        if prob is None or prob.nodes is None or prob.T is None:
            return
        axis_idx = {"x": 0, "y": 1, "z": 2}[self._slice_axis]
        coords = prob.nodes[:, axis_idx]
        cmin, cmax = float(coords.min()), float(coords.max())
        slice_pos = cmin + self._slice_pos_norm * (cmax - cmin)
        # Толщина полосы — 5 % габарита.
        tol = max(1e-9, 0.05 * (cmax - cmin))
        mask = np.abs(coords - slice_pos) <= tol
        if not np.any(mask):
            return

        Tmin, Tmax = prob.temperature_range()
        nodes_sel = prob.nodes[mask]
        T_sel = prob.T[mask]

        for i in range(nodes_sel.shape[0]):
            x, y, z = nodes_sel[i]
            px, py, _ = self._project(x, y, z)
            color = self._color_for_T(T_sel[i], Tmin, Tmax)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawEllipse(QPoint(int(round(px)), int(round(py))), 3, 3)

    @staticmethod
    def _color_for_T(T: float, Tmin: float, Tmax: float) -> QColor:
        """Цветовая шкала: синий → бирюзовый → жёлтый → красный."""
        if Tmax - Tmin < 1e-12:
            t = 0.5
        else:
            t = (T - Tmin) / (Tmax - Tmin)
        t = max(0.0, min(1.0, t))
        # Простая 4-точечная карта.
        if t < 1/3:
            r, g, b = 0.1 + (0.2 - 0.1) * (3 * t),     0.4 + 0.4 * (3 * t),       0.8
        elif t < 2/3:
            tt = 3 * (t - 1/3)
            r, g, b = 0.2 + 0.8 * tt,                   0.8,                       0.8 - 0.5 * tt
        else:
            tt = 3 * (t - 2/3)
            r, g, b = 1.0,                              0.8 - 0.6 * tt,            0.3 - 0.3 * tt
        return QColor(int(r * 255), int(g * 255), int(b * 255))

    # -------------------------------------------------------------------------
    def _paint_legend(self, p: QPainter) -> None:
        prob = self._problem
        if prob is None or prob.T is None:
            return
        Tmin, Tmax = prob.temperature_range()
        rect = QRect(self.width() - 50, 30, 24, 220)
        # Градиент.
        for i in range(rect.height()):
            t = 1.0 - i / max(1, rect.height() - 1)
            T_val = Tmin + t * (Tmax - Tmin)
            p.setPen(self._color_for_T(T_val, Tmin, Tmax))
            p.drawLine(rect.left(), rect.top() + i, rect.right(), rect.top() + i)
        p.setPen(QColor(COLOR_TEXT_DIM))
        p.drawRect(rect)
        p.setPen(QColor(COLOR_TEXT))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(rect.left() - 50, rect.top() + 8, f"{Tmax:.1f} °C")
        p.drawText(rect.left() - 50, rect.bottom() + 4, f"{Tmin:.1f} °C")

    def _paint_hot_spot(self, p: QPainter) -> None:
        if self._problem is None:
            return
        hs = self._problem.hot_spot()
        if hs is None:
            return
        _, x, y, z = hs
        px, py, _ = self._project(x, y, z)
        p.setBrush(QColor("#ffd24a"))
        p.setPen(QPen(QColor("#000"), 1))
        p.drawEllipse(QPoint(int(round(px)), int(round(py))), 6, 6)
        p.setPen(QColor("#ffd24a"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(int(round(px)) + 10, int(round(py)) + 4, "T_max")

    def _paint_overlay(self, p: QPainter) -> None:
        p.setPen(QColor(COLOR_TEXT_DIM))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(10, 18, f"Режим: {self._mode}    Сечение: {self._slice_axis.upper()}    "
                           f"Перетащите мышью для вращения, колесо — масштаб.")

    # =========================================================================
    # Взаимодействие.
    # =========================================================================

    def mousePressEvent(self, e) -> None:
        self._last_mouse = e.pos()

    def mouseMoveEvent(self, e) -> None:
        if self._last_mouse is None:
            return
        dx = e.x() - self._last_mouse.x()
        dy = e.y() - self._last_mouse.y()
        self._yaw += dx * 0.01
        self._pitch = max(-1.4, min(1.4, self._pitch + dy * 0.01))
        self._last_mouse = e.pos()
        self.update()

    def mouseReleaseEvent(self, _e) -> None:
        self._last_mouse = None

    def wheelEvent(self, e) -> None:
        delta = e.angleDelta().y() / 240.0
        self._dist *= np.exp(-delta * 0.2)
        self._dist = max(0.5, min(8.0, self._dist))
        self.update()


# =============================================================================
# Диалоги добавления локальных источников.
# =============================================================================

class PointSourceDialog(QDialog):
    """Диалог ввода параметров точечного источника: координата + мощность."""

    def __init__(self, geom: BoxGeometry, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Добавить точечный источник")
        self.setMinimumWidth(320)

        form = QFormLayout(self)

        # Координаты (по умолчанию — центр габарита).
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(0.0, geom.Lx); self.x_spin.setDecimals(4)
        self.x_spin.setValue(geom.Lx / 2)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(0.0, geom.Ly); self.y_spin.setDecimals(4)
        self.y_spin.setValue(geom.Ly / 2)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0.0, geom.Lz); self.z_spin.setDecimals(4)
        self.z_spin.setValue(geom.Lz / 2)
        form.addRow("x, м:", self.x_spin)
        form.addRow("y, м:", self.y_spin)
        form.addRow("z, м:", self.z_spin)

        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(-1.0e6, 1.0e6); self.p_spin.setDecimals(2)
        self.p_spin.setValue(10.0)
        self.p_spin.setSuffix(" Вт")
        form.addRow("Мощность P:", self.p_spin)

        hint = QLabel("Будет привязан к ближайшему узлу сетки. "
                      "Положительное P — нагрев, отрицательное — отвод.")
        hint.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 9pt;")
        hint.setWordWrap(True)
        form.addRow(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple:
        return (self.x_spin.value(), self.y_spin.value(),
                self.z_spin.value(), self.p_spin.value())


class VolumeSourceDialog(QDialog):
    """Диалог ввода параметров объёмного источника в сферической подобласти."""

    def __init__(self, geom: BoxGeometry, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Добавить объёмный источник (сфера)")
        self.setMinimumWidth(320)

        form = QFormLayout(self)

        self.cx_spin = QDoubleSpinBox()
        self.cx_spin.setRange(0.0, geom.Lx); self.cx_spin.setDecimals(4)
        self.cx_spin.setValue(geom.Lx / 2)
        self.cy_spin = QDoubleSpinBox()
        self.cy_spin.setRange(0.0, geom.Ly); self.cy_spin.setDecimals(4)
        self.cy_spin.setValue(geom.Ly / 2)
        self.cz_spin = QDoubleSpinBox()
        self.cz_spin.setRange(0.0, geom.Lz); self.cz_spin.setDecimals(4)
        self.cz_spin.setValue(geom.Lz / 2)
        form.addRow("Центр x, м:", self.cx_spin)
        form.addRow("Центр y, м:", self.cy_spin)
        form.addRow("Центр z, м:", self.cz_spin)

        self.r_spin = QDoubleSpinBox()
        self.r_spin.setRange(1e-5, max(geom.Lx, geom.Ly, geom.Lz))
        self.r_spin.setDecimals(4)
        self.r_spin.setValue(min(geom.Lx, geom.Ly, geom.Lz) / 5)
        form.addRow("Радиус, м:", self.r_spin)

        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(-1.0e10, 1.0e10); self.q_spin.setDecimals(0)
        self.q_spin.setValue(1.0e6)
        self.q_spin.setSuffix(" Вт/м³")
        form.addRow("Плотность Q₀:", self.q_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple:
        return (self.cx_spin.value(), self.cy_spin.value(),
                self.cz_spin.value(), self.r_spin.value(),
                self.q_spin.value())


# =============================================================================
# Главное окно.
# =============================================================================

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Программный комплекс МКЭ — расчёт теплопроводности")
        self.setMinimumSize(1200, 800)

        # Состояние.
        self.problem = Problem()
        self._thread: Optional[QThread] = None
        self._worker: Optional[SolverWorker] = None

        # UI.
        self._build_ui()
        self._sync_to_problem()

    # -------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Верхняя часть: трёхпанельный сплиттер (левая, центр, правая).
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([320, 800, 360])
        root.addWidget(splitter, 1)

        # Нижняя панель.
        root.addWidget(self._build_bottom_panel(), 0)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    # =========================================================================
    # Левая панель — управление моделью.
    # =========================================================================

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

        # Заголовок панели.
        title = QLabel("<b>Управление моделью</b>")
        title.setStyleSheet("font-size: 11pt;")
        layout.addWidget(title)

        # --- Геометрия --------------------------------------------------------
        geom_box = QGroupBox("Геометрия")
        geom_layout = QVBoxLayout(geom_box)
        geom_layout.setSpacing(6)

        # Тип геометрии: параллелепипед или импорт MSH (Ф1.2 ТЗ).
        geom_layout.addWidget(QLabel("Тип:"))
        self.geom_type_combo = QComboBox()
        self.geom_type_combo.addItem("Параллелепипед", "box")
        self.geom_type_combo.addItem("Импорт сетки (MSH)", "msh")
        self.geom_type_combo.currentIndexChanged.connect(self._on_geom_type_changed)
        geom_layout.addWidget(self.geom_type_combo)

        # ===== Контейнер для параметров параллелепипеда =====
        self.box_params_widget = QWidget()
        box_params_layout = QVBoxLayout(self.box_params_widget)
        box_params_layout.setContentsMargins(0, 0, 0, 0)
        box_params_layout.setSpacing(6)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("— Произвольный размер —", None)
        for ps in PRESETS:
            self.preset_combo.addItem(ps.label, ps)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        box_params_layout.addWidget(QLabel("Пресет:"))
        box_params_layout.addWidget(self.preset_combo)

        size_grid = QGridLayout()
        size_grid.setHorizontalSpacing(6)
        size_grid.setVerticalSpacing(4)
        size_grid.addWidget(QLabel("X, м:"), 0, 0)
        size_grid.addWidget(QLabel("Y, м:"), 0, 1)
        size_grid.addWidget(QLabel("Z, м:"), 0, 2)
        self.size_x = QDoubleSpinBox(); self.size_x.setRange(1e-4, 100.0); self.size_x.setDecimals(4); self.size_x.setValue(0.10)
        self.size_y = QDoubleSpinBox(); self.size_y.setRange(1e-4, 100.0); self.size_y.setDecimals(4); self.size_y.setValue(0.10)
        self.size_z = QDoubleSpinBox(); self.size_z.setRange(1e-4, 100.0); self.size_z.setDecimals(4); self.size_z.setValue(0.10)
        size_grid.addWidget(self.size_x, 1, 0)
        size_grid.addWidget(self.size_y, 1, 1)
        size_grid.addWidget(self.size_z, 1, 2)
        box_params_layout.addLayout(size_grid)

        mesh_grid = QGridLayout()
        mesh_grid.setHorizontalSpacing(6)
        mesh_grid.setVerticalSpacing(4)
        mesh_grid.addWidget(QLabel("nx:"), 0, 0)
        mesh_grid.addWidget(QLabel("ny:"), 0, 1)
        mesh_grid.addWidget(QLabel("nz:"), 0, 2)
        self.n_x = QSpinBox(); self.n_x.setRange(2, 200); self.n_x.setValue(15)
        self.n_y = QSpinBox(); self.n_y.setRange(2, 200); self.n_y.setValue(15)
        self.n_z = QSpinBox(); self.n_z.setRange(2, 200); self.n_z.setValue(15)
        mesh_grid.addWidget(self.n_x, 1, 0)
        mesh_grid.addWidget(self.n_y, 1, 1)
        mesh_grid.addWidget(self.n_z, 1, 2)
        box_params_layout.addLayout(mesh_grid)

        geom_layout.addWidget(self.box_params_widget)

        # ===== Контейнер для импорта MSH =====
        self.msh_params_widget = QWidget()
        msh_params_layout = QVBoxLayout(self.msh_params_widget)
        msh_params_layout.setContentsMargins(0, 0, 0, 0)
        msh_params_layout.setSpacing(4)
        self.msh_path_label = QLabel("<i>Файл не выбран</i>")
        self.msh_path_label.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        self.msh_path_label.setWordWrap(True)
        msh_params_layout.addWidget(self.msh_path_label)
        self.btn_browse_msh = QPushButton("Выбрать файл .msh...")
        self.btn_browse_msh.clicked.connect(self._on_browse_msh)
        msh_params_layout.addWidget(self.btn_browse_msh)
        self._msh_path: Optional[str] = None
        geom_layout.addWidget(self.msh_params_widget)
        self.msh_params_widget.setVisible(False)

        # Главная кнопка генерации/загрузки.
        self.gen_button = QPushButton("Сгенерировать сетку")
        self.gen_button.setObjectName("AccentButton")
        self.gen_button.clicked.connect(self._on_generate_mesh)
        geom_layout.addWidget(self.gen_button)

        layout.addWidget(geom_box)

        # --- Материал ---------------------------------------------------------
        mat_box = QGroupBox("Материал")
        mat_layout = QVBoxLayout(mat_box)
        mat_layout.setSpacing(6)

        self.material_combo = QComboBox()
        self.material_combo.addItem("— Произвольный —", None)
        for m in MATERIALS:
            self.material_combo.addItem(f"{m.name}  (λ = {m.lambda_:g})", m)
        # По умолчанию выбираем алюминий.
        for i in range(self.material_combo.count()):
            data = self.material_combo.itemData(i)
            if isinstance(data, Material) and data.name == "Алюминий":
                self.material_combo.setCurrentIndex(i)
                break
        self.material_combo.currentIndexChanged.connect(self._on_material_changed)
        mat_layout.addWidget(self.material_combo)

        param_grid = QGridLayout()
        param_grid.setHorizontalSpacing(6)
        param_grid.setVerticalSpacing(4)
        param_grid.addWidget(QLabel("λ, Вт/(м·К):"), 0, 0)
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(1e-4, 1e5)
        self.lambda_spin.setDecimals(4)
        self.lambda_spin.setValue(237.0)
        self.lambda_spin.valueChanged.connect(self._sync_to_problem)
        param_grid.addWidget(self.lambda_spin, 0, 1)
        param_grid.addWidget(QLabel("Q, Вт/м³:"), 1, 0)
        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(-1e9, 1e9)
        self.q_spin.setDecimals(2)
        self.q_spin.setValue(0.0)
        self.q_spin.valueChanged.connect(self._sync_to_problem)
        param_grid.addWidget(self.q_spin, 1, 1)
        mat_layout.addLayout(param_grid)
        layout.addWidget(mat_box)

        # --- Информация о сетке -----------------------------------------------
        info_box = QGroupBox("Информация о сетке")
        info_layout = QVBoxLayout(info_box)
        self.info_label = QLabel("Сетка не построена.")
        self.info_label.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_box)

        # --- Локальные источники (раздел 3.3.11 ПЗ, рис. 3.3) -----------------
        src_box = QGroupBox("Локальные источники")
        src_layout = QVBoxLayout(src_box)
        src_layout.setSpacing(4)

        self.sources_list = QListWidget()
        self.sources_list.setMaximumHeight(120)
        self.sources_list.setStyleSheet(
            f"QListWidget {{ background-color: #1a1d22; "
            f"border: 1px solid #3c4049; border-radius: 3px; }}")
        src_layout.addWidget(self.sources_list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_add_point = QPushButton("+ Точка")
        self.btn_add_point.clicked.connect(self._on_add_point_source)
        self.btn_add_volume = QPushButton("+ Сфера")
        self.btn_add_volume.clicked.connect(self._on_add_volume_source)
        self.btn_remove_source = QPushButton("Удалить")
        self.btn_remove_source.clicked.connect(self._on_remove_source)
        btn_row.addWidget(self.btn_add_point)
        btn_row.addWidget(self.btn_add_volume)
        btn_row.addWidget(self.btn_remove_source)
        src_layout.addLayout(btn_row)
        layout.addWidget(src_box)

        layout.addStretch(1)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return panel

    # =========================================================================
    # Центральная панель — 3D-вид.
    # =========================================================================

    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("<b>3D-вид модели</b>"), 1)
        title_row.addWidget(QLabel("Режим:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(CentralView.MODE_MODEL)
        self.mode_combo.addItem(CentralView.MODE_TEMPERATURE)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        title_row.addWidget(self.mode_combo)
        layout.addLayout(title_row)

        self.view = CentralView()
        layout.addWidget(self.view, 1)

        slice_row = QHBoxLayout()
        slice_row.addWidget(QLabel("Сечение:"))
        for axis in ("X", "Y", "Z"):
            btn = QToolButton()
            btn.setText(axis)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, a=axis: self._on_slice_axis(a))
            if axis == "Z":
                btn.setChecked(True)
            slice_row.addWidget(btn)
            setattr(self, f"_slice_btn_{axis}", btn)
        slice_row.addStretch(1)
        layout.addLayout(slice_row)

        return panel

    # =========================================================================
    # Правая панель — граничные условия.
    # =========================================================================

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

        # Шесть карточек.
        self.face_cards: Dict[int, FaceCard] = {}
        # Удобный порядок для пользователя: верх, низ, потом боковые.
        order = [FACE_Z_PLUS, FACE_Z_MINUS, FACE_X_PLUS, FACE_X_MINUS,
                 FACE_Y_PLUS, FACE_Y_MINUS]
        for fid in order:
            card = FaceCard(fid)
            card.changed.connect(self._on_bc_changed)
            self.face_cards[fid] = card
            layout.addWidget(card)

        # Шаблоны.
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

    # =========================================================================
    # Нижняя панель — расчёт.
    # =========================================================================

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
        self.progress.setRange(0, 0)  # «бесконечный» режим во время расчёта
        self.progress.setVisible(False)
        layout.addWidget(self.progress, 1)

        # Сводка результатов.
        self.result_label = QLabel("Готов к расчёту")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 10pt;")
        layout.addWidget(self.result_label, 2)

        # Кнопки экспорта.
        self.btn_vtu = QPushButton("Экспорт .vtu")
        self.btn_vtu.clicked.connect(self._export_vtu)
        self.btn_vtu.setEnabled(False)
        layout.addWidget(self.btn_vtu)

        self.btn_csv = QPushButton("Экспорт .csv")
        self.btn_csv.clicked.connect(self._export_csv)
        self.btn_csv.setEnabled(False)
        layout.addWidget(self.btn_csv)

        self.btn_report = QPushButton("Отчёт .txt")
        self.btn_report.clicked.connect(self._export_report)
        self.btn_report.setEnabled(False)
        layout.addWidget(self.btn_report)

        return panel

    # =========================================================================
    # Слоты управления.
    # =========================================================================

    def _on_preset_changed(self, _idx: int) -> None:
        ps = self.preset_combo.currentData()
        if ps is None:
            return
        self.size_x.setValue(ps.Lx)
        self.size_y.setValue(ps.Ly)
        self.size_z.setValue(ps.Lz)
        self.n_x.setValue(ps.nx)
        self.n_y.setValue(ps.ny)
        self.n_z.setValue(ps.nz)
        self._sync_to_problem()

    def _on_geom_type_changed(self, _idx: int) -> None:
        kind = self.geom_type_combo.currentData()
        is_box = (kind == "box")
        self.box_params_widget.setVisible(is_box)
        self.msh_params_widget.setVisible(not is_box)

    def _on_browse_msh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл сетки",
            "", "Сетка Gmsh (*.msh);;Все файлы (*.*)"
        )
        if not path:
            return
        self._msh_path = path
        self.msh_path_label.setText(f"<b>{os.path.basename(path)}</b>")

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

    def _on_mode_changed(self, mode: str) -> None:
        self.view.set_mode(mode)

    def _on_slice_axis(self, axis: str) -> None:
        for a in ("X", "Y", "Z"):
            btn = getattr(self, f"_slice_btn_{a}")
            btn.setChecked(a == axis)
        self.view.set_slice_axis(axis)

    # -------------------------------------------------------------------------
    def _sync_to_problem(self) -> None:
        # Геометрия — обновляем только если используется box-режим (т.е. не
        # импортированная сетка).
        if not self.problem.has_external_mesh():
            self.problem.geometry = BoxGeometry(
                Lx=self.size_x.value(), Ly=self.size_y.value(), Lz=self.size_z.value(),
                nx=self.n_x.value(), ny=self.n_y.value(), nz=self.n_z.value(),
            )
        # Материал.
        self.problem.lambda_ = self.lambda_spin.value()
        self.problem.Q = self.q_spin.value()
        # ГУ.
        for fid, card in self.face_cards.items():
            self.problem.bcs[fid] = card.bc
        # Передаём во view.
        self.view.set_geometry(self.problem.geometry)
        self.view.set_bcs(self.problem.bcs)

    # =========================================================================
    # Действия.
    # =========================================================================

    def _on_generate_mesh(self) -> None:
        # В зависимости от выбранного типа: либо параметрическая сетка,
        # либо импорт из MSH-файла.
        kind = self.geom_type_combo.currentData()
        if kind == "msh":
            if not self._msh_path:
                QMessageBox.warning(
                    self, "Файл не выбран",
                    "Сначала выберите MSH-файл кнопкой «Выбрать файл .msh...»."
                )
                return
            try:
                from fem3d.mesh import import_msh
                nodes, tets, bnd_nodes, bnd_face_ids = import_msh(self._msh_path)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Ошибка импорта",
                    f"Не удалось загрузить MSH-файл:\n{exc}\n\n"
                    "Проверьте, что установлен пакет meshio:\n"
                    "    pip install meshio"
                )
                return
            # Сохраняем импортированную сетку в Problem.
            self.problem.external_nodes = nodes
            self.problem.external_elements = tets
            self.problem.external_bnd_nodes = bnd_nodes
            self.problem.external_bnd_face_ids = bnd_face_ids
            # Подгоним BoxGeometry под габариты импорта (для 3D-вида).
            bbox_min = nodes.min(axis=0)
            bbox_max = nodes.max(axis=0)
            self.problem.geometry = BoxGeometry(
                Lx=float(bbox_max[0] - bbox_min[0]),
                Ly=float(bbox_max[1] - bbox_min[1]),
                Lz=float(bbox_max[2] - bbox_min[2]),
                nx=1, ny=1, nz=1,
            )
        else:
            # Сбросим импортированную сетку, если возвращаемся к параллелепипеду.
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
        # Обновим 3D-вид (использует self.problem.geometry для габаритов).
        self.view.set_geometry(self.problem.geometry)
        self.view.set_bcs(self.problem.bcs)
        self.view.set_problem(None)
        self.view.update()

    def _on_run(self) -> None:
        self._sync_to_problem()
        # Проверим, что есть хотя бы одно нетривиальное условие.
        types = {bc.type for bc in self.problem.bcs.values()}
        if BC_DIRICHLET not in types and BC_ROBIN not in types:
            QMessageBox.warning(
                self, "Неполные условия",
                "Хотя бы на одной грани должно быть задано условие Дирихле "
                "или Робена — иначе задача определена не однозначно."
            )
            return

        # Запуск воркера в отдельном потоке.
        self.run_button.setEnabled(False)
        self.progress.setVisible(True)
        self.result_label.setText("Идёт расчёт...")

        self._thread = QThread(self)
        self._worker = SolverWorker(self.problem, tol=1e-8, max_iter=5000)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_worker_progress(self, msg: str) -> None:
        self.result_label.setText(msg)

    def _on_worker_finished(self, info: SolverInfo) -> None:
        self.view.set_problem(self.problem)
        self.mode_combo.setCurrentText(CentralView.MODE_TEMPERATURE)
        Tmin, Tmax = self.problem.temperature_range()
        msg = (f"Tmin = {Tmin:.2f} °C    Tmax = {Tmax:.2f} °C    "
               f"итераций: {info.iterations}    невязка: {info.residual:.2e}    "
               f"время: {info.time_seconds*1000:.1f} мс    "
               f"{'сошёлся' if info.converged else 'НЕ сошёлся'}")
        self.result_label.setText(msg)
        self.btn_vtu.setEnabled(True)
        self.btn_csv.setEnabled(True)
        self.btn_report.setEnabled(True)

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

    # =========================================================================
    # Экспорт.
    # =========================================================================

    def _export_vtu(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт результатов в VTU", "result.vtu", "VTK Unstructured (*.vtu)"
        )
        if not path:
            return
        try:
            export_vtu(self.problem, path)
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт результатов в CSV", "result.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            export_csv(self.problem, path)
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def _export_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", "report.txt", "Текстовый файл (*.txt)"
        )
        if not path:
            return
        try:
            export_report(self.problem, path)
            self.statusBar().showMessage(f"Сохранено: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    # =========================================================================
    # Локальные источники.
    # =========================================================================

    def _refresh_sources_list(self) -> None:
        """Перестраивает QListWidget из текущих списков источников в Problem."""
        self.sources_list.clear()
        for ps in self.problem.point_sources:
            item = QListWidgetItem(f"⊙  {ps.description()}")
            item.setData(Qt.UserRole, ("point", ps))
            self.sources_list.addItem(item)
        for vs in self.problem.volume_sources:
            item = QListWidgetItem(f"◯  {vs.description()}")
            item.setData(Qt.UserRole, ("volume", vs))
            self.sources_list.addItem(item)

    def _on_add_point_source(self) -> None:
        if self.problem.nodes is None:
            QMessageBox.information(
                self, "Сетка не построена",
                "Сначала сгенерируйте сетку — точечный источник привязывается "
                "к ближайшему узлу."
            )
            return
        dlg = PointSourceDialog(self.problem.geometry, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        x, y, z, P = dlg.values()
        # Найдём ближайший узел.
        diff = self.problem.nodes - np.array([x, y, z])
        idx = int(np.argmin(np.sum(diff * diff, axis=1)))
        self.problem.point_sources.append(PointSource(node_idx=idx, power=P))
        self._refresh_sources_list()

    def _on_add_volume_source(self) -> None:
        dlg = VolumeSourceDialog(self.problem.geometry, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        cx, cy, cz, r, Q0 = dlg.values()
        self.problem.volume_sources.append(
            VolumeSource(shape=VOLSRC_SPHERE,
                         params=(cx, cy, cz, r),
                         Q0=Q0)
        )
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


# =============================================================================
# Точка входа.
# =============================================================================

def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    # Тёмная палитра — на случай, если stylesheet не покроет какие-то системные виджеты.
    palette = QPalette()
    palette.setColor(QPalette.Window,        QColor(COLOR_BG_DARK))
    palette.setColor(QPalette.WindowText,    QColor(COLOR_TEXT))
    palette.setColor(QPalette.Base,          QColor("#1a1d22"))
    palette.setColor(QPalette.AlternateBase, QColor(COLOR_PANEL))
    palette.setColor(QPalette.Text,          QColor(COLOR_TEXT))
    palette.setColor(QPalette.Button,        QColor("#3c4049"))
    palette.setColor(QPalette.ButtonText,    QColor(COLOR_TEXT))
    palette.setColor(QPalette.Highlight,     QColor(COLOR_ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("white"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
