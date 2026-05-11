# -*- coding: utf-8 -*-
"""
gui.face_card — карточка граничного условия на одной грани.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QVBoxLayout, QWidget)

from fem3d import (BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN,
                   BoundaryCondition, FACE_NAMES)

from .theme import COLOR_NONE, COLOR_TEXT_DIM, bc_colors


class FaceCard(QFrame):
    """Карточка грани: тип ГУ + параметры. Цветной индикатор слева."""

    changed = pyqtSignal(int)

    def __init__(self, face_id: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.face_id = face_id
        self.bc = BoundaryCondition()
        self.setObjectName("Panel")
        self._build_ui()
        self._update_indicator()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 8, 0)
        outer.setSpacing(8)

        self.indicator = QFrame()
        self.indicator.setFixedWidth(6)
        self.indicator.setStyleSheet(
            f"background-color: {COLOR_NONE};"
            "border-top-left-radius: 6px;"
            "border-bottom-left-radius: 6px;"
        )
        outer.addWidget(self.indicator)

        body = QVBoxLayout()
        body.setContentsMargins(8, 8, 8, 8)
        body.setSpacing(4)

        self.title_label = QLabel(f"<b>{FACE_NAMES[self.face_id]}</b>")
        body.addWidget(self.title_label)

        self.desc_label = QLabel(self.bc.description())
        self.desc_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 9pt;")
        body.addWidget(self.desc_label)

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

        self.params_widget = QWidget()
        params_layout = QGridLayout(self.params_widget)
        params_layout.setContentsMargins(0, 4, 0, 0)
        params_layout.setHorizontalSpacing(6)
        params_layout.setVerticalSpacing(4)

        self.t0_label = QLabel("T₀, °C:")
        self.t0_spin = QDoubleSpinBox()
        self.t0_spin.setRange(-273.0, 5000.0); self.t0_spin.setDecimals(2); self.t0_spin.setValue(0.0)
        self.t0_spin.valueChanged.connect(self._collect_and_emit)
        params_layout.addWidget(self.t0_label, 0, 0)
        params_layout.addWidget(self.t0_spin,  0, 1)

        self.q0_label = QLabel("q, Вт/м²:")
        self.q0_spin = QDoubleSpinBox()
        self.q0_spin.setRange(-1e9, 1e9); self.q0_spin.setDecimals(2); self.q0_spin.setValue(0.0)
        self.q0_spin.valueChanged.connect(self._collect_and_emit)
        params_layout.addWidget(self.q0_label, 1, 0)
        params_layout.addWidget(self.q0_spin,  1, 1)

        self.alpha_label = QLabel("α, Вт/(м²·К):")
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1e6); self.alpha_spin.setDecimals(2); self.alpha_spin.setValue(25.0)
        self.alpha_spin.valueChanged.connect(self._collect_and_emit)
        params_layout.addWidget(self.alpha_label, 2, 0)
        params_layout.addWidget(self.alpha_spin,  2, 1)

        self.tinf_label = QLabel("T∞, °C:")
        self.tinf_spin = QDoubleSpinBox()
        self.tinf_spin.setRange(-273.0, 5000.0); self.tinf_spin.setDecimals(2); self.tinf_spin.setValue(20.0)
        self.tinf_spin.valueChanged.connect(self._collect_and_emit)
        params_layout.addWidget(self.tinf_label, 3, 0)
        params_layout.addWidget(self.tinf_spin,  3, 1)

        body.addWidget(self.params_widget)
        outer.addLayout(body, 1)
        self._refresh_visibility()

    def _on_type_changed(self, _idx: int) -> None:
        self._refresh_visibility()
        self._collect_and_emit()

    def _refresh_visibility(self) -> None:
        bc_type = self.type_combo.currentData()
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
        color = bc_colors().get(self.bc.type, COLOR_NONE)
        self.indicator.setStyleSheet(
            f"background-color: {color};"
            "border-top-left-radius: 6px;"
            "border-bottom-left-radius: 6px;"
        )

    def set_bc(self, bc: BoundaryCondition) -> None:
        for w in (self.type_combo, self.t0_spin, self.q0_spin,
                  self.alpha_spin, self.tinf_spin):
            w.blockSignals(True)
        for i in range(self.type_combo.count()):
            if int(self.type_combo.itemData(i)) == int(bc.type):
                self.type_combo.setCurrentIndex(i)
                break
        self.t0_spin.setValue(bc.T0)
        self.q0_spin.setValue(bc.q0)
        self.alpha_spin.setValue(bc.alpha)
        self.tinf_spin.setValue(bc.T_inf)
        for w in (self.type_combo, self.t0_spin, self.q0_spin,
                  self.alpha_spin, self.tinf_spin):
            w.blockSignals(False)
        self.bc = bc
        self.desc_label.setText(bc.description())
        self._refresh_visibility()
        self._update_indicator()
