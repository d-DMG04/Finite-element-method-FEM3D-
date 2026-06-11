# -*- coding: utf-8 -*-
"""
gui.whatif — вкладка «Что будет, если…».

Интерактивное исследование: пользователь двигает слайдеры (λ, α, Q, T_inf),
а температурное поле пересчитывается автоматически на грубой сетке через
~150 мс после остановки слайдера. Показывает T_min/T_max/среднее вживую.

Для отзывчивости расчёт идёт на огрублённой сетке (фиксированный небольшой
box), что даёт время отклика ~0.05–0.5 с.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel,
                              QSlider, QVBoxLayout, QWidget)

from .theme import current_theme


class _LabeledSlider(QWidget):
    """Слайдер с подписью значения и логарифмическим масштабом по желанию."""
    changed = pyqtSignal()

    def __init__(self, label, vmin, vmax, vinit, unit="", log=False, parent=None):
        super().__init__(parent)
        self._vmin = vmin; self._vmax = vmax; self._log = log; self._unit = unit
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self._name = QLabel(label); self._name.setMinimumWidth(60)
        lay.addWidget(self._name)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0); self.slider.setMaximum(1000)
        self.slider.setValue(self._to_slider(vinit))
        self.slider.valueChanged.connect(lambda _v: self.changed.emit())
        lay.addWidget(self.slider, 1)
        self._val = QLabel(""); self._val.setMinimumWidth(110)
        lay.addWidget(self._val)
        self._update_label()
        self.changed.connect(self._update_label)

    def _to_slider(self, value):
        import math
        if self._log:
            t = (math.log10(value) - math.log10(self._vmin)) / \
                (math.log10(self._vmax) - math.log10(self._vmin))
        else:
            t = (value - self._vmin) / (self._vmax - self._vmin)
        return int(round(max(0, min(1, t)) * 1000))

    def value(self):
        import math
        t = self.slider.value() / 1000.0
        if self._log:
            return 10 ** (math.log10(self._vmin)
                          + t * (math.log10(self._vmax) - math.log10(self._vmin)))
        return self._vmin + t * (self._vmax - self._vmin)

    def _update_label(self):
        v = self.value()
        self._val.setText(f"{v:.4g} {self._unit}")


class WhatIfView(QWidget):
    """Вкладка интерактивного исследования параметров."""

    # Сигнал: пересчитать с параметрами (dict). Обрабатывается в main_gui.
    recompute_requested = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        title = QLabel("<b>Что будет, если…</b>")
        title.setStyleSheet("font-size: 12pt;")
        outer.addWidget(title)

        hint = QLabel(
            "Двигайте слайдеры — поле температуры пересчитывается автоматически "
            "на огрублённой сетке. Удобно, чтобы быстро прочувствовать, как "
            "параметры влияют на результат. Точный расчёт — на вкладке 3D-вид.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {current_theme().text_dim}; font-size: 9pt;")
        outer.addWidget(hint)
        self._hint = hint

        form = QFormLayout()
        self.s_lambda = _LabeledSlider("λ", 0.1, 500.0, 50.0,
                                        "Вт/(м·К)", log=True)
        self.s_alpha = _LabeledSlider("α", 1.0, 10000.0, 100.0,
                                       "Вт/(м²·К)", log=True)
        self.s_Q = _LabeledSlider("Q", 0.0, 5.0e6, 0.0, "Вт/м³")
        self.s_tinf = _LabeledSlider("T∞", -50.0, 200.0, 20.0, "°C")
        form.addRow("Теплопроводность:", self.s_lambda)
        form.addRow("Коэф. теплоотдачи:", self.s_alpha)
        form.addRow("Объёмный источник:", self.s_Q)
        form.addRow("Температура среды:", self.s_tinf)
        outer.addLayout(form)

        scen_row = QHBoxLayout()
        scen_row.addWidget(QLabel("Сценарий ГУ:"))
        self.scenario = QComboBox()
        self.scenario.addItem("Конвекция со всех сторон", "all_conv")
        self.scenario.addItem("Источник + конвекция", "source_conv")
        self.scenario.addItem("Горячая грань X− + конвекция", "one_hot")
        scen_row.addWidget(self.scenario, 1)
        outer.addLayout(scen_row)

        self.result = QLabel("Двигайте слайдер для расчёта…")
        self.result.setWordWrap(True)
        outer.addWidget(self.result)
        self.apply_theme()

        outer.addStretch(1)

        # Debounce-таймер: пересчёт через 150 мс после остановки слайдера.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._emit_recompute)
        for s in (self.s_lambda, self.s_alpha, self.s_Q, self.s_tinf):
            s.changed.connect(self._schedule)
        self.scenario.currentIndexChanged.connect(self._schedule)

    def apply_theme(self) -> None:
        """Перекрасить панель результата и подсказку под текущую тему.

        Раньше фон панели был прошит тёмным (#23262c), и в светлой/бежевой
        теме получалось тёмное пятно с нечитаемым текстом.
        """
        th = current_theme()
        self.result.setStyleSheet(
            f"font-size: 11pt; padding: 8px; background: {th.input_bg}; "
            f"color: {th.text}; border: 1px solid {th.border}; "
            "border-radius: 6px;")
        if hasattr(self, "_hint"):
            self._hint.setStyleSheet(
                f"color: {th.text_dim}; font-size: 9pt;")

    def _schedule(self):
        self._timer.start()

    def _emit_recompute(self):
        self.recompute_requested.emit(self.current_params())

    def current_params(self) -> dict:
        return {
            "lambda_": self.s_lambda.value(),
            "alpha": self.s_alpha.value(),
            "Q": self.s_Q.value(),
            "T_inf": self.s_tinf.value(),
            "scenario": self.scenario.currentData(),
        }

    def show_result(self, text: str):
        self.result.setText(text)
