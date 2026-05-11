# -*- coding: utf-8 -*-
"""
gui.calculations — вкладка с историей выполненных расчётов.

Хранит снимки задачи (Problem) после каждого успешного решения и позволяет
пользователю переключаться между ними.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QPushButton, QSplitter, QTextEdit, QVBoxLayout,
                             QWidget)

from fem3d import FACE_NAMES, BC_DIRICHLET, BC_NEUMANN, BC_ROBIN, Problem


@dataclass
class CalculationRecord:
    """Один сохранённый расчёт."""
    title: str
    timestamp: datetime
    problem: Problem
    notes: str = ""

    def short_label(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        info = self.problem.info
        if info is None:
            tail = "—"
        else:
            tail = (f"{info.iterations} итер, "
                    f"{info.time_seconds * 1000:.1f} мс, "
                    f"{'OK' if info.converged else 'не сошёлся'}")
        return f"{ts}  ·  {self.title}  ·  {tail}"


class CalculationsView(QWidget):
    """Вкладка «Расчёты»."""

    # Пользователь выбрал расчёт из списка.
    selected = pyqtSignal(object)  # CalculationRecord

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        title = QLabel("<b>История расчётов в текущей сессии</b>")
        title.setStyleSheet("font-size: 11pt;")
        outer.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)

        # Слева — список.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self._on_select)
        left_layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.btn_remove = QPushButton("Удалить")
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_clear = QPushButton("Очистить все")
        self.btn_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear)
        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        # Справа — детали.
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setStyleSheet("background-color: #1a1d22; color: #dcdee2;")
        splitter.addWidget(self.details)
        splitter.setSizes([400, 500])

        outer.addWidget(splitter, 1)

        self._records: List[CalculationRecord] = []

    # -------------------------------------------------------------------------
    def add_record(self, problem: Problem, title: str = "Расчёт") -> None:
        """Добавляет снимок текущего problem (включая результаты)."""
        # Делаем глубокую копию — иначе последующие изменения в GUI
        # модифицируют сохранённую запись.
        snapshot = copy.deepcopy(problem)
        record = CalculationRecord(title=title, timestamp=datetime.now(),
                                    problem=snapshot)
        self._records.append(record)
        item = QListWidgetItem(record.short_label())
        item.setData(Qt.UserRole, record)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentItem(item)

    def _on_select(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            self.details.clear()
            return
        record: CalculationRecord = item.data(Qt.UserRole)
        self.details.setHtml(self._format_details(record))
        self.selected.emit(record)

    def _on_remove(self) -> None:
        row = self.list_widget.currentRow()
        if row < 0:
            return
        del self._records[row]
        self.list_widget.takeItem(row)

    def _on_clear(self) -> None:
        self._records.clear()
        self.list_widget.clear()
        self.details.clear()

    # -------------------------------------------------------------------------
    @staticmethod
    def _format_details(rec: CalculationRecord) -> str:
        p = rec.problem
        info = p.info
        Tmin, Tmax = p.temperature_range()

        lines = []
        lines.append(f"<h3>{rec.title}</h3>")
        lines.append(f"<p style='color:#9aa0a6'>"
                     f"{rec.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>")

        lines.append("<h4>Геометрия</h4>")
        g = p.geometry
        lines.append(f"<p>Lx = {g.Lx:g}, Ly = {g.Ly:g}, Lz = {g.Lz:g} м<br>")
        if p.has_external_mesh():
            lines.append("Сетка: импортирована (внешний файл)<br>")
        else:
            lines.append(f"Разбиение: nx = {g.nx}, ny = {g.ny}, nz = {g.nz}<br>")
        if p.nodes is not None:
            lines.append(f"Узлов: {p.nodes.shape[0]}, "
                         f"элементов: {p.elements.shape[0]}</p>")

        lines.append("<h4>Материал</h4>")
        lines.append(f"<p>λ = {p.lambda_:g} Вт/(м·К), "
                     f"Q = {p.Q:g} Вт/м³</p>")

        lines.append("<h4>Граничные условия</h4>")
        lines.append("<p>")
        for fid in range(6):
            bc = p.bcs[fid]
            lines.append(f"<b>{FACE_NAMES[fid]}</b>: {bc.description()}<br>")
        lines.append("</p>")

        if p.point_sources or p.volume_sources:
            lines.append("<h4>Локальные источники</h4>")
            lines.append("<p>")
            for ps in p.point_sources:
                lines.append(ps.description() + "<br>")
            for vs in p.volume_sources:
                lines.append(vs.description() + "<br>")
            lines.append("</p>")

        if info is not None:
            lines.append("<h4>Решатель</h4>")
            lines.append(f"<p>Итераций: {info.iterations}<br>"
                         f"Норма невязки: {info.residual:.3e}<br>"
                         f"Сошёлся: {'да' if info.converged else 'нет'}<br>"
                         f"Время: {info.time_seconds * 1000:.1f} мс</p>")

        lines.append("<h4>Результаты</h4>")
        lines.append(f"<p>Tmin = {Tmin:.4f} °C<br>"
                     f"Tmax = {Tmax:.4f} °C<br>"
                     f"Размах ΔT = {Tmax - Tmin:.4f} °C</p>")

        return "\n".join(lines)
