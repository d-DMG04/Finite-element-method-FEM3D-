# -*- coding: utf-8 -*-
"""
gui.calculations — вкладка «Расчёты»: таблица всех выполненных расчётов
с сортировкой по столбцам, переключением расчёта двойным кликом,
экспортом таблицы в CSV и сравнением выбранных расчётов.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog,
                              QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                              QMessageBox, QPushButton, QSplitter, QTabWidget,
                              QTableWidget, QTableWidgetItem, QTextEdit,
                              QVBoxLayout, QWidget)

from .theme import current_theme

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
    """Вкладка «Расчёты» в виде таблицы."""

    selected = pyqtSignal(object)

    # Колонки таблицы: (заголовок, тип значения для сортировки).
    COLUMNS = [
        ("№",          int),
        ("Время",      str),
        ("Название",   str),
        ("λ, Вт/(м·К)", float),
        ("T_min, °C",  float),
        ("T_max, °C",  float),
        ("ΔT, °C",     float),
        ("Итераций",   int),
        ("t решения, мс", float),
        ("Сошёлся",    str),
        ("Энергобаланс, %", float),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        title = QLabel("<b>История расчётов в текущей сессии</b>")
        title.setStyleSheet("font-size: 11pt;")
        outer.addWidget(title)

        hint = QLabel(
            "Двойной клик по строке — открыть расчёт. "
            "Колонки можно сортировать. Выберите несколько строк через "
            "Ctrl и нажмите «Сравнить» для наложения графиков.")
        hint.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        outer.addWidget(hint)

        splitter = QSplitter(Qt.Vertical)

        # ----- Таблица -----
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in self.COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemDoubleClicked.connect(self._on_row_double_clicked)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.table)

        # ----- Нижняя часть: вкладки «Сводка» и «По узлам» -----
        self.detail_tabs = QTabWidget()

        # Вкладка «Сводка» — документационное описание.
        self.details = QTextEdit(); self.details.setReadOnly(True)
        self.detail_tabs.addTab(self.details, "Сводка")

        # Вкладка «Результаты по узлам».
        node_panel = QWidget()
        np_layout = QVBoxLayout(node_panel)
        np_layout.setContentsMargins(4, 4, 4, 4)

        np_ctrl = QHBoxLayout()
        np_ctrl.addWidget(QLabel("Расчёт:"))
        self.node_calc_combo = QComboBox()
        self.node_calc_combo.currentIndexChanged.connect(self._on_node_calc_changed)
        np_ctrl.addWidget(self.node_calc_combo, 1)
        np_ctrl.addWidget(QLabel("Фильтр T ≥"))
        self.node_filter_edit = QLineEdit()
        self.node_filter_edit.setPlaceholderText("например, 50")
        self.node_filter_edit.setMaximumWidth(90)
        self.node_filter_edit.editingFinished.connect(self._refresh_node_table)
        np_ctrl.addWidget(self.node_filter_edit)
        self.btn_node_export = QPushButton("Экспорт узлов CSV")
        self.btn_node_export.clicked.connect(self._on_export_nodes_csv)
        np_ctrl.addWidget(self.btn_node_export)
        np_layout.addLayout(np_ctrl)

        self.node_info = QLabel("")
        self.node_info.setStyleSheet(f"color: {current_theme().text_dim}; font-size: 9pt;")
        np_layout.addWidget(self.node_info)

        self.node_table = QTableWidget(0, 8)
        self.node_table.setHorizontalHeaderLabels(
            ["№ узла", "X, м", "Y, м", "Z, м", "T, °C",
             "|q|, Вт/м²", "q_x", "q_z"])
        self.node_table.setSortingEnabled(True)
        self.node_table.verticalHeader().setVisible(False)
        self.node_table.horizontalHeader().setStretchLastSection(True)
        np_layout.addWidget(self.node_table, 1)

        self.detail_tabs.addTab(node_panel, "Результаты по узлам")
        splitter.addWidget(self.detail_tabs)
        splitter.setSizes([300, 280])
        outer.addWidget(splitter, 1)

        # ----- Кнопки -----
        row = QHBoxLayout()
        self.btn_open = QPushButton("Открыть расчёт")
        self.btn_open.clicked.connect(self._on_open_selected)
        row.addWidget(self.btn_open)

        self.btn_compare = QPushButton("Сравнить выбранные")
        self.btn_compare.setEnabled(False)
        self.btn_compare.clicked.connect(self._on_compare_selected)
        row.addWidget(self.btn_compare)

        self.btn_export = QPushButton("Экспорт таблицы CSV")
        self.btn_export.clicked.connect(self._on_export_csv)
        row.addWidget(self.btn_export)

        row.addStretch(1)
        self.btn_remove = QPushButton("Удалить")
        self.btn_remove.clicked.connect(self._on_remove)
        row.addWidget(self.btn_remove)
        self.btn_clear = QPushButton("Очистить все")
        self.btn_clear.clicked.connect(self._on_clear)
        row.addWidget(self.btn_clear)
        outer.addLayout(row)

        # Список записей и счётчик для номера.
        self._records: List[CalculationRecord] = []
        self._next_id = 1

    # =========================================================================
    # API.
    # =========================================================================

    def add_record(self, problem: Problem, title: str = "") -> None:
        snap = copy.deepcopy(problem)
        rec = CalculationRecord(
            title=title or f"Расчёт #{self._next_id}",
            timestamp=datetime.now(),
            problem=snap,
        )
        self._records.append(rec)
        self._next_id += 1
        self._add_row(rec, len(self._records) - 1)
        # Обновляем выпадающий список расчётов для таблицы узлов.
        self._refresh_node_combo()

    def _refresh_node_combo(self) -> None:
        self.node_calc_combo.blockSignals(True)
        self.node_calc_combo.clear()
        for i, rec in enumerate(self._records):
            self.node_calc_combo.addItem(f"#{i+1} · {rec.title}", i)
        self.node_calc_combo.blockSignals(False)
        if self._records:
            self.node_calc_combo.setCurrentIndex(len(self._records) - 1)
            self._refresh_node_table()

    def records(self) -> List[CalculationRecord]:
        return list(self._records)

    # =========================================================================
    # Слоты.
    # =========================================================================

    def _add_row(self, rec: CalculationRecord, idx: int) -> None:
        self.table.setSortingEnabled(False)
        row = self.table.rowCount()
        self.table.insertRow(row)

        p = rec.problem
        info = p.info
        Tmin, Tmax = p.temperature_range()
        bal = p.energy_balance()
        bal_pct = bal["rel_err"] * 100 if bal else 0.0

        # Хранится сам объект записи в UserRole первой ячейки.
        cells = [
            self._num_item(idx + 1),
            self._str_item(rec.timestamp.strftime("%H:%M:%S")),
            self._str_item(rec.title),
            self._num_item(p.lambda_),
            self._num_item(Tmin),
            self._num_item(Tmax),
            self._num_item(Tmax - Tmin),
            self._num_item(info.iterations if info else 0),
            self._num_item(info.time_seconds * 1000 if info else 0),
            self._str_item("✓" if info and info.converged else "✗"),
            self._num_item(bal_pct),
        ]
        cells[0].setData(Qt.UserRole, rec)
        for c, item in enumerate(cells):
            self.table.setItem(row, c, item)

        # Цвет строки энергобаланса.
        if bal_pct < 5:
            color = QColor("#1e3a1e")
        elif bal_pct < 20:
            color = QColor("#3a3a1e")
        else:
            color = QColor("#3a1e1e")
        # Применяем только к колонке энергобаланса.
        cells[10].setBackground(color)

        self.table.setSortingEnabled(True)

    def _num_item(self, value) -> QTableWidgetItem:
        it = QTableWidgetItem()
        # Используем setData с DisplayRole + EditRole для корректной сортировки.
        try:
            v = float(value)
            it.setData(Qt.DisplayRole, f"{v:.4g}" if isinstance(value, float)
                       else str(int(value)))
            it.setData(Qt.EditRole, v)
        except Exception:
            it.setData(Qt.DisplayRole, str(value))
        return it

    def _str_item(self, value: str) -> QTableWidgetItem:
        return QTableWidgetItem(str(value))

    def _selected_records(self) -> List[CalculationRecord]:
        rows = sorted({r.row() for r in self.table.selectedIndexes()})
        recs = []
        for r in rows:
            item = self.table.item(r, 0)
            if item:
                rec = item.data(Qt.UserRole)
                if isinstance(rec, CalculationRecord):
                    recs.append(rec)
        return recs

    def _on_selection_changed(self):
        recs = self._selected_records()
        self.btn_compare.setEnabled(len(recs) >= 2)
        if len(recs) == 1:
            self.details.setHtml(self._format_details(recs[0]))
        elif len(recs) > 1:
            self.details.setHtml(
                f"<p>Выбрано {len(recs)} записей. "
                f"Нажмите «Сравнить выбранные» для наложения графиков, "
                f"либо двойной клик по одной строке для её открытия.</p>")
        else:
            self.details.setHtml("")

    def _on_row_double_clicked(self, item: QTableWidgetItem) -> None:
        rec = self.table.item(item.row(), 0).data(Qt.UserRole)
        if isinstance(rec, CalculationRecord):
            self.selected.emit(rec)

    def _on_open_selected(self):
        recs = self._selected_records()
        if recs:
            self.selected.emit(recs[0])

    def _on_compare_selected(self):
        recs = self._selected_records()
        if len(recs) < 2:
            return
        # Открываем диалог сравнения.
        dlg = ComparisonDialog(recs, self)
        dlg.exec_()

    def _on_export_csv(self):
        if not self._records:
            QMessageBox.information(self, "Нет данных", "История расчётов пуста.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт таблицы расчётов",
            "calculations.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([c[0] for c in self.COLUMNS])
                for r in range(self.table.rowCount()):
                    row_data = []
                    for c in range(self.table.columnCount()):
                        item = self.table.item(r, c)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            QMessageBox.information(self, "Готово", f"Сохранено: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _on_remove(self):
        rows = sorted({r.row() for r in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            item = self.table.item(r, 0)
            if item:
                rec = item.data(Qt.UserRole)
                if rec in self._records:
                    self._records.remove(rec)
            self.table.removeRow(r)
        self._refresh_node_combo()

    def _on_clear(self):
        if QMessageBox.question(self, "Очистить всё",
                                 "Удалить все записи из истории?",
                                 QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No) == QMessageBox.Yes:
            self.table.setRowCount(0)
            self._records.clear()
            self.details.setHtml("")
            self.node_table.setRowCount(0)
            self.node_calc_combo.clear()
            self.node_info.setText("")
            self._next_id = 1

    # =========================================================================
    # Таблица результатов по узлам.
    # =========================================================================

    def _on_node_calc_changed(self, _idx: int) -> None:
        self._refresh_node_table()

    def _current_node_record(self):
        idx = self.node_calc_combo.currentData()
        if idx is None or idx < 0 or idx >= len(self._records):
            return None
        return self._records[idx]

    def _refresh_node_table(self) -> None:
        import numpy as np
        rec = self._current_node_record()
        if rec is None:
            self.node_table.setRowCount(0)
            self.node_info.setText("Нет данных")
            return
        p = rec.problem
        if p.T is None or p.nodes is None:
            self.node_table.setRowCount(0)
            self.node_info.setText("Расчёт не содержит поля температур")
            return
        nodes = p.nodes
        T = p.T
        flux = p.flux  # (N,3) или None
        # Фильтр по температуре.
        thr = None
        txt = self.node_filter_edit.text().strip().replace(",", ".")
        if txt:
            try:
                thr = float(txt)
            except ValueError:
                thr = None
        if thr is not None:
            mask = T >= thr
            indices = np.flatnonzero(mask)
        else:
            indices = np.arange(len(T))

        # Ограничение для производительности: не более 5000 строк в таблице.
        MAX_ROWS = 5000
        truncated = len(indices) > MAX_ROWS
        show_idx = indices[:MAX_ROWS]

        self.node_table.setSortingEnabled(False)
        self.node_table.setRowCount(len(show_idx))
        for r, ni in enumerate(show_idx):
            ni = int(ni)
            x, y, z = nodes[ni]
            t_val = float(T[ni])
            if flux is not None:
                qx, qy, qz = flux[ni]
                qmag = float((qx*qx + qy*qy + qz*qz) ** 0.5)
            else:
                qx = qz = qmag = float("nan")
            vals = [ni, x, y, z, t_val, qmag, float(qx),
                    float(qz) if flux is not None else float("nan")]
            for c, v in enumerate(vals):
                if c == 0:
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, int(v))
                else:
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, round(float(v), 6))
                self.node_table.setItem(r, c, item)
        self.node_table.setSortingEnabled(True)

        total = len(indices)
        info_txt = (f"Узлов показано: {len(show_idx)} из {len(T)} "
                    f"(после фильтра: {total})")
        if truncated:
            info_txt += f"  · обрезано до {MAX_ROWS} строк"
        info_txt += (f"   |   T: {float(T.min()):.3f} … {float(T.max()):.3f} °C, "
                     f"среднее {float(T.mean()):.3f} °C")
        self.node_info.setText(info_txt)

    def _on_export_nodes_csv(self) -> None:
        import numpy as np
        rec = self._current_node_record()
        if rec is None or rec.problem.T is None:
            QMessageBox.information(self, "Нет данных",
                                     "Выберите расчёт с результатами.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт узлов", "nodes.csv", "CSV (*.csv)")
        if not path:
            return
        p = rec.problem
        nodes = p.nodes; T = p.T; flux = p.flux
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["node", "x_m", "y_m", "z_m", "T_C",
                            "q_mag_W_m2", "qx", "qy", "qz"])
                for ni in range(len(T)):
                    x, y, z = nodes[ni]
                    row = [ni, f"{x:.6g}", f"{y:.6g}", f"{z:.6g}",
                           f"{float(T[ni]):.6f}"]
                    if flux is not None:
                        qx, qy, qz = flux[ni]
                        qmag = (qx*qx + qy*qy + qz*qz) ** 0.5
                        row += [f"{qmag:.4f}", f"{qx:.4f}", f"{qy:.4f}", f"{qz:.4f}"]
                    else:
                        row += ["", "", "", ""]
                    w.writerow(row)
            self.node_info.setText(f"Сохранено: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    # =========================================================================
    # Форматирование деталей.
    # =========================================================================

    @staticmethod
    def _format_details(rec: CalculationRecord) -> str:
        p = rec.problem
        info = p.info
        Tmin, Tmax = p.temperature_range()
        lines = [f"<h3>{rec.title}</h3>",
                  f"<p><b>Создан:</b> {rec.timestamp.strftime('%H:%M:%S %d.%m.%Y')}</p>"]

        lines.append("<h4>Постановка</h4>")
        lines.append("<p>")
        lines.append(f"λ = {p.lambda_} Вт/(м·К)<br>")
        lines.append(f"Q = {p.Q} Вт/м³<br>")
        lines.append(f"Размеры: {p.geometry.Lx} × {p.geometry.Ly} × {p.geometry.Lz} м<br>")
        if p.material_regions:
            lines.append(f"<br>Регионов материалов: {len(p.material_regions)}<br>")
        if p.point_sources:
            lines.append(f"Точечных источников: {len(p.point_sources)}<br>")
        if p.volume_sources:
            lines.append(f"Объёмных источников: {len(p.volume_sources)}<br>")
        lines.append("</p>")

        lines.append("<h4>Граничные условия</h4>")
        lines.append("<ul>")
        for fid in range(6):
            bc = p.bcs[fid]
            lines.append(f"<li><b>{FACE_NAMES[fid]}:</b> {bc.description()}</li>")
        lines.append("</ul>")

        if info is not None:
            lines.append("<h4>Решатель</h4>")
            lines.append(f"<p>Итераций: {info.iterations}<br>"
                         f"Норма невязки: {info.residual:.3e}<br>"
                         f"Сошёлся: {'да' if info.converged else 'нет'}<br>"
                         f"Время: {info.time_seconds * 1000:.1f} мс</p>")

        lines.append("<h4>Результаты</h4>")
        lines.append(f"<p>T_min = {Tmin:.4f} °C<br>"
                     f"T_max = {Tmax:.4f} °C<br>"
                     f"Размах ΔT = {Tmax - Tmin:.4f} °C</p>")

        bal = p.energy_balance()
        if bal is not None:
            rel = bal["rel_err"]
            sc = "#3aa55a" if rel < 0.05 else ("#e0a020" if rel < 0.2 else "#d05050")
            st = ("отлично" if rel < 0.05
                  else "приемлемо" if rel < 0.2 else "плохо")
            lines.append("<h4>Энергобаланс</h4>")
            lines.append(
                f"<p>Генерируется внутри: <b>{bal['q_gen_W']:.3f} Вт</b><br>"
                f"Уходит через границу: <b>{bal['net_out_W']:.3f} Вт</b><br>"
                f"Невязка: {bal['imbalance_W']:.3f} Вт<br>"
                f"Отн. ошибка: <span style='color:{sc}'><b>{rel*100:.2f}%</b> "
                f"({st})</span></p>")
        return "\n".join(lines)


# =============================================================================
# Диалог сравнения нескольких расчётов.
# =============================================================================

class ComparisonDialog(QWidget):
    """Сравнение 2-N расчётов: графики T(x) на одной оси."""

    def __init__(self, records, parent=None):
        # Используем QDialog без модальности — отдельное окно.
        from PyQt5.QtWidgets import QDialog
        # Костыль: делаем виджет QDialog через переопределение __class__.
        # Здесь упрощу — пусть Comparison будет диалогом.
        pass

    def exec_(self):
        # Простая реализация: открывает matplotlib window.
        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from PyQt5.QtWidgets import QDialog
        except Exception as e:
            QMessageBox.critical(None, "Ошибка", str(e))
            return

        recs = self._records
        dlg = QDialog()
        dlg.setWindowTitle("Сравнение расчётов")
        dlg.resize(900, 600)
        outer = QVBoxLayout(dlg)
        th = current_theme()
        fig = Figure(figsize=(9, 6), facecolor=th.panel)
        canvas = FigureCanvasQTAgg(fig)
        outer.addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.set_facecolor(th.bg)
        ax.set_xlabel("x, м", color=th.text)
        ax.set_ylabel("T, °C", color=th.text)
        ax.tick_params(colors=th.text)
        ax.set_title("Профиль T(x) вдоль центральной линии y=Ly/2, z=Lz/2",
                     color=th.text)

        import numpy as np
        for i, rec in enumerate(recs):
            p = rec.problem
            if p.T is None or p.nodes is None:
                continue
            g = p.geometry
            cy, cz = g.Ly/2, g.Lz/2
            tol = 0.05 * max(g.Ly, g.Lz)
            mask = ((np.abs(p.nodes[:, 1] - cy) < tol)
                    & (np.abs(p.nodes[:, 2] - cz) < tol))
            if not np.any(mask):
                continue
            xs = p.nodes[mask, 0]
            Ts = p.T[mask]
            order = np.argsort(xs)
            ax.plot(xs[order], Ts[order], "-o", markersize=3,
                    label=rec.title, linewidth=1.5)
        ax.legend(facecolor=th.panel, edgecolor=th.border,
                  labelcolor=th.text)
        ax.grid(alpha=0.3)
        canvas.draw()
        dlg.exec_()


# Привязка: ComparisonDialog должен иметь self._records — установим в _init_:
def _comp_init(self, records, parent=None):
    from PyQt5.QtWidgets import QDialog
    QDialog.__init__(self, parent)
    self._records = records


# Используем простой подход — наследуем от QDialog корректно.
from PyQt5.QtWidgets import QDialog as _QDialog

class ComparisonDialog(_QDialog):  # type: ignore[no-redef]
    """Сравнение 2-N расчётов: графики T(x) на одной оси."""

    def __init__(self, records, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сравнение расчётов")
        self.resize(900, 600)
        self._records = records

        outer = QVBoxLayout(self)

        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        except Exception:
            outer.addWidget(QLabel("matplotlib недоступен."))
            return

        th = current_theme()
        fig = Figure(figsize=(9, 6), facecolor=th.panel)
        canvas = FigureCanvasQTAgg(fig)
        outer.addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.set_facecolor(th.bg)
        ax.set_xlabel("x, м", color=th.text)
        ax.set_ylabel("T, °C", color=th.text)
        ax.tick_params(colors=th.text)
        ax.set_title("Профиль T(x) вдоль центральной линии y=Ly/2, z=Lz/2",
                     color=th.text)

        import numpy as np
        for rec in records:
            p = rec.problem
            if p.T is None or p.nodes is None:
                continue
            g = p.geometry
            cy, cz = g.Ly / 2, g.Lz / 2
            tol = 0.05 * max(g.Ly, g.Lz)
            mask = ((np.abs(p.nodes[:, 1] - cy) < tol)
                    & (np.abs(p.nodes[:, 2] - cz) < tol))
            if not np.any(mask):
                continue
            xs = p.nodes[mask, 0]; Ts = p.T[mask]
            order = np.argsort(xs)
            ax.plot(xs[order], Ts[order], "-o", markersize=3,
                    label=rec.title, linewidth=1.5)
        ax.legend(facecolor=th.panel, edgecolor=th.border,
                  labelcolor=th.text)
        ax.grid(alpha=0.3)
        canvas.draw()
