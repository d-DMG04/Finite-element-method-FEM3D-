# -*- coding: utf-8 -*-
"""
gui.plots — вкладка с двумерными графиками результатов расчёта.

Содержит:
  - гистограмму температурного поля;
  - профиль температуры вдоль выбранной оси (через bbox-центр);
  - сходимость CG (если есть данные).
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel,
                             QSizePolicy, QSplitter, QVBoxLayout, QWidget)


def _styled_figure() -> Figure:
    fig = Figure(figsize=(7, 4), facecolor="#2a2e36")
    return fig


def _style_ax(ax) -> None:
    ax.set_facecolor("#1f2228")
    ax.spines["bottom"].set_color("#5a606b")
    ax.spines["top"].set_color("#5a606b")
    ax.spines["left"].set_color("#5a606b")
    ax.spines["right"].set_color("#5a606b")
    ax.tick_params(colors="#dcdee2")
    ax.xaxis.label.set_color("#dcdee2")
    ax.yaxis.label.set_color("#dcdee2")
    ax.title.set_color("#dcdee2")
    ax.grid(True, color="#3c4049", alpha=0.5)


class PlotsView(QWidget):
    """Вкладка «Графики»."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # Верхняя строка: выбор графика.
        top = QHBoxLayout()
        top.addWidget(QLabel("График:"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItem("Гистограмма температуры", "histogram")
        self.plot_combo.addItem("Профиль T вдоль оси", "profile")
        self.plot_combo.addItem("Распределение тепловых потоков", "flux_hist")
        self.plot_combo.currentIndexChanged.connect(self._refresh)
        top.addWidget(self.plot_combo, 1)

        top.addWidget(QLabel("Ось профиля:"))
        self.axis_combo = QComboBox()
        for a in ("x", "y", "z"):
            self.axis_combo.addItem(a)
        self.axis_combo.setCurrentText("z")
        self.axis_combo.currentTextChanged.connect(self._refresh)
        top.addWidget(self.axis_combo)
        outer.addLayout(top)

        # Холст.
        self.fig = _styled_figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.canvas, 1)

        # Состояние.
        self._nodes: Optional[np.ndarray] = None
        self._T: Optional[np.ndarray] = None
        self._flux: Optional[np.ndarray] = None
        self._refresh()

    # -------------------------------------------------------------------------
    def set_results(self, nodes: Optional[np.ndarray],
                    T: Optional[np.ndarray],
                    flux: Optional[np.ndarray]) -> None:
        self._nodes = nodes
        self._T = T
        self._flux = flux
        self._refresh()

    def _refresh(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        _style_ax(ax)
        if self._T is None or self._nodes is None or self._T.size == 0:
            ax.text(0.5, 0.5, "Нет данных. Выполните расчёт.",
                    ha="center", va="center", color="#9aa0a6", fontsize=11,
                    transform=ax.transAxes)
            self.canvas.draw_idle()
            return

        kind = self.plot_combo.currentData()
        if kind == "histogram":
            self._plot_histogram(ax)
        elif kind == "profile":
            self._plot_profile(ax)
        elif kind == "flux_hist":
            self._plot_flux_hist(ax)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # -------------------------------------------------------------------------
    def _plot_histogram(self, ax) -> None:
        ax.hist(self._T, bins=40, color="#7a6cf0", edgecolor="#1f2228")
        ax.set_xlabel("T, °C")
        ax.set_ylabel("Число узлов")
        ax.set_title("Распределение температуры по узлам сетки")
        Tmin, Tmax = float(self._T.min()), float(self._T.max())
        Tmean = float(self._T.mean())
        ax.axvline(Tmean, color="#3aa55a", linestyle="--", linewidth=1.5,
                   label=f"среднее: {Tmean:.2f} °C")
        ax.legend(facecolor="#2a2e36", labelcolor="#dcdee2",
                  edgecolor="#5a606b")
        ax.text(0.02, 0.97,
                f"Tmin = {Tmin:.2f} °C\nTmax = {Tmax:.2f} °C\n"
                f"размах = {Tmax - Tmin:.2f} °C",
                transform=ax.transAxes, va="top", color="#dcdee2",
                fontsize=9, bbox=dict(facecolor="#1f2228",
                                       edgecolor="#5a606b"))

    def _plot_profile(self, ax) -> None:
        axis = self.axis_combo.currentText()
        ax_idx = {"x": 0, "y": 1, "z": 2}[axis]
        # Делаем профиль через bbox-центр: усредняем T по всем узлам с
        # одинаковым значением координаты на этой оси (бинами).
        coord = self._nodes[:, ax_idx]
        # 50 бинов вдоль оси — достаточно для гладкой кривой.
        nbins = 50
        cmin, cmax = float(coord.min()), float(coord.max())
        edges = np.linspace(cmin, cmax, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        means = np.full(nbins, np.nan)
        mins  = np.full(nbins, np.nan)
        maxs  = np.full(nbins, np.nan)
        for i in range(nbins):
            mask = (coord >= edges[i]) & (coord < edges[i + 1])
            if i == nbins - 1:
                mask |= (coord == edges[i + 1])
            if np.any(mask):
                vals = self._T[mask]
                means[i] = vals.mean()
                mins[i] = vals.min()
                maxs[i] = vals.max()
        ok = ~np.isnan(means)
        ax.fill_between(centers[ok], mins[ok], maxs[ok],
                        color="#7a6cf0", alpha=0.25, label="разброс")
        ax.plot(centers[ok], means[ok], color="#7a6cf0", linewidth=2,
                label="среднее T(·)")
        ax.set_xlabel(f"{axis}, м")
        ax.set_ylabel("T, °C")
        ax.set_title(f"Профиль температуры вдоль оси {axis.upper()}")
        ax.legend(facecolor="#2a2e36", labelcolor="#dcdee2",
                  edgecolor="#5a606b")

    def _plot_flux_hist(self, ax) -> None:
        if self._flux is None:
            ax.text(0.5, 0.5, "Тепловые потоки не вычислены.",
                    ha="center", va="center", color="#9aa0a6",
                    transform=ax.transAxes)
            return
        magnitude = np.linalg.norm(self._flux, axis=1)
        ax.hist(magnitude, bins=40, color="#3a78d0", edgecolor="#1f2228")
        ax.set_xlabel("|q|, Вт/м²")
        ax.set_ylabel("Число узлов")
        ax.set_title("Распределение модуля теплового потока")
        ax.text(0.02, 0.97,
                f"|q|min = {magnitude.min():.3g}\n"
                f"|q|max = {magnitude.max():.3g}\n"
                f"|q|среднее = {magnitude.mean():.3g}",
                transform=ax.transAxes, va="top", color="#dcdee2",
                fontsize=9, bbox=dict(facecolor="#1f2228",
                                       edgecolor="#5a606b"))
