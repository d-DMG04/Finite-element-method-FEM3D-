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

from .theme import current_theme


def _styled_figure() -> Figure:
    """Фигура в цветах ТЕКУЩЕЙ темы (тёмная/светлая/бежевая)."""
    th = current_theme()
    fig = Figure(figsize=(7, 4), facecolor=th.panel)
    return fig


def _style_ax(ax) -> None:
    """Оси в цветах текущей темы — графики читаемы в любой теме."""
    th = current_theme()
    ax.set_facecolor(th.bg)
    for side in ("bottom", "top", "left", "right"):
        ax.spines[side].set_color(th.border_strong)
    ax.tick_params(colors=th.text)
    ax.xaxis.label.set_color(th.text)
    ax.yaxis.label.set_color(th.text)
    ax.title.set_color(th.text)
    ax.grid(True, color=th.border, alpha=0.6)


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

    def apply_theme(self) -> None:
        """Перекрасить графики после смены темы приложения."""
        self._refresh()

    def _refresh(self) -> None:
        th = current_theme()
        self.fig.clear()
        self.fig.set_facecolor(th.panel)
        ax = self.fig.add_subplot(111)
        _style_ax(ax)
        if self._T is None or self._nodes is None or self._T.size == 0:
            ax.text(0.5, 0.5, "Нет данных. Выполните расчёт.",
                    ha="center", va="center", color=th.text_dim, fontsize=11,
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
        th = current_theme()
        ax.hist(self._T, bins=40, color=th.accent, edgecolor=th.bg)
        ax.set_xlabel("T, °C")
        ax.set_ylabel("Число узлов")
        ax.set_title("Распределение температуры по узлам сетки")
        Tmin, Tmax = float(self._T.min()), float(self._T.max())
        Tmean = float(self._T.mean())
        ax.axvline(Tmean, color=th.run, linestyle="--", linewidth=1.5,
                   label=f"среднее: {Tmean:.2f} °C")
        ax.legend(facecolor=th.panel, labelcolor=th.text,
                  edgecolor=th.border_strong)
        ax.text(0.02, 0.97,
                f"Tmin = {Tmin:.2f} °C\nTmax = {Tmax:.2f} °C\n"
                f"размах = {Tmax - Tmin:.2f} °C",
                transform=ax.transAxes, va="top", color=th.text,
                fontsize=9, bbox=dict(facecolor=th.panel,
                                       edgecolor=th.border_strong))

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
        th = current_theme()
        ax.fill_between(centers[ok], mins[ok], maxs[ok],
                        color=th.accent, alpha=0.25, label="разброс")
        ax.plot(centers[ok], means[ok], color=th.accent, linewidth=2,
                label="среднее T(·)")
        ax.set_xlabel(f"{axis}, м")
        ax.set_ylabel("T, °C")
        ax.set_title(f"Профиль температуры вдоль оси {axis.upper()}")
        ax.legend(facecolor=th.panel, labelcolor=th.text,
                  edgecolor=th.border_strong)

    def _plot_flux_hist(self, ax) -> None:
        th = current_theme()
        if self._flux is None:
            ax.text(0.5, 0.5, "Тепловые потоки не вычислены.",
                    ha="center", va="center", color=th.text_dim,
                    transform=ax.transAxes)
            return
        magnitude = np.linalg.norm(self._flux, axis=1)
        ax.hist(magnitude, bins=40, color=th.robin, edgecolor=th.bg)
        ax.set_xlabel("|q|, Вт/м²")
        ax.set_ylabel("Число узлов")
        ax.set_title("Распределение модуля теплового потока")
        ax.text(0.02, 0.97,
                f"|q|min = {magnitude.min():.3g}\n"
                f"|q|max = {magnitude.max():.3g}\n"
                f"|q|среднее = {magnitude.mean():.3g}",
                transform=ax.transAxes, va="top", color=th.text,
                fontsize=9, bbox=dict(facecolor=th.panel,
                                       edgecolor=th.border_strong))
