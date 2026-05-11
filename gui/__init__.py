# -*- coding: utf-8 -*-
"""
gui — пакет с компонентами графического интерфейса.

Модули:
    theme         — цвета и stylesheet;
    face_card     — карточка одной грани;
    viz3d         — 3D-визуализация (PyVista или matplotlib);
    plots         — двумерные графики результатов;
    calculations  — список выполненных расчётов;
    dialogs       — диалоги настроек, справки и добавления источников;
    worker        — поток для запуска расчёта.
"""

from .theme import (STYLESHEET, COLOR_BG_DARK, COLOR_PANEL, COLOR_TEXT,
                    COLOR_TEXT_DIM, COLOR_ACCENT)
from .face_card import FaceCard
from .viz3d import create_view, HAS_PYVISTA, Visualization3D
from .plots import PlotsView
from .calculations import CalculationsView, CalculationRecord
from .dialogs import (AppSettings, SettingsDialog, HelpDialog,
                      PointSourceDialog, VolumeSourceDialog)
from .worker import SolverWorker

__all__ = [
    "STYLESHEET", "COLOR_BG_DARK", "COLOR_PANEL", "COLOR_TEXT",
    "COLOR_TEXT_DIM", "COLOR_ACCENT",
    "FaceCard",
    "create_view", "HAS_PYVISTA", "Visualization3D",
    "PlotsView",
    "CalculationsView", "CalculationRecord",
    "AppSettings", "SettingsDialog", "HelpDialog",
    "PointSourceDialog", "VolumeSourceDialog",
    "SolverWorker",
]
