# -*- coding: utf-8 -*-
"""
gui — пакет с компонентами графического интерфейса.

Модули:
    theme         — палитры тем (dark/light/sepia), генератор stylesheet;
    face_card     — карточка одной грани;
    viz3d         — 3D-визуализация (PyVista или matplotlib);
    plots         — двумерные графики результатов;
    calculations  — список выполненных расчётов;
    dialogs       — диалоги настроек, справки, источников, регионов;
    worker        — поток для запуска расчёта.
"""

from .theme import (STYLESHEET, COLOR_BG_DARK, COLOR_PANEL, COLOR_TEXT,
                    COLOR_TEXT_DIM, COLOR_ACCENT,
                    THEMES, ThemePalette,
                    current_theme, set_theme, bc_colors,
                    build_stylesheet, build_palette)
from .face_card import FaceCard
from .viz3d import create_view, HAS_PYVISTA, Visualization3D
from .plots import PlotsView
from .calculations import CalculationsView, CalculationRecord
from .whatif import WhatIfView
from .dialogs import (AppSettings, SettingsDialog, HelpDialog,
                      PointSourceDialog, VolumeSourceDialog,
                      MaterialRegionDialog, MaterialRegionsDialog,
                      GeometryDialog, MaterialDialog,
                      MaterialEditorDialog, MaterialLibraryDialog,
                      BoundaryConditionsDialog, TransientParamsDialog,
                      TemplateGalleryDialog)
from .worker import SolverWorker

__all__ = [
    "STYLESHEET", "COLOR_BG_DARK", "COLOR_PANEL", "COLOR_TEXT",
    "COLOR_TEXT_DIM", "COLOR_ACCENT",
    "THEMES", "ThemePalette",
    "current_theme", "set_theme", "bc_colors",
    "build_stylesheet", "build_palette",
    "FaceCard",
    "create_view", "HAS_PYVISTA", "Visualization3D",
    "PlotsView",
    "CalculationsView", "CalculationRecord",
    "WhatIfView",
    "AppSettings", "SettingsDialog", "HelpDialog",
    "PointSourceDialog", "VolumeSourceDialog",
    "MaterialRegionDialog", "MaterialRegionsDialog",
    "GeometryDialog", "MaterialDialog",
    "MaterialEditorDialog", "MaterialLibraryDialog",
    "BoundaryConditionsDialog", "TransientParamsDialog",
    "TemplateGalleryDialog",
    "SolverWorker",
]
