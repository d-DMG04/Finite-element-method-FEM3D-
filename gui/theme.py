# -*- coding: utf-8 -*-
"""
gui.theme — единая тёмная тема и цветовые константы.
"""

from __future__ import annotations

# Цветовая схема (раздел 3.3.7 ПЗ).
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

# Импорт BC_* для словаря; делаем ленивым, чтобы не тянуть весь fem3d отсюда.
def bc_colors():
    from fem3d import BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN
    return {
        BC_NONE:      COLOR_NONE,
        BC_DIRICHLET: COLOR_DIRICHLET,
        BC_NEUMANN:   COLOR_NEUMANN,
        BC_ROBIN:     COLOR_ROBIN,
    }


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
QLabel {{ background-color: transparent; }}
QPushButton {{
    background-color: #3c4049;
    color: {COLOR_TEXT};
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
}}
QPushButton:hover {{ background-color: #4a4f5a; }}
QPushButton:pressed {{ background-color: #2f333b; }}
QPushButton:disabled {{ background-color: #2a2e36; color: #5a606b; }}
QPushButton#AccentButton {{
    background-color: {COLOR_ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton#AccentButton:hover {{ background-color: #8b7eff; }}
QPushButton#RunButton {{
    background-color: {COLOR_RUN};
    color: white;
    font-weight: 600;
    padding: 8px 18px;
}}
QPushButton#RunButton:hover  {{ background-color: #4ec070; }}
QPushButton#RunButton:disabled {{ background-color: #2e4a37; color: #6f8a76; }}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
    background-color: #1a1d22;
    color: {COLOR_TEXT};
    border: 1px solid #3c4049;
    border-radius: 3px;
    padding: 3px 6px;
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QLineEdit:hover {{
    border-color: #5a606b;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
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
QScrollArea {{ border: none; }}
QSplitter::handle {{ background-color: #1a1d22; width: 3px; }}
QStatusBar {{ background-color: #1a1d22; color: {COLOR_TEXT_DIM}; }}
QListWidget {{
    background-color: #1a1d22;
    border: 1px solid #3c4049;
    border-radius: 3px;
}}
QListWidget::item:selected {{
    background-color: {COLOR_ACCENT};
    color: white;
}}
QTabWidget::pane {{
    border: 1px solid #3c4049;
    border-radius: 4px;
    background-color: {COLOR_PANEL};
    top: -1px;
}}
QTabBar::tab {{
    background-color: #2a2e36;
    color: {COLOR_TEXT_DIM};
    padding: 6px 14px;
    border: 1px solid #3c4049;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT};
    border-bottom: 1px solid {COLOR_PANEL};
}}
QTabBar::tab:hover:!selected {{ color: {COLOR_TEXT}; }}
QMenuBar {{ background-color: #1a1d22; color: {COLOR_TEXT}; }}
QMenuBar::item:selected {{ background-color: {COLOR_ACCENT}; color: white; }}
QMenu {{
    background-color: #1a1d22;
    color: {COLOR_TEXT};
    border: 1px solid #3c4049;
}}
QMenu::item:selected {{ background-color: {COLOR_ACCENT}; color: white; }}
QCheckBox {{ background-color: transparent; }}
QCheckBox::indicator {{ width: 14px; height: 14px; }}
QCheckBox::indicator:unchecked {{
    background-color: #1a1d22;
    border: 1px solid #5a606b;
    border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border: 1px solid {COLOR_ACCENT};
    border-radius: 3px;
}}
QSlider::groove:horizontal {{
    background: #1a1d22;
    height: 6px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLOR_ACCENT};
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
"""
