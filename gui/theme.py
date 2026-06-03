# -*- coding: utf-8 -*-
"""
gui.theme — система тем приложения.

Поддерживаются три темы: dark, light, sepia. Палитра вынесена в данные;
stylesheet строится функцией build_stylesheet().

  dark   — тёмная (по умолчанию): чёрный фон, светлый текст, фиолетовый акцент;
  light  — светлая: белый фон, тёмный текст;
  sepia  — бежевая «книжная»: тёплый кремовый фон, тёмно-коричневый текст,
           комфортна для длительной работы и не даёт бликов в тёмном
           помещении.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemePalette:
    name: str
    title: str
    bg: str
    panel: str
    input_bg: str
    text: str
    text_dim: str
    border: str
    border_strong: str
    accent: str
    accent_hover: str
    run: str
    run_hover: str

    # Цвета граничных условий — общие для всех тем (семантические).
    dirichlet: str = "#d05050"
    robin:     str = "#3a78d0"
    neumann:   str = "#7a7e88"
    none:      str = "#3c4049"

    # Фон 3D-вьюпорта.
    viewport_bg: str = "#1f2228"


THEME_DARK = ThemePalette(
    name="dark", title="Тёмная",
    bg="#1f2228", panel="#2a2e36", input_bg="#1a1d22",
    text="#dcdee2", text_dim="#9aa0a6",
    border="#3c4049", border_strong="#5a606b",
    accent="#7a6cf0", accent_hover="#8b7eff",
    run="#3aa55a", run_hover="#4ec070",
    viewport_bg="#1f2228",
)

THEME_LIGHT = ThemePalette(
    name="light", title="Светлая",
    bg="#f4f5f7", panel="#ffffff", input_bg="#ffffff",
    text="#1f2228", text_dim="#5a606b",
    border="#d0d4d9", border_strong="#9aa0a6",
    accent="#5b4ddc", accent_hover="#7163e8",
    run="#2f9648", run_hover="#3aa55a",
    viewport_bg="#ebecef",
)

# Бежевая «книжная» — solarized-light-style, низкий контраст, тёплая.
THEME_SEPIA = ThemePalette(
    name="sepia", title="Бежевая",
    bg="#efe4cf", panel="#f7eed8", input_bg="#fdf6e3",
    text="#3a2d1a", text_dim="#5e4e30",
    border="#c8b896", border_strong="#8a7656",
    accent="#8a5c2a", accent_hover="#a06c30",
    run="#5d7a3a", run_hover="#6d8a48",
    viewport_bg="#e6dabf",
)

THEMES = {"dark": THEME_DARK, "light": THEME_LIGHT, "sepia": THEME_SEPIA}

_current_theme: ThemePalette = THEME_DARK


def current_theme() -> ThemePalette:
    return _current_theme


def set_theme(name: str) -> ThemePalette:
    global _current_theme
    if name not in THEMES:
        raise ValueError(f"Неизвестная тема: {name}; доступны: {list(THEMES)}")
    _current_theme = THEMES[name]
    return _current_theme


def bc_colors():
    from fem3d import BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN, BC_RADIATION
    t = _current_theme
    return {
        BC_NONE: t.none, BC_DIRICHLET: t.dirichlet,
        BC_NEUMANN: t.neumann, BC_ROBIN: t.robin,
        BC_RADIATION: "#e066b3",  # розовый для радиации
    }


def build_stylesheet(theme: ThemePalette = None) -> str:
    t = theme if theme is not None else _current_theme
    return f"""
QWidget {{
    background-color: {t.bg}; color: {t.text};
    font-family: "Segoe UI", "Helvetica", "Arial", sans-serif; font-size: 10pt;
}}
QFrame#Panel {{ background-color: {t.panel}; border-radius: 6px; }}
QFrame#Card {{
    background-color: {t.panel}; border: 1px solid {t.border}; border-radius: 4px;
}}
QGroupBox {{
    border: 1px solid {t.border}; border-radius: 4px;
    margin-top: 12px; padding-top: 14px;
    color: {t.text_dim}; font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 8px; padding: 0 4px;
}}
QLabel {{ background-color: transparent; }}
QPushButton {{
    background-color: {t.panel}; color: {t.text};
    border: 1px solid {t.border}; border-radius: 4px; padding: 6px 12px;
}}
QPushButton:hover {{ border-color: {t.border_strong}; }}
QPushButton:pressed {{ background-color: {t.bg}; }}
QPushButton:disabled {{ color: {t.text_dim}; border-color: {t.border}; }}
QPushButton#AccentButton {{
    background-color: {t.accent}; color: white; font-weight: 600; border: none;
}}
QPushButton#AccentButton:hover {{ background-color: {t.accent_hover}; }}
QPushButton#RunButton {{
    background-color: {t.run}; color: white; font-weight: 600;
    padding: 8px 18px; border: none;
}}
QPushButton#RunButton:hover {{ background-color: {t.run_hover}; }}
QPushButton#RunButton:disabled {{ background-color: {t.border}; color: {t.text_dim}; }}
QPushButton#Chip {{
    background-color: {t.input_bg}; color: {t.text};
    border: 1px solid {t.border}; padding: 4px 10px; text-align: left;
}}
QPushButton#Chip:hover {{ border-color: {t.border_strong}; }}
QToolButton {{
    background-color: {t.input_bg}; color: {t.text};
    border: 1px solid {t.border}; border-radius: 3px; padding: 4px 8px;
}}
QToolButton:hover {{ border-color: {t.border_strong}; }}
QToolButton:checked {{
    background-color: {t.accent}; color: white; border-color: {t.accent};
}}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
    background-color: {t.input_bg}; color: {t.text};
    border: 1px solid {t.border}; border-radius: 3px; padding: 3px 6px;
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QLineEdit:hover {{
    border-color: {t.border_strong};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {t.input_bg}; color: {t.text};
    selection-background-color: {t.accent}; selection-color: white;
}}
QProgressBar {{
    background-color: {t.input_bg}; border: 1px solid {t.border};
    border-radius: 3px; text-align: center; color: {t.text};
}}
QProgressBar::chunk {{ background-color: {t.accent}; border-radius: 2px; }}
QSplitter::handle {{ background-color: {t.border}; width: 3px; }}
QStatusBar {{ background-color: {t.input_bg}; color: {t.text_dim}; }}
QListWidget {{
    background-color: {t.input_bg};
    border: 1px solid {t.border}; border-radius: 3px;
}}
QListWidget::item:selected {{ background-color: {t.accent}; color: white; }}
QTabWidget::pane {{
    border: 1px solid {t.border}; border-radius: 4px;
    background-color: {t.panel}; top: -1px;
}}
QTabBar::tab {{
    background-color: {t.bg}; color: {t.text_dim};
    padding: 6px 14px; border: 1px solid {t.border}; border-bottom: none;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {t.panel}; color: {t.text};
    border-bottom: 1px solid {t.panel};
}}
QTabBar::tab:hover:!selected {{ color: {t.text}; }}
QMenuBar {{ background-color: {t.input_bg}; color: {t.text}; }}
QMenuBar::item:selected {{ background-color: {t.accent}; color: white; }}
QMenu {{
    background-color: {t.input_bg}; color: {t.text}; border: 1px solid {t.border};
}}
QMenu::item:selected {{ background-color: {t.accent}; color: white; }}
QCheckBox {{ background-color: transparent; }}
QCheckBox::indicator {{ width: 14px; height: 14px; }}
QCheckBox::indicator:unchecked {{
    background-color: {t.input_bg};
    border: 1px solid {t.border_strong}; border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    background-color: {t.accent};
    border: 1px solid {t.accent}; border-radius: 3px;
}}
QSlider::groove:horizontal {{
    background: {t.input_bg}; height: 6px; border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {t.accent}; width: 14px; margin: -4px 0; border-radius: 7px;
}}
QDialog {{ background-color: {t.bg}; }}
QTextEdit, QTextBrowser {{
    background-color: {t.input_bg}; color: {t.text}; border: 1px solid {t.border};
}}
QHeaderView::section {{
    background-color: {t.panel}; color: {t.text};
    padding: 4px; border: 1px solid {t.border};
}}
QTableWidget {{
    background-color: {t.input_bg}; color: {t.text};
    border: 1px solid {t.border}; gridline-color: {t.border};
}}
QTableWidget::item:selected {{ background-color: {t.accent}; color: white; }}
"""


def build_palette(theme: ThemePalette = None):
    from PyQt5.QtGui import QColor, QPalette
    t = theme if theme is not None else _current_theme
    p = QPalette()
    p.setColor(QPalette.Window,         QColor(t.bg))
    p.setColor(QPalette.WindowText,     QColor(t.text))
    p.setColor(QPalette.Base,           QColor(t.input_bg))
    p.setColor(QPalette.AlternateBase,  QColor(t.panel))
    p.setColor(QPalette.Text,           QColor(t.text))
    p.setColor(QPalette.Button,         QColor(t.panel))
    p.setColor(QPalette.ButtonText,     QColor(t.text))
    p.setColor(QPalette.Highlight,      QColor(t.accent))
    p.setColor(QPalette.HighlightedText, QColor("white"))
    p.setColor(QPalette.ToolTipBase,    QColor(t.panel))
    p.setColor(QPalette.ToolTipText,    QColor(t.text))
    return p


# =============================================================================
# Прокси-«константы» для совместимости со старым кодом — читают текущую тему.
# =============================================================================

class _DynamicColor:
    def __init__(self, attr: str):
        self._attr = attr

    def __str__(self) -> str:
        return getattr(_current_theme, self._attr)

    def __repr__(self) -> str:
        return self.__str__()

    def __format__(self, _spec: str) -> str:
        return self.__str__()

    def __add__(self, other) -> str:
        return self.__str__() + str(other)

    def __radd__(self, other) -> str:
        return str(other) + self.__str__()


COLOR_BG_DARK  = _DynamicColor("bg")
COLOR_PANEL    = _DynamicColor("panel")
COLOR_TEXT     = _DynamicColor("text")
COLOR_TEXT_DIM = _DynamicColor("text_dim")
COLOR_ACCENT   = _DynamicColor("accent")
COLOR_NONE     = _DynamicColor("none")

# Совместимость со старым импортом STYLESHEET — это просто текущая
# stylesheet-строка. Для обновления нужно вызвать build_stylesheet() заново.
STYLESHEET = build_stylesheet()
