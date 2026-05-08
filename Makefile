# =============================================================================
# Makefile — корневой уровень проекта fem_heat3d.
# -----------------------------------------------------------------------------
# Цели:
#   make build   — сборка C++ ядра (fem_core);
#   make test    — запуск unit-тестов (run_tests.py — мини-раннер без pytest);
#   make verify  — запуск верификационных задач T1–T4;
#   make demo    — headless-демо: 3 сценария + PNG/CSV/отчёт в demo_output/;
#   make gui     — запуск GUI на PyQt5;
#   make clean   — очистка артефактов сборки и demo_output;
#   make help    — список целей.
# =============================================================================

PROJECT_ROOT := $(CURDIR)
CORE_DIR     := $(PROJECT_ROOT)/fem_core
BUILD_DIR    := $(CORE_DIR)/build
JOBS         ?= 4

CMAKE        ?= cmake
PYTHON       ?= python3

ifeq ($(OS),Windows_NT)
    LIB_NAME := fem_core.dll
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        LIB_NAME := fem_core.dylib
    else
        LIB_NAME := fem_core.so
    endif
endif

.PHONY: all build test test-pytest verify demo gui clean help

all: build

help:
	@echo "Цели сборки fem_heat3d:"
	@echo "  make build    — сборка C++ ядра (fem_core)"
	@echo "  make test     — запуск unit-тестов через run_tests.py"
	@echo "  make test-pytest — запуск тестов через pytest (если установлен)"
	@echo "  make verify   — верификационные задачи T1–T4"
	@echo "  make demo     — headless-демо (3 сценария, картинки и CSV)"
	@echo "  make gui      — запуск GUI"
	@echo "  make clean    — очистка артефактов"

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. -DCMAKE_BUILD_TYPE=Release
	@$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)
	@if [ -f "$(BUILD_DIR)/$(LIB_NAME)" ]; then \
	    cp "$(BUILD_DIR)/$(LIB_NAME)" "$(CORE_DIR)/$(LIB_NAME)"; \
	    echo "Ядро собрано: $(CORE_DIR)/$(LIB_NAME)"; \
	fi

test:
	@$(PYTHON) run_tests.py

test-pytest:
	@$(PYTHON) -m pytest tests/ -v

verify:
	@$(PYTHON) -m fem3d.verify

demo:
	@$(PYTHON) headless_demo.py

gui:
	@$(PYTHON) main_gui.py

clean:
	@rm -rf $(BUILD_DIR)
	@rm -f $(CORE_DIR)/$(LIB_NAME)
	@rm -rf demo_output
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Очистка завершена."
