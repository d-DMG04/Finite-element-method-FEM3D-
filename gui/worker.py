# -*- coding: utf-8 -*-
"""
gui.worker — фоновый поток для выполнения расчёта.
"""

from __future__ import annotations

import os
import traceback

from PyQt5.QtCore import QObject, pyqtSignal

from fem3d import CoreBridge, CoreError, Problem, SolverInfo


class SolverWorker(QObject):
    """Запускает полный цикл расчёта в QThread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # SolverInfo
    error = pyqtSignal(str)

    def __init__(self, problem: Problem,
                 tol: float = 1e-8, max_iter: int = 5000,
                 omp_threads: int = 0) -> None:
        super().__init__()
        self.problem = problem
        self.tol = tol
        self.max_iter = max_iter
        self.omp_threads = omp_threads

    def run(self) -> None:
        try:
            if self.omp_threads > 0:
                os.environ["OMP_NUM_THREADS"] = str(self.omp_threads)
            self.progress.emit("Сборка матрицы...")
            with CoreBridge() as bridge:
                self.problem.build_mesh_in_core(bridge)
                self.progress.emit(
                    f"Решение СЛАУ ({self.problem.nodes.shape[0]} узлов)..."
                )
                info = self.problem.solve(bridge, tol=self.tol,
                                          max_iter=self.max_iter)
            self.progress.emit("Готово.")
            self.finished.emit(info)
        except CoreError as exc:
            self.error.emit(f"Ошибка ядра: {exc}")
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"Внутренняя ошибка:\n{exc}\n\n{tb}")
