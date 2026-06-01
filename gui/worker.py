# -*- coding: utf-8 -*-
"""
gui.worker — фоновый поток для выполнения расчёта.

Прогресс CG передаётся обратно в GUI через сигнал `cg_progress(iter, residual)`.
Прерывание расчёта — через метод `request_cancel()`.
"""

from __future__ import annotations

import math
import os
import traceback

from PyQt5.QtCore import QObject, pyqtSignal

from fem3d import CoreBridge, CoreError, Problem, SolverInfo


class SolverWorker(QObject):
    """Запускает полный цикл расчёта в QThread."""

    # Текстовый прогресс (для строки внизу).
    progress = pyqtSignal(str)
    # Прогресс CG: (iter, residual, percent).
    cg_progress = pyqtSignal(int, float, int)
    # Завершено: SolverInfo (converged может быть False — это нормально).
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, problem: Problem,
                 tol: float = 1e-8, max_iter: int = 5000,
                 omp_threads: int = 0) -> None:
        super().__init__()
        self.problem = problem
        self.tol = tol
        self.max_iter = max_iter
        self.omp_threads = omp_threads
        self._bridge: 'CoreBridge | None' = None
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Помечает расчёт для отмены. Вызывается из GUI-потока."""
        self._cancel_requested = True
        if self._bridge is not None:
            try:
                self._bridge.request_cancel()
            except Exception:
                pass

    def run(self) -> None:
        try:
            if self.omp_threads > 0:
                os.environ["OMP_NUM_THREADS"] = str(self.omp_threads)
            self.progress.emit("Сборка матрицы...")
            with CoreBridge() as bridge:
                self._bridge = bridge
                self.problem.build_mesh_in_core(bridge)
                n_nodes = self.problem.nodes.shape[0]
                self.progress.emit(
                    f"Решение СЛАУ ({n_nodes} узлов)..."
                )

                # Callback прогресса CG. Преобразуем относительную невязку
                # в проценты: лог-шкала от 100% (старт, res = 1) до 0% (res = tol).
                log_tol = math.log10(max(self.tol, 1e-30))
                def cg_cb(it: int, residual: float) -> bool:
                    if self._cancel_requested:
                        return False
                    try:
                        log_res = math.log10(max(residual, 1e-30))
                        # 0 при res=1, 100 при res=tol.
                        pct = max(0.0, min(100.0,
                            100.0 * (1.0 - log_res / log_tol) if log_tol < 0
                            else 0.0))
                        self.cg_progress.emit(int(it), float(residual), int(pct))
                    except Exception:
                        pass
                    return True

                info = self.problem.solve(bridge, tol=self.tol,
                                          max_iter=self.max_iter,
                                          progress_callback=cg_cb)
            self._bridge = None
            if self._cancel_requested:
                self.progress.emit("Прервано пользователем.")
            else:
                self.progress.emit("Готово.")
            self.finished.emit(info)
        except CoreError as exc:
            self.error.emit(f"Ошибка ядра: {exc}")
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"Внутренняя ошибка:\n{exc}\n\n{tb}")
