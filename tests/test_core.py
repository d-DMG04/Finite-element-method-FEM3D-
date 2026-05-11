# -*- coding: utf-8 -*-
"""
Тесты программного комплекса.

Запуск:
    pytest -v

Не требует PyQt5.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fem3d import (BC_DIRICHLET, BC_NEUMANN, BC_ROBIN, BoundaryCondition,
                   BoxGeometry, CoreBridge, FACE_X_MINUS, FACE_X_PLUS,
                   FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS,
                   Problem, template_bottom_heat_top_cool)


# =============================================================================
# Тесты ядра через CoreBridge (соответствуют верификации T1, T2, T4 ПЗ).
# =============================================================================

class TestCoreBasics:

    def test_box_mesh_size(self):
        """fem_generate_box должен создать (nx+1)(ny+1)(nz+1) узлов."""
        with CoreBridge() as br:
            br.generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 5, 4, 3)
            assert br.n_nodes == 6 * 5 * 4
            assert br.n_elements == 6 * 5 * 4 * 3
            # Граничных треугольников должно быть = 2 * (nx*ny + ny*nz + nx*nz) * 2.
            # Каждая прямоугольная грань разбивается на 2 треугольника, всего
            # граничных прямоугольников: nx*ny + ny*nz + nx*nz, умножаем на 2
            # (две противоположные грани) и на 2 (треугольника на квадрат).
            expected_faces = 4 * (5 * 4 + 4 * 3 + 5 * 3)
            assert br.n_boundary_faces == expected_faces

    def test_node_array_shape_and_dtype(self):
        with CoreBridge() as br:
            br.generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3, 3, 3)
            nodes = br.get_nodes()
            assert nodes.shape == (4 * 4 * 4, 3)
            assert nodes.dtype == np.float64

    def test_invalid_lambda_raises(self):
        with CoreBridge() as br:
            br.generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3, 3, 3)
            with pytest.raises(Exception):
                br.set_material(-1.0)


class TestVerification:
    """Воспроизводят T1, T2, T4 из раздела 3.4 ПЗ."""

    def test_T1_linear_solution_machine_precision(self):
        """Линейное решение T(x)=x должно воспроизводиться точно."""
        with CoreBridge() as br:
            br.generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 8, 8, 8)
            nodes = br.get_nodes()
            br.set_material(1.0, 0.0)
            br.set_bc(FACE_X_MINUS, BC_DIRICHLET, T0=0.0)
            br.set_bc(FACE_X_PLUS,  BC_DIRICHLET, T0=1.0)
            for f in (FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS):
                br.set_bc(f, BC_NEUMANN)
            info = br.solve(tol=1e-12, max_iter=5000)
            assert info.converged
            T = br.get_temperature()
        err = float(np.max(np.abs(T - nodes[:, 0])))
        assert err < 1e-8, f"T1 ошибка слишком велика: {err:.3e}"

    def test_T2_robin_1d_analytic(self):
        """Дирихле на одной грани + Робен на противоположной → 1D-аналитика."""
        L, lam, T0, alpha, T_inf = 0.1, 50.0, 100.0, 25.0, 20.0
        with CoreBridge() as br:
            br.generate_box(0.0, L, 0.0, L, 0.0, L, 12, 12, 12)
            nodes = br.get_nodes()
            br.set_material(lam, 0.0)
            br.set_bc(FACE_Z_MINUS, BC_DIRICHLET, T0=T0)
            br.set_bc(FACE_Z_PLUS,  BC_ROBIN, alpha=alpha, T_inf=T_inf)
            for f in (FACE_X_MINUS, FACE_X_PLUS, FACE_Y_MINUS, FACE_Y_PLUS):
                br.set_bc(f, BC_NEUMANN)
            info = br.solve(tol=1e-12, max_iter=5000)
            assert info.converged
            T = br.get_temperature()
        T_ex = T0 + (T_inf - T0) * nodes[:, 2] / (L + lam / alpha)
        err = float(np.max(np.abs(T - T_ex)))
        assert err < 1e-6, f"T2 ошибка: {err:.3e}"

    def test_T4_flux_for_linear(self):
        """Тепловой поток для линейного решения должен быть постоянным."""
        with CoreBridge() as br:
            br.generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 6, 6, 6)
            br.set_material(1.0, 0.0)
            br.set_bc(FACE_Z_MINUS, BC_DIRICHLET, T0=0.0)
            br.set_bc(FACE_Z_PLUS,  BC_DIRICHLET, T0=100.0)
            for f in (FACE_X_MINUS, FACE_X_PLUS, FACE_Y_MINUS, FACE_Y_PLUS):
                br.set_bc(f, BC_NEUMANN)
            br.solve(tol=1e-12, max_iter=5000)
            flux = br.compute_fluxes()
        # Ожидаем (0, 0, -100).
        assert np.allclose(flux[:, 0], 0.0, atol=1e-6)
        assert np.allclose(flux[:, 1], 0.0, atol=1e-6)
        assert np.allclose(flux[:, 2], -100.0, atol=1e-6)


# =============================================================================
# Тесты высокоуровневого API (Problem).
# =============================================================================

class TestProblem:

    def test_solve_template_does_not_crash(self):
        problem = Problem(
            geometry=BoxGeometry(Lx=0.10, Ly=0.10, Lz=0.10, nx=8, ny=8, nz=8),
            lambda_=237.0, Q=0.0,
            bcs=template_bottom_heat_top_cool(),
        )
        with CoreBridge() as br:
            problem.build_mesh_in_core(br)
            info = problem.solve(br)
        assert info.converged
        Tmin, Tmax = problem.temperature_range()
        # 100 °C снизу, конвекция сверху → решение между T_inf=20 и 100.
        assert 19.0 < Tmin < 100.5
        assert 19.0 < Tmax <= 100.5
        assert Tmax >= Tmin

    def test_hot_spot_within_box(self):
        problem = Problem(
            geometry=BoxGeometry(Lx=0.10, Ly=0.10, Lz=0.10, nx=6, ny=6, nz=6),
            lambda_=237.0, Q=0.0,
            bcs=template_bottom_heat_top_cool(),
        )
        with CoreBridge() as br:
            problem.build_mesh_in_core(br)
            problem.solve(br)
        hs = problem.hot_spot()
        assert hs is not None
        idx, x, y, z = hs
        # Самая горячая точка должна быть на нижней грани (T = 100 °C).
        assert z == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Тест экспорта (без PyQt5).
# =============================================================================

class TestExport:

    def test_csv_and_report(self, tmp_path):
        from fem3d.postprocess import export_csv, export_report

        problem = Problem(
            geometry=BoxGeometry(Lx=0.05, Ly=0.05, Lz=0.05, nx=5, ny=5, nz=5),
            lambda_=237.0, Q=0.0,
            bcs=template_bottom_heat_top_cool(),
        )
        with CoreBridge() as br:
            problem.build_mesh_in_core(br)
            problem.solve(br)
        csv_path = tmp_path / "out.csv"
        rpt_path = tmp_path / "out.txt"
        export_csv(problem, str(csv_path))
        export_report(problem, str(rpt_path))
        assert csv_path.exists() and csv_path.stat().st_size > 0
        assert rpt_path.exists() and rpt_path.stat().st_size > 0
        # Проверим содержимое CSV: первая строка — заголовок.
        head = csv_path.read_text(encoding="utf-8").splitlines()[0]
        assert head.startswith("x,y,z,T")


# =============================================================================
# Тесты локальных источников (Ф2.4 ТЗ, раздел 3.3.11 ПЗ).
# =============================================================================

class TestLocalSources:

    def test_point_source_creates_local_maximum(self):
        """Точечный источник в изотропно охлаждаемом кубе должен создать
        максимум температуры именно в узле, куда он вложен."""
        with CoreBridge() as br:
            br.generate_box(0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 12, 12, 12)
            br.set_material(401.0, 0.0)  # медь
            for f in (FACE_X_MINUS, FACE_X_PLUS, FACE_Y_MINUS,
                      FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS):
                br.set_bc(f, BC_ROBIN, alpha=20.0, T_inf=20.0)

            nodes = br.get_nodes()
            center = np.array([0.05, 0.05, 0.05])
            idx_center = int(np.argmin(np.linalg.norm(nodes - center, axis=1)))
            br.add_point_source(idx_center, 50.0)
            info = br.solve()
            assert info.converged
            T = br.get_temperature()
        idx_max = int(np.argmax(T))
        assert idx_max == idx_center, \
            f"максимум должен быть в узле источника {idx_center}, оказался в {idx_max}"
        # Температура должна заметно превысить T_inf.
        assert T.max() > 25.0

    def test_volume_sphere_source(self):
        """Объёмный сферический источник должен дать положительный нагрев."""
        with CoreBridge() as br:
            br.generate_box(0.0, 0.1, 0.0, 0.1, 0.0, 0.02, 16, 16, 5)
            br.set_material(237.0, 0.0)
            for f in (FACE_X_MINUS, FACE_X_PLUS, FACE_Y_MINUS, FACE_Y_PLUS):
                br.set_bc(f, BC_NEUMANN)
            for f in (FACE_Z_MINUS, FACE_Z_PLUS):
                br.set_bc(f, BC_ROBIN, alpha=15.0, T_inf=20.0)
            br.add_volume_source_sphere(0.05, 0.05, 0.01, 0.01, 1.0e7)
            info = br.solve()
            assert info.converged
            T = br.get_temperature()
        # Тепло поступает — температура должна быть выше окружающей.
        assert T.min() > 19.5
        assert T.max() > T.min()

    def test_clear_sources(self):
        """clear_sources() должна обнулить вклад источников: при
        одинаковой геометрии и ГУ результат должен совпасть с задачей
        вообще без источников."""
        Tinf = 20.0
        # Задача без источников.
        with CoreBridge() as br1:
            br1.generate_box(0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 8, 8, 8)
            br1.set_material(237.0, 0.0)
            for f in range(6):
                br1.set_bc(f, BC_ROBIN, alpha=20.0, T_inf=Tinf)
            br1.solve()
            T1 = br1.get_temperature().copy()
        # Та же задача, но с добавленным и потом очищенным источником.
        with CoreBridge() as br2:
            br2.generate_box(0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 8, 8, 8)
            br2.set_material(237.0, 0.0)
            for f in range(6):
                br2.set_bc(f, BC_ROBIN, alpha=20.0, T_inf=Tinf)
            br2.add_point_source(0, 100.0)
            br2.clear_sources()
            br2.solve()
            T2 = br2.get_temperature()
        assert np.allclose(T1, T2, atol=1e-9), \
            "после clear_sources() решение должно совпасть с задачей без источников"


# =============================================================================
# Тесты импорта сетки (load_mesh без файла, через numpy-массивы).
# =============================================================================

class TestLoadMesh:

    def _make_simple_cube_mesh(self):
        """Куб 1×1×1 из 6 тетраэдров (декомпозиция Куна) + грани z=0."""
        nodes = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        ], dtype=np.float64)
        tets = np.array([
            [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
            [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
        ], dtype=np.int32)
        # 2 треугольника на нижней грани (FACE_Z_MINUS = 4).
        bnd_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        bnd_face_ids = np.array([4, 4], dtype=np.int32)
        return nodes, tets, bnd_nodes, bnd_face_ids

    def test_load_mesh_basic(self):
        nodes, tets, bnd, bnd_ids = self._make_simple_cube_mesh()
        with CoreBridge() as br:
            br.load_mesh(nodes, tets, bnd, bnd_ids)
            assert br.n_nodes == 8
            assert br.n_elements == 6
            assert br.n_boundary_faces == 2
            # Проверим, что узлы возвращаются в том же порядке.
            n2 = br.get_nodes()
            assert np.allclose(n2, nodes)

    def test_loaded_mesh_can_be_solved(self):
        """Простая задача на загруженной сетке должна решаться."""
        nodes, tets, bnd, bnd_ids = self._make_simple_cube_mesh()
        with CoreBridge() as br:
            br.load_mesh(nodes, tets, bnd, bnd_ids)
            br.set_material(1.0, 0.0)
            # Задаём Дирихле T = 50 °C на единственной граничной группе.
            br.set_bc(4, BC_DIRICHLET, T0=50.0)
            info = br.solve()
            assert info.converged
            T = br.get_temperature()
        # Все узлы граничной грани z=0 должны иметь T=50.
        for face_node_idx in (0, 1, 2, 3):
            assert abs(T[face_node_idx] - 50.0) < 1e-6

    def test_supported_extensions_listed(self):
        """Проверка, что SUPPORTED_IMPORT_EXTENSIONS экспортируется."""
        from fem3d import SUPPORTED_IMPORT_EXTENSIONS
        # Минимум — поддержка MSH, VTU, STL, STEP.
        assert ".msh" in SUPPORTED_IMPORT_EXTENSIONS
        assert ".vtu" in SUPPORTED_IMPORT_EXTENSIONS
        assert ".stl" in SUPPORTED_IMPORT_EXTENSIONS
        assert ".step" in SUPPORTED_IMPORT_EXTENSIONS

    def test_import_mesh_file_unsupported(self):
        """Неподдерживаемое расширение → понятная ошибка."""
        from fem3d import import_mesh_file
        try:
            import_mesh_file("/tmp/nonexistent.xyz")
        except Exception as exc:
            assert "Неподдерживаемое расширение" in str(exc) or \
                   "Unsupported" in str(exc)
            return
        assert False, "ожидалось исключение"


# =============================================================================
# Тесты переопределения Дирихле по узлам (используется в T3).
# =============================================================================

class TestNodeDirichlet:

    def test_set_node_dirichlet_overrides_face(self):
        """fem_set_node_dirichlet переопределяет значение T в узле."""
        with CoreBridge() as br:
            br.generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 4, 4, 4)
            br.set_material(1.0, 0.0)
            # Все грани — Дирихле T = 0.
            for f in range(6):
                br.set_bc(f, BC_DIRICHLET, T0=0.0)
            # Зафиксируем узел 0 (это (0,0,0)) в значении 100.
            br.set_node_dirichlet(0, 100.0)
            br.solve(tol=1e-12)
            T = br.get_temperature()
            assert abs(T[0] - 100.0) < 1e-9, \
                f"узел 0 должен быть T=100, получено {T[0]}"

    def test_clear_node_dirichlet_resets(self):
        """clear_node_dirichlet возвращает решение к базовому."""
        with CoreBridge() as br:
            br.generate_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 4, 4, 4)
            br.set_material(1.0, 0.0)
            for f in range(6):
                br.set_bc(f, BC_DIRICHLET, T0=10.0)
            br.set_node_dirichlet(0, 999.0)
            br.clear_node_dirichlet()
            br.solve(tol=1e-12)
            T = br.get_temperature()
            assert abs(T[0] - 10.0) < 1e-9
