# -*- coding: utf-8 -*-
"""
gui_smoke_test.py — сквозной тест реального GUI с настоящим PyQt5 + PyVista.

В отличие от mock-проверок (которые только импортируют модули), этот тест
СОЗДАЁТ настоящее окно и прогоняет ключевые пользовательские сценарии:
построение сетки, расчёт, whatif, термопары, сложные геометрии. Это ловит
баги именно PyVista-бэкенда, где живёт основная часть 3D-логики.

Запуск (нужен дисплей или xvfb):
    xvfb-run -a python3 gui_smoke_test.py

Требует: PyQt5, pyvista, pyvistaqt. Если их нет — тест пропускается
(не падает), потому что это окружение без GUI.
"""
import sys
import os

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")


def main() -> int:
    try:
        from PyQt5.QtWidgets import QApplication
    except Exception as e:
        print(f"SKIP: PyQt5 недоступен ({e})")
        return 0

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    app = QApplication.instance() or QApplication([])

    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "main_gui", os.path.join(here, "main_gui.py"))
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)

    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  OK   {name}")
        except Exception as exc:
            import traceback
            print(f"  FAIL {name}: {exc}")
            traceback.print_exc()
            failures.append(name)

    win = mg.MainWindow()
    print(f"Бэкенд 3D: {win.viz.backend_name}")

    from fem3d import HEATING_TEMPLATES_FULL, SHAPE_PRESETS, BoundaryCondition

    def s_mesh():
        win._on_generate_mesh()
        assert win.problem.nodes is not None

    def s_steady():
        for fid, bc in HEATING_TEMPLATES_FULL[3][1]().items():
            win.problem.bcs[fid] = bc
        win._run_steady()
        if win._thread:
            win._thread.wait(15000)
        for _ in range(20):
            app.processEvents()
        assert win.problem.T is not None

    def s_complex():
        nodes, tets, bn, fids = SHAPE_PRESETS[0].build(density=1.0)
        win._apply_external_mesh(nodes, tets, bn, fids)
        assert win.problem.nodes.shape[0] > 0

    def s_whatif():
        win._on_whatif_recompute({"lambda_": 237, "alpha": 500, "Q": 0,
                                   "T_inf": 15, "scenario": "all_conv"})

    def s_obs():
        win.problem.observation_points.append((0.0, 0.0, 0.05))
        win._refresh_observation_markers()

    def s_gallery():
        # Применение всех шаблонов ГУ через их фабрики.
        for label, factory, desc, cat in HEATING_TEMPLATES_FULL:
            bcs = factory()
            for fid, bc in bcs.items():
                win.problem.bcs[fid] = bc
            win._refresh_bc_overlay()

    def s_project_roundtrip():
        # Сохранение/загрузка проекта со всеми полями.
        import tempfile, os as _os
        from fem3d.project import save_project, load_project
        win.problem.rho = 8960; win.problem.cp = 385
        win.problem.material_name = "Медь(тест)"
        win.problem.observation_points = [(0.05, 0.05, 0.05)]
        path = tempfile.mktemp(suffix=".fem3d")
        save_project(win.problem, path)
        p2 = load_project(path)
        _os.remove(path)
        assert p2.rho == 8960 and p2.cp == 385, "rho/cp потеряны"
        assert p2.material_name == "Медь(тест)", "имя материала потеряно"
        assert len(p2.observation_points) == 1, "термопары потеряны"

    def s_radiation_complex():
        # Радиация на сложной геометрии — линеаризация должна сходиться.
        from fem3d import BC_RADIATION, BC_DIRICHLET
        nodes, tets, bn, fids = SHAPE_PRESETS[0].build(density=1.0)
        win._apply_external_mesh(nodes, tets, bn, fids)
        win.problem.bcs[4] = BoundaryCondition(type=BC_DIRICHLET, T0=300.0)
        for f in [0, 1, 2, 3, 5]:
            win.problem.bcs[f] = BoundaryCondition(
                type=BC_RADIATION, emissivity=0.85, T_inf=25.0)
        win._run_steady()
        if win._thread:
            win._thread.wait(20000)
        for _ in range(20):
            app.processEvents()
        bal = win.problem.energy_balance()
        assert bal is not None and abs(bal["rel_err"]) < 0.1, \
            f"радиация на цилиндре: баланс {bal['rel_err']*100:.1f}%"

    check("Построение сетки (box)", s_mesh)
    check("Стационарный расчёт", s_steady)
    check("Сложная геометрия (цилиндр)", s_complex)
    check("Whatif пересчёт", s_whatif)
    check("Точки наблюдения", s_obs)
    check("Галерея шаблонов ГУ (все 17)", s_gallery)
    check("Сохранение/загрузка проекта", s_project_roundtrip)
    check("Радиация на цилиндре (линеаризация)", s_radiation_complex)

    win.close()
    app.processEvents()

    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)} — {', '.join(failures)}")
        return 1
    print("ВСЕ СЦЕНАРИИ GUI ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
