# -*- coding: utf-8 -*-
"""
demo_convection.py — демонстрация конвективного теплообмена при обтекании.

Показывает добавленную функциональность (по требованию преподавателя):
  • коэффициент теплообмена h;
  • площадь поверхности фигур;
  • направление и скорость потока воздуха;
  • конвективный теплообмен при обтекании (Re → Nu → h → Q);
  • число Нуссельта.

Запуск:
    python3 demo_convection.py
"""
import numpy as np

from fem3d import CoreBridge, Problem, BoxGeometry, FACE_X_MINUS
from fem3d import convection as cv
from fem3d.postprocess import compute_nusselt


def line(c="─"):
    print(c * 64)


def demo_air_properties():
    line("═")
    print("1) СВОЙСТВА ВОЗДУХА при разных температурах плёнки")
    line()
    for T in (0, 20, 60, 100, 200):
        a = cv.air_properties(T)
        print(f"  T={T:4d} °C: ρ={a.rho:.4f}  ν={a.nu:.3e}  "
              f"λ={a.k:.4f}  Pr={a.Pr:.3f}")


def demo_surface_area():
    line("═")
    print("2) ПЛОЩАДЬ ПОВЕРХНОСТИ ФИГУР")
    line()
    p = Problem(geometry=BoxGeometry(Lx=0.20, Ly=0.15, Lz=0.05))
    A = cv.surface_areas(p)
    for fid, area in A["per_face"].items():
        print(f"  грань {A['labels'][fid]:>3}: {area:.4f} м²")
    print(f"  ИТОГО площадь поверхности: {A['total']:.4f} м²")
    # Проверка: 2(ab+bc+ac)
    chk = 2 * (0.20 * 0.15 + 0.15 * 0.05 + 0.20 * 0.05)
    print(f"  контроль 2(ab+bc+ac) = {chk:.4f} м²  → "
          f"{'OK' if abs(chk - A['total']) < 1e-9 else 'ОШИБКА'}")


def demo_flow_and_convection():
    line("═")
    print("3) НАПРАВЛЕНИЕ/СКОРОСТЬ ПОТОКА И ОБТЕКАНИЕ")
    line()
    p = Problem(geometry=BoxGeometry(Lx=0.30, Ly=0.20, Lz=0.02))

    print("  Грани относительно потока вдоль +X:")
    roles = cv.faces_exposed_to_flow(p, "+x")
    for fid, info in roles.items():
        print(f"    {cv._FACE_LABEL[fid]:>3}: {info['role']:<14} (cosθ={info['cos']:+.2f})")
    print()

    for U in (1.0, 5.0, 15.0):
        r = cv.analyze_forced_convection(
            p, speed=U, direction="+x", shape=cv.SHAPE_PLATE,
            T_inf=20.0, T_surface=80.0, compare_natural=False)
        print(f"  U={U:5.1f} м/с:  Re={r.Re:.3g}  Nu={r.Nu:6.1f}  "
              f"h={r.h:6.2f} Вт/(м²·К)  Q={r.Q_total:6.1f} Вт")


def demo_shapes():
    line("═")
    print("4) ЧИСЛО НУССЕЛЬТА ДЛЯ РАЗНЫХ ФОРМ (U=10 м/с, воздух)")
    line()
    p = Problem(geometry=BoxGeometry(Lx=0.10, Ly=0.10, Lz=0.10))
    for shape in (cv.SHAPE_PLATE, cv.SHAPE_CYLINDER, cv.SHAPE_SPHERE):
        r = cv.analyze_forced_convection(
            p, speed=10.0, direction="+x", shape=shape,
            T_inf=20.0, T_surface=80.0, compare_natural=False)
        print(f"  {cv.SHAPE_NAMES[shape]:<42}")
        print(f"     L={r.char_length:.3f} м  Re={r.Re:.3g}  "
              f"Nu={r.Nu:.1f}  h={r.h:.2f} Вт/(м²·К)")


def demo_full_report_and_closure():
    line("═")
    print("5) ПОЛНЫЙ ОТЧЁТ + ПРОВЕРКА ЗАМЫКАНИЯ ЧЕРЕЗ МКЭ")
    line()
    p = Problem(geometry=BoxGeometry(Lx=0.20, Ly=0.20, Lz=0.02,
                                     nx=24, ny=24, nz=4),
                lambda_=237.0)
    p.Q = 1.0e5  # внутренний источник, чтобы тело было теплее среды

    res = cv.apply_forced_convection_bc(
        p, speed=10.0, direction="+x", shape=cv.SHAPE_PLATE,
        T_inf=20.0, T_surface=80.0)
    print(res.report_text())

    with CoreBridge() as b:
        p.build_mesh_in_core(b)
        p.solve(b)
    print(f"\n  Решение МКЭ: T = {p.T.min():.1f}…{p.T.max():.1f} °C "
          f"(среднее {p.T.mean():.1f})")

    nu = compute_nusselt(p, FACE_X_MINUS,
                         characteristic_length=res.char_length,
                         fluid_lambda=res.fluid.k)
    print(f"  Восстановлено из поля T: h_МКЭ={nu['h_actual']:.2f}  "
          f"vs заданный h={res.h:.2f}  "
          f"(расхождение {100*abs(nu['h_actual']-res.h)/res.h:.1f} %)")


if __name__ == "__main__":
    demo_air_properties()
    demo_surface_area()
    demo_flow_and_convection()
    demo_shapes()
    demo_full_report_and_closure()
    line("═")
    print("Готово.")
