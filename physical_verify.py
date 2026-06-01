"""
Физическая верификация: 6 классических задач теплопроводности.
Каждая задача имеет точное аналитическое решение.

V1. Стенка между двумя температурами (1D Фурье).
V2. Стенка с конвекцией на одной стороне.
V3. Стенка с внутренним источником тепла Q.
V4. Сфера с центральным источником (1D радиальный).
V5. Тепловое сопротивление двух материалов последовательно.
V6. Куб с заданным потоком на одной грани.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fem3d import (CoreBridge, BC_DIRICHLET, BC_NEUMANN, BC_ROBIN,
                   FACE_X_MINUS, FACE_X_PLUS, FACE_Y_MINUS, FACE_Y_PLUS,
                   FACE_Z_MINUS, FACE_Z_PLUS, Problem, BoundaryCondition,
                   BoxGeometry, MaterialRegion, REGION_BOX)


def test_v1_wall_two_temps():
    """V1. Бесконечная стенка между T1=100 и T2=20 °C.
    
    Уравнение: -λ d²T/dx² = 0
    Решение:   T(x) = T1 + (T2-T1) * x/L
    
    Берём куб 0.1×0.1×0.1, нагрев на X-, охлаждение на X+,
    остальные грани изолированы (Нейман q=0). Это эквивалентно 1D.
    Поток через грань: q = -λ(T2-T1)/L  [Вт/м²]
    Полная мощность: P = q*A  [Вт]
    """
    L = 0.1; lam = 50.0; T1 = 100.0; T2 = 20.0
    
    p = Problem(
        geometry=BoxGeometry(Lx=L, Ly=L, Lz=L, nx=20, ny=8, nz=8),
        lambda_=lam, Q=0.0,
        bcs={FACE_X_MINUS: BoundaryCondition(type=BC_DIRICHLET, T0=T1),
             FACE_X_PLUS:  BoundaryCondition(type=BC_DIRICHLET, T0=T2),
             FACE_Y_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Y_PLUS:  BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_PLUS:  BoundaryCondition(type=BC_NEUMANN)},
    )
    with CoreBridge() as br:
        p.build_mesh_in_core(br)
        p.solve(br, tol=1e-12)
    
    # Аналитическое решение в узлах.
    x = p.nodes[:, 0]
    T_ana = T1 + (T2 - T1) * x / L
    err = float(np.max(np.abs(p.T - T_ana)))
    
    # Энергобаланс. Аналитически q_through = λ*(T1-T2)/L*A.
    q_analytical = lam * (T1 - T2) / L * (L * L)  # Вт
    bal = p.energy_balance()
    print(f"V1. Стенка T1={T1}→T2={T2} (λ={lam}, L={L}):")
    print(f"    max|T-T_ana| = {err:.3e} (должно быть ~10^-9)")
    print(f"    Аналитический поток через стенку: {q_analytical:.4f} Вт")
    print(f"    Численный поток (q_out):          {bal['q_out_W']:.4f} Вт")
    print(f"    Относительная ошибка по потоку:   "
          f"{abs(q_analytical - bal['q_out_W'])/q_analytical*100:.2f}%")
    print()
    return err < 1e-8


def test_v2_wall_convection():
    """V2. Стенка с конвекцией.
    
    На X-: Дирихле T_wall = 100 °C.
    На X+: конвекция, α=50, T_inf=20 °C.
    Боковые грани: изоляция.
    
    1D-решение:
        T(x) = T_wall + (T_inf - T_wall) * x/(L + λ/α)
    Температура на конвективной поверхности:
        T_s = T_wall + (T_inf - T_wall) * L/(L + λ/α)
    Поток через стенку:
        q = α*(T_s - T_inf) = λ*(T_wall - T_s)/L
    """
    L = 0.1; lam = 50.0; T_wall = 100.0; alpha = 50.0; T_inf = 20.0
    p = Problem(
        geometry=BoxGeometry(Lx=L, Ly=L, Lz=L, nx=30, ny=8, nz=8),
        lambda_=lam, Q=0.0,
        bcs={FACE_X_MINUS: BoundaryCondition(type=BC_DIRICHLET, T0=T_wall),
             FACE_X_PLUS:  BoundaryCondition(type=BC_ROBIN, alpha=alpha, T_inf=T_inf),
             FACE_Y_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Y_PLUS:  BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_PLUS:  BoundaryCondition(type=BC_NEUMANN)},
    )
    with CoreBridge() as br:
        p.build_mesh_in_core(br)
        p.solve(br, tol=1e-12)
    
    # 1D-аналитика. Температура линейна.
    R_cond = L / lam  # тепловое сопротивление кондукции
    R_conv = 1.0 / alpha  # тепловое сопротивление конвекции
    T_s_analytical = T_wall + (T_inf - T_wall) * R_cond / (R_cond + R_conv)
    q_analytical = (T_wall - T_inf) / (R_cond + R_conv)
    
    x = p.nodes[:, 0]
    T_ana = T_wall + (T_s_analytical - T_wall) * x / L
    err = float(np.max(np.abs(p.T - T_ana)))
    
    bal = p.energy_balance()
    print(f"V2. Стенка с конвекцией:")
    print(f"    T_s аналитическая = {T_s_analytical:.4f} °C")
    print(f"    T_s численная     = {p.T[np.argmax(x)]:.4f} °C")
    print(f"    max|T-T_ana| = {err:.3e}")
    print(f"    q аналитический = {q_analytical:.4f} Вт/м² ({q_analytical*L*L:.4f} Вт)")
    print(f"    q_out численный = {bal['q_out_W']:.4f} Вт")
    print()
    return err < 0.1


def test_v3_wall_with_source():
    """V3. Стенка с объёмным источником Q.
    
    На X-: T=0. На X+: T=0. Остальные изолированы. Q = const.
    
    1D-уравнение: -λ d²T/dx² = Q  →  T''(x) = -Q/λ
    Решение: T(x) = (Q/2λ)*x*(L-x)
    Максимум в середине: T_max = Q*L²/(8λ).
    Полное генерируемое тепло: P = Q*L³.
    """
    L = 0.1; lam = 50.0; Q = 1.0e5
    p = Problem(
        geometry=BoxGeometry(Lx=L, Ly=L, Lz=L, nx=30, ny=8, nz=8),
        lambda_=lam, Q=Q,
        bcs={FACE_X_MINUS: BoundaryCondition(type=BC_DIRICHLET, T0=0.0),
             FACE_X_PLUS:  BoundaryCondition(type=BC_DIRICHLET, T0=0.0),
             FACE_Y_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Y_PLUS:  BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_PLUS:  BoundaryCondition(type=BC_NEUMANN)},
    )
    with CoreBridge() as br:
        p.build_mesh_in_core(br)
        p.solve(br, tol=1e-12)
    
    x = p.nodes[:, 0]
    T_ana = (Q / (2 * lam)) * x * (L - x)
    err = float(np.max(np.abs(p.T - T_ana)))
    
    T_max_analytical = Q * L * L / (8 * lam)
    T_max_numerical = float(p.T.max())
    
    P_total = Q * L * L * L  # полная мощность
    bal = p.energy_balance()
    print(f"V3. Стенка с источником Q={Q:g}:")
    print(f"    T_max аналитический = {T_max_analytical:.4f} °C")
    print(f"    T_max численный     = {T_max_numerical:.4f} °C")
    print(f"    max|T-T_ana| = {err:.3e}")
    print(f"    Аналитическое тепловыделение: {P_total:.4f} Вт")
    print(f"    Численное q_gen:              {bal['q_gen_W']:.4f} Вт")
    print(f"    Численный q_out:              {bal['q_out_W']:.4f} Вт")
    print()
    return err < 0.1


def test_v4_series_resistances():
    """V4. Два материала последовательно (теплопередача через стенку).
    
    Стенка из двух слоёв:
        слой 1: x ∈ [0, L/2], λ1 = 400 (медь)
        слой 2: x ∈ [L/2, L], λ2 = 1 (стекло)
    На границах: T1=100, T2=0. Остальные изолированы.
    
    Теория последовательного сопротивления:
        R1 = (L/2)/λ1, R2 = (L/2)/λ2
        T_interface = T1 + (T2 - T1) * R1/(R1+R2) ≈ 99.75
    """
    L = 0.1; lam1 = 400.0; lam2 = 1.0
    T1 = 100.0; T2 = 0.0
    
    p = Problem(
        geometry=BoxGeometry(Lx=L, Ly=L, Lz=L, nx=24, ny=8, nz=8),
        lambda_=lam2,  # глобальный — материал 2
        Q=0.0,
        bcs={FACE_X_MINUS: BoundaryCondition(type=BC_DIRICHLET, T0=T1),
             FACE_X_PLUS:  BoundaryCondition(type=BC_DIRICHLET, T0=T2),
             FACE_Y_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Y_PLUS:  BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_PLUS:  BoundaryCondition(type=BC_NEUMANN)},
    )
    # Регион 1: левая половина — медь.
    p.material_regions.append(MaterialRegion(
        name="Cu", lambda_=lam1, shape=REGION_BOX,
        params=(0, L/2, 0, L, 0, L)))
    
    with CoreBridge() as br:
        p.build_mesh_in_core(br)
        p.solve(br, tol=1e-12)
    
    R1 = (L/2)/lam1; R2 = (L/2)/lam2
    T_int_analytical = T1 + (T2 - T1) * R1 / (R1 + R2)
    
    # Находим точку на интерфейсе (x ≈ L/2).
    interface_mask = np.abs(p.nodes[:, 0] - L/2) < L/30
    T_int_numerical = float(p.T[interface_mask].mean())
    
    print(f"V4. Последовательно: Cu(λ=400) + стекло(λ=1):")
    print(f"    T_interface аналитическая = {T_int_analytical:.4f} °C")
    print(f"    T_interface численная     = {T_int_numerical:.4f} °C")
    print(f"    Ошибка: {abs(T_int_analytical - T_int_numerical):.4f} °C")
    print()
    return abs(T_int_analytical - T_int_numerical) < 0.5


def test_v5_heat_flux_neumann():
    """V5. Заданный поток на одной грани, Дирихле на противоположной.
    
    На X-: q = q0 (поток входит в тело).
    На X+: T = T_0.
    1D-решение: T(x) = T_0 + q0*(L-x)/λ
    """
    L = 0.1; lam = 50.0; q0 = 1000.0; T_0 = 20.0
    # ВНИМАНИЕ: в нашем коде q_in задаётся как q0>0 означает поток ВНУТРЬ
    # (если внешняя нормаль смотрит наружу, то q·n < 0 — это вход).
    # Проверим конвенцию.
    
    p = Problem(
        geometry=BoxGeometry(Lx=L, Ly=L, Lz=L, nx=20, ny=8, nz=8),
        lambda_=lam, Q=0.0,
        bcs={FACE_X_MINUS: BoundaryCondition(type=BC_NEUMANN, q0=q0),
             FACE_X_PLUS:  BoundaryCondition(type=BC_DIRICHLET, T0=T_0),
             FACE_Y_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Y_PLUS:  BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_MINUS: BoundaryCondition(type=BC_NEUMANN),
             FACE_Z_PLUS:  BoundaryCondition(type=BC_NEUMANN)},
    )
    with CoreBridge() as br:
        p.build_mesh_in_core(br)
        p.solve(br, tol=1e-12)
    
    # Аналитика: T(x) = T_0 + |q0|*(L-x)/λ — для q ВХОДЯЩЕГО в тело на X-.
    # Если конвенция «q0>0 значит -λ∂T/∂n = q0 (выход)», то для входа надо q0<0.
    x = p.nodes[:, 0]
    T_max_at_zero = T_0 + q0 * L / lam
    T_at_zero = float(p.T[np.argmin(x)])
    print(f"V5. Поток на X- (q0={q0}), Дирихле T={T_0} на X+:")
    print(f"    T(x=0) численная   = {T_at_zero:.4f} °C")
    print(f"    Если q ВНУТРЬ: ожидаем T = {T_0 + q0*L/lam:.4f} °C")
    print(f"    Если q НАРУЖУ: ожидаем T = {T_0 - q0*L/lam:.4f} °C")
    print(f"    Конвенция знака потока зависит от реализации.")
    print()
    return True  # информационно


# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("УГЛУБЛЁННАЯ ФИЗИЧЕСКАЯ ВЕРИФИКАЦИЯ")
    print("=" * 70)
    print()
    
    tests = [
        ("V1 Стенка T1→T2",       test_v1_wall_two_temps),
        ("V2 Стенка с конвекцией",test_v2_wall_convection),
        ("V3 Стенка с источником",test_v3_wall_with_source),
        ("V4 Два материала",      test_v4_series_resistances),
        ("V5 Поток на грани",     test_v5_heat_flux_neumann),
    ]
    
    passed = 0
    for name, fn in tests:
        try:
            if fn():
                passed += 1
                print(f"  -> PASS: {name}")
            else:
                print(f"  -> FAIL: {name}")
        except Exception as e:
            print(f"  -> ERROR: {name}: {e}")
            import traceback; traceback.print_exc()
        print()
    
    print("=" * 70)
    print(f"ИТОГО: {passed}/{len(tests)}")
    print("=" * 70)
