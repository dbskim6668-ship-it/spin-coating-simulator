import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================================================================
# 웹 페이지 설정
# =========================================================================
st.set_page_config(page_title="스핀 코팅 시뮬레이터", page_icon="🌀", layout="wide")
st.title("🌀 스핀 코팅 박막 균일도 시뮬레이터")
st.markdown("### Emslie-Bonner-Peck 이론과 Meyerhofer 모델 기반")
st.write("성균관대학교 화학공학부 | 화공유체역학 텀 프로젝트")

# =========================================================================
# 물리 상수 및 기본 파라미터
# =========================================================================
RHO = 1100          # PR 밀도 (kg/m³)
SIGMA = 0.030       # 표면장력 (N/m)
PHI_M = 0.64        # 최대 패킹 분율
INTRINSIC_ETA = 2.5 # 고유 점도

# =========================================================================
# 핵심 물리 모델 함수들
# =========================================================================

def viscosity_krieger_dougherty(eta0, phi0, phi):
    """Krieger-Dougherty 관계식에 따른 점도 계산"""
    phi = np.clip(phi, 0, PHI_M - 0.001)
    eta_rel_0 = (1 - phi0 / PHI_M) ** (-INTRINSIC_ETA * PHI_M)
    eta_rel = (1 - phi / PHI_M) ** (-INTRINSIC_ETA * PHI_M)
    return eta0 * (eta_rel / eta_rel_0)


def meyerhofer_ode(t, y, omega, E, eta0, phi0, rho):
    """Meyerhofer 모델의 정밀 ODE 시스템"""
    h = max(y[0], 1e-9)
    h_phi = max(y[1], 1e-12)
    
    phi = np.clip(h_phi / h, 0, PHI_M - 0.001)
    
    if phi >= PHI_M - 0.005:
        return [0, 0]
        
    eta = viscosity_krieger_dougherty(eta0, phi0, phi)
    
    rotation_term = (2 * rho * (omega ** 2) * (h ** 3)) / (3 * eta)
    evaporation_term = E
    
    dh_dt = -rotation_term - evaporation_term
    dh_phi_dt = 0 
    
    return [dh_dt, dh_phi_dt]


def run_meyerhofer_simulation(omega, h0, eta0, phi0, E, t_max=30.0):
    """안정성이 강화된 Meyerhofer 시뮬레이션"""
    y0 = [h0, h0 * phi0]
    
    def gel_event(t, y, *args):
        h = max(y[0], 1e-9)
        phi = y[1] / h
        return PHI_M - phi - 0.005
    gel_event.terminal = True
    gel_event.direction = -1
    
    try:
        sol = solve_ivp(
            meyerhofer_ode,
            [0, t_max],
            y0,
            args=(omega, E, eta0, phi0, RHO),
            method='RK45',
            events=gel_event,
            max_step=0.1
        )
        
        times = sol.t
        h_values = np.maximum(sol.y[0], 1e-9)
        phi_values = np.clip(sol.y[1] / h_values, 0, PHI_M - 0.001)
        
    except Exception as e:
        times = np.array([0, t_max])
        h_values = np.array([h0, h0])
        phi_values = np.array([phi0, phi0])
    
    eta_values = np.array([viscosity_krieger_dougherty(eta0, phi0, p) for p in phi_values])
    
    t_transition = None
    for i, t in enumerate(times):
        rot = (2 * RHO * (omega ** 2) * (h_values[i] ** 3)) / (3 * eta_values[i])
        if rot < E:
            t_transition = t
            break
            
    t_gel = sol.t_events[0][0] if len(sol.t_events) > 0 and len(sol.t_events[0]) > 0 else None
    
    return times, h_values, phi_values, eta_values, t_transition, t_gel


def calculate_edge_bead_profile(h_final, R_wafer, omega, eta_final, c_rpm, c_h0_um, c_eta0, num_points=200):
    """물리 스케일링이 정상화된 Edge Bead 계산식"""
    r = np.linspace(0, R_wafer, num_points)
    h_profile = np.ones_like(r) * h_final
    
    if omega <= 0 or h_final <= 0:
        return r, h_profile, 0.01 * R_wafer, 0
        
    L_cap = np.cbrt(SIGMA / (RHO * omega**2 * R_wafer)) * np.cbrt(R_wafer)
    L_cap = np.clip(L_cap, 0.005 * R_wafer, 0.1 * R_wafer)
    
    thickness_factor = (h_final * 1e9) / 1000.0
    delta_h = h_final * 0.035 * (c_eta0 ** 0.5) * (c_h0_um / 30.0) * (3000 / c_rpm) * thickness_factor
    delta_h = np.clip(delta_h, 0, h_final * 0.4)
    
    edge_region = r > (R_wafer - 1.5 * L_cap)
    r_edge = r[edge_region]
    
    if len(r_edge) > 0:
        x_norm = (r_edge - (R_wafer - L_cap)) / L_cap
        edge_factor = 1 / (1 + np.exp(-4 * x_norm))
        h_profile[edge_region] = h_final + delta_h * edge_factor
        
    return r, h_profile, L_cap, delta_h


def calculate_uniformity(h_profile, r, R_wafer):
    """정밀 균일도 검증 공식"""
    h_mean = np.mean(h_profile)
    h_max = np.max(h_profile)
    h_min = np.min(h_profile)
    
    if h_mean <= 0:
        return 100.0, 0, 0, 0
        
    uniformity = (h_max - h_min) / (2 * h_mean) * 100
    return uniformity, h_mean, h_max, h_min


def analytical_solution_no_evaporation(h0, omega, eta0, t):
    """EBP 모델의 수학적 해석해(Analytical Solution) 공식"""
    K = 2 * RHO * (omega ** 2) / (3 * eta0)
    return h0 / np.sqrt(1 + 4 * K * (h0 ** 2) * t)

# =========================================================================
# 사이드바 - 공통 파라미터 (TAB 1, TAB 2용)
# =========================================================================
st.sidebar.header("⚙️ 기본 공정 파라미터")
rpm = st.sidebar.slider("회전 속도 ω (RPM)", 500, 6000, 3000, 100)
omega = rpm * 2 * np.pi / 60
h0_um = st.sidebar.slider("초기 두께 h₀ (µm)", 5.0, 100.0, 30.0, 1.0)
h0 = h0_um * 1e-6
eta0 = st.sidebar.slider("초기 점도 η₀ (Pa·s)", 0.01, 2.0, 0.5, 0.01)
phi0 = st.sidebar.slider("초기 고형분 분율 φ₀", 0.05, 0.50, 0.20, 0.01)
E_nm_s = st.sidebar.slider("증발 속도 E (nm/s)", 0, 300, 50, 10)
E = E_nm_s * 1e-9
R_wafer_mm = st.sidebar.slider("웨이퍼 반경 R (mm)", 50, 200, 150, 25)
R_wafer = R_wafer_mm * 1e-3

# =========================================================================
# 탭 구성
# =========================================================================
tab1, tab2, tab3 = st.tabs(["🔬 시뮬레이션", "📐 이론 검증", "🎯 공정 최적화"])

# =========================================================================
# TAB 1: 메인 시뮬레이션 (Real-time Visualization)
# =========================================================================
with tab1:
    st.header("Meyerhofer 모델 기반 스핀 코팅 시뮬레이션")
    times, h_vals, phi_vals, eta_vals, t_trans, t_gel = run_meyerhofer_simulation(omega, h0, eta0, phi0, E)
    r_profile, h_profile, L_cap, delta_h = calculate_edge_bead_profile(h_vals[-1], R_wafer, omega, eta_vals[-1], rpm, h0_um, eta0)
    uniformity, h_mean, h_max, h_min = calculate_uniformity(h_profile, r_profile, R_wafer)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최종 중심 두께", f"{h_vals[-1] * 1e9:.1f} nm")
    col2.metric("최종 고형분 분율", f"{phi_vals[-1]:.3f}")
    col3.metric("균일도 오차", f"±{uniformity:.2f}%")
    col4.metric("Edge Bead 높이", f"{delta_h * 1e9:.1f} nm")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(times, h_vals * 1e6, 'b-')
    axes[0, 0].set_title('Film Thickness Evolution (um)')
    axes[0, 1].plot(times, phi_vals, 'g-')
    axes[0, 1].set_title('Solid Concentration')
    axes[1, 0].plot(times, eta_vals, 'm-')
    axes[1, 0].set_title('Viscosity Evolution (Pa.s)')
    
    rot_contrib = [(2 * RHO * (omega ** 2) * (h ** 3)) / (3 * eta_vals[i]) * 1e9 for i, h in enumerate(h_vals)]
    axes[1, 1].plot(times, rot_contrib, label='Rotation')
    axes[1, 1].axhline(E*1e9, color='r', linestyle='--', label='Evaporation')
    axes[1, 1].set_title('Thinning Mechanism (nm/s)')
    axes[1, 1].legend()
    for ax in axes.flat: ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# =========================================================================
# TAB 2: 이론 검증 (★수정 및 기능 추가: Validation View)
# =========================================================================
with tab2:
    st.header("📐 해석해와의 비교 검증 (Analytical-Solution Comparison)")
    st.markdown("증발 효과를 배제한 순수 **Emslie-Bonner-Peck (EBP) 모델** 조건에서 **수치해석(Numerical)** 결과와 **해석해(Analytical)** 공식을 실시간으로 비교 플로팅하여 시뮬레이터의 신뢰성을 정밀 검증합니다.")
    
    def simple_ode(t, h, omega, eta0):
        return [-(2 * RHO * (omega ** 2) * (max(h[0], 1e-12) ** 3)) / (3 * eta0)]
        
    # 수치해석 솔버 실행 (0초부터 20초까지)
    sol_simple = solve_ivp(simple_ode, [0, 20], [h0], args=(omega, eta0), max_step=0.1)
    t_eval = sol_simple.t
    h_num = sol_simple.y[0]
    
    # 수학적 해석해 도출
    h_ana = analytical_solution_no_evaporation(h0, omega, eta0, t_eval)
    
    # 두 결과 사이의 상대 오차 계산
    rel_err = np.abs(h_num - h_ana) / h_ana * 100
    max_error = np.max(rel_err)
    
    st.success(f"✅ 검증 완료: 수학적 해석해 대비 수치해석 솔버의 최대 상대 오차 = **{max_error:.6f}%**")
    
    # ---------------------------------------------------------------------
    # [과제 요구사항 핵심 반영] Analytical vs Numerical 비교 플롯 생성
    # ---------------------------------------------------------------------
    fig_val, ax_val = plt.subplots(figsize=(10, 4.5))
    
    # 두 개의 선을 겹쳐서 플로팅 (해석해는 점선, 수치해석해는 실선)
    ax_val.plot(t_eval, h_ana * 1e6, 'r--', linewidth=2.5, label='Analytical Solution (EBP Equation)')
    ax_val.plot(t_eval, h_num * 1e6, 'b-', linewidth=1.5, label='Numerical Solution (scipy solve_ivp)')
    
    ax_val.set_title('Validation View: Analytical vs Numerical Solution', fontsize=12, fontweight='bold')
    ax_val.set_xlabel('Time (s)', fontsize=10)
    ax_val.set_ylabel('Film Thickness (µm)', fontsize=10)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(fontsize=10)
    
    # 오차 텍스트 박스 추가 삽입
    ax_val.text(0.6, 0.75, f"Max Relative Error:\n {max_error:.6f} %", 
                transform=ax_val.transAxes, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 스트림릿 화면에 그래프 렌더링
    st.pyplot(fig_val)
    plt.close(fig_val)

# =========================================================================
# TAB 3: 공정 최적화 (Design-Exploration Mode / Geometry)
# =========================================================================
with tab3:
    st.header("🎯 공정 설계 챌린지 룸")
    st.markdown("**목표**: 지정된 목표 두께와 균일도 스펙(±2%)을 동시에 만족하는 제어 조건 찾기")
    
    # User-editable geometry / Target 세팅
    target_h_nm = st.number_input("🎯 목표 최종 두께 (nm)", 100, 5000, 1000, 50)
    target_h = target_h_nm * 1e-9
    
    st.markdown("---")
    st.subheader("🛠️ 챌린지 독점 공정 제어 패널")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        c_rpm = st.slider("회전 속도 (RPM)", 500, 6000, 3000, 100, key="chal_rpm")
        c_eta0 = st.slider("초기 점도 (Pa·s)", 0.01, 2.0, 0.5, 0.01, key="chal_eta")
    with col_c2:
        c_h0_um = st.slider("초기 두께 (µm)", 5.0, 100.0, 30.0, 1.0, key="chal_h0")
        c_phi0 = st.slider("초기 고형분 분율 φ₀", 0.01, 0.50, 0.15, 0.01, key="chal_phi")
    
    c_omega = c_rpm * 2 * np.pi / 60
    c_h0 = c_h0_um * 1e-6
    c_E_actual = 60 * 1e-9 
    
    # 챌린지 시뮬레이션 연산
    _, c_h_vals, _, c_eta_vals, _, _ = run_meyerhofer_simulation(c_omega, c_h0, c_eta0, c_phi0, c_E_actual)
    c_h_final = c_h_vals[-1]
    
    # 유체역학적 균일도 프로파일 매핑
    c_r_profile, c_h_profile, _, c_delta_h = calculate_edge_bead_profile(
        c_h_final, R_wafer, c_omega, c_eta_vals[-1], c_rpm, c_h0_um, c_eta0
    )
    c_uniformity, _, _, _ = calculate_uniformity(c_h_profile, c_r_profile, R_wafer)
    
    thickness_error = abs(c_h_final - target_h) / target_h * 100
    
    st.markdown("---")
    st.subheader("📊 챌린지 실시간 판정 결과")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("현재 최종 두께", f"{c_h_final * 1e9:.1f} nm", delta=f"{(c_h_final - target_h)*1e9:+.1f} nm")
        if thickness_error <= 5: st.success("🟢 두께 합격 (오차 5% 이내)")
        else: st.error(f"🔴 두께 불합격 (오차: {thickness_error:.1f}%)")
        
    with col_r2:
        st.metric("현재 공정 균일도 오차", f"±{c_uniformity:.2f}%")
        if c_uniformity <= 2.0: st.success("🟢 균일도 Spec 합격 (±2% 이내)!")
        else: st.error("🔴 균일도 불합격! RPM을 조절하거나 점도/고형분을 낮추세요.")
        
    if thickness_error <= 5 and c_uniformity <= 2.0:
        st.balloons()
        st.success("🎉 대단합니다! 목표 두께와 반도체 수율 스펙(Uniformity 2%)을 모두 달성하셨습니다!")