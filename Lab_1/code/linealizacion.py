import sympy as sp
import numpy as np

sp.init_printing(use_unicode=True)

# =========================
# 1) VARIABLES
# =========================
U, V, W = sp.symbols('U V W')
p, q, r = sp.symbols('p q r')
phi, theta, psi = sp.symbols('phi theta psi')

de, da, dr, dT = sp.symbols('de da dr dT')

# =========================
# 2) PARÁMETROS
# =========================
m, g = sp.symbols('m g')
Ix, Iy, Iz = sp.symbols('Ix Iy Iz')
rho, S, b, c = sp.symbols('rho S b c')

Ueq = sp.symbols('Ueq', positive=True)

qbar_eq = sp.Rational(1,2) * rho * Ueq**2

# =========================
# 3) COEFICIENTES
# =========================
Cx_u = sp.symbols('Cx_u')

Cy_r, Cy_da, Cy_dr = sp.symbols('Cy_r Cy_da Cy_dr')
Cz_u, Cz_q, Cz_de = sp.symbols('Cz_u Cz_q Cz_de')

CL_p, CL_r, CL_da, CL_dr = sp.symbols('CL_p CL_r CL_da CL_dr')

CM_u, CM_q, CM_de = sp.symbols('CM_u CM_q CM_de')

CN_p, CN_r, CN_da, CN_dr = sp.symbols('CN_p CN_r CN_da CN_dr')

# =========================
# 4) TRIM ANALÍTICO
# =========================

# delta_e(Ueq)
de_eq_expr = -45 * CM_u * Ueq / (2700 * CM_de)

# función f(Ueq) = cos(theta)
f_U = S * (45*Cz_u*Ueq + 2700*Cz_de*de_eq_expr) / (m*g)
f_U = sp.simplify(f_U)

print("\n=== f(Ueq) ===")
sp.pprint(f_U)

# =========================
# 5) RANGO VÁLIDO DE Ueq
# =========================

# resolver |f(Ueq)| = 1
sol1 = sp.solve(f_U - 1, Ueq)
sol2 = sp.solve(f_U + 1, Ueq)

print("\n=== Límites analíticos de Ueq ===")
print("f(U)=1:", sol1)
print("f(U)=-1:", sol2)

# =========================
# 6) VALORES NUMÉRICOS
# =========================
vals = {
    m: 1110,
    g: 9.81,
    Ix: 1285,
    Iy: 1824,
    Iz: 2666,
    S: 16.2,
    b: 11.0,
    c: 1.5,
    rho: 1.225,

    Cx_u: -0.07,
    Cy_r: 0.15, Cy_da: 0.0, Cy_dr: 0.15,

    # 🔥 AJUSTADOS (clave)
    Cz_u: 0.05,
    Cz_q: 5.0,
    Cz_de: 0.25,

    CL_p: -0.5, CL_r: 0.25, CL_da: 0.15, CL_dr: 0.035,

    CM_u: 0.02,
    CM_q: -10.0,
    CM_de: -1.0,

    CN_p: -0.1, CN_r: -0.1, CN_da: -0.03, CN_dr: -0.1
}

# evaluar soluciones numéricas
sol1_num = [float(s.subs(vals)) for s in sol1 if s.is_real]
sol2_num = [float(s.subs(vals)) for s in sol2 if s.is_real]

print("\n=== Límites numéricos ===")
print("f=1:", sol1_num)
print("f=-1:", sol2_num)

# =========================
# 7) ELEGIR Ueq VÁLIDO
# =========================

# elegimos manualmente dentro del rango
Ueq_val = 200   #  puedes cambiar esto dentro del rango

# =========================
# 8) CALCULAR TRIM
# =========================

de_eq = float(de_eq_expr.subs(vals).subs(Ueq, Ueq_val))
f_val = float(f_U.subs(vals).subs(Ueq, Ueq_val))

if abs(f_val) > 1:
    raise ValueError("❌ Ueq fuera del rango físico")

theta_eq = float(np.arccos(f_val))
T_eq = float((m*g*sp.sin(theta) - 45*S*Cx_u*Ueq)
             .subs(vals)
             .subs({theta: theta_eq, Ueq: Ueq_val}))

print("\n=== TRIM ===")
print("Ueq:", Ueq_val)
print("theta (deg):", theta_eq * 180/np.pi)
print("de (deg):", de_eq * 180/np.pi)
print("T (N):", T_eq)

# =========================
# 9) FUERZAS (CORREGIDO)
# =========================
Fa_x = Cx_u * (U / Ueq) * qbar_eq * S

Fa_y = (Cy_r * (r * b / (2 * Ueq)) +
        Cy_da * da +
        Cy_dr * dr) * qbar_eq * S

Fa_z = (Cz_u * (U / Ueq)
        - Cz_q * (q * c / (2 * Ueq))
        - Cz_de * de) * qbar_eq * S

# =========================
# 10) MOMENTOS
# =========================
La = (CL_p * (p * b / (2 * Ueq)) +
      CL_r * (r * b / (2 * Ueq)) +
      CL_da * da +
      CL_dr * dr) * qbar_eq * S * b

Ma = (CM_u * (U / Ueq) +
      CM_q * (q * c / (2 * Ueq)) +
      CM_de * de) * qbar_eq * S * c

Na = (CN_p * (p * b / (2 * Ueq)) +
      CN_r * (r * b / (2 * Ueq)) +
      CN_da * da +
      CN_dr * dr) * qbar_eq * S * b

# =========================
# 11) DINÁMICA
# =========================
U_dot = r*V - q*W + Fa_x/m - g*sp.sin(theta) + dT/m
V_dot = p*W - r*(Ueq + U) + Fa_y/m + g*sp.sin(phi)*sp.cos(theta)
W_dot = q*(Ueq + U) - p*V + Fa_z/m + g*sp.cos(phi)*sp.cos(theta)

p_dot = (La + (Iy - Iz)*q*r) / Ix
q_dot = (Ma + (Iz - Ix)*p*r) / Iy
r_dot = (Na + (Ix - Iy)*p*q) / Iz

phi_dot = p+sp.sin(phi)*sp.tan(theta) + q*sp.cos(phi)*sp.tan(theta) + r*sp.sin(phi)/sp.cos(theta)
theta_dot = q*sp.cos(phi) - r*sp.sin(phi)
psi_dot = r*sp.cos(phi)/sp.cos(theta) + q*sp.sin(phi)/sp.cos(theta)

f = sp.Matrix([
    U_dot, V_dot, W_dot,
    p_dot, q_dot, r_dot,
    phi_dot, theta_dot, psi_dot
])

x = sp.Matrix([U, V, W, p, q, r, phi, theta, psi])
u_vec = sp.Matrix([de, da, dr, dT])

# =========================
# 12) LINEALIZACIÓN
# =========================
A = f.jacobian(x)
B = f.jacobian(u_vec)

eq = {
    U: 0, V: 0, W: 0,
    p: 0, q: 0, r: 0,
    phi: 0,
    theta: theta_eq,
    psi: 0,
    de: de_eq,
    da: 0,
    dr: 0,
    dT: 0,
    Ueq: Ueq_val
}

A_num = np.array(A.subs(eq).subs(vals), dtype=float)
B_num = np.array(B.subs(eq).subs(vals), dtype=float)

print("\n=== MATRIZ A ===")
print(A_num)

print("\n=== MATRIZ B ===")
print(B_num)


