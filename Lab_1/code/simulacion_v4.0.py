import numpy as np
import sympy as sp
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from itertools import product, combinations

# =========================
# PARÁMETROS
# =========================
m = 507 + 200
g = 9.81

Ix, Iy, Iz = 2.435474e3, 3.742951e3, 1.504225e3

S = 16.2
b = 11.0
c = 1.5

# Coeficientes
Cx_u = -0.1
Cy_r = 0.2

Cz_u = 0.6
Cz_q = 7.0
Cz_de = 0.4

CL_p = -0.4
CL_r = 0.4
CL_da = 0.2
CL_dr = -0.05

CM_u = 0.01
CM_q = -5.0
CM_de = -0.5

CN_p = -0.05
CN_r = -0.05
CN_da = -0.01
CN_dr = -0.05

# Ángulo de trayectoria
theta_trim = math.radians(10)
phi_trim = 0

# =========================
# TRIM
# =========================
U, T, de = sp.symbols('U T de')

eq1 = T + 45*S*Cx_u*U - m*g*math.sin(theta_trim)
eq2 = m*g*math.cos(theta_trim) - (45*S*Cz_u*U + 2700*S*Cz_de*de)
eq3 = 45*CM_u*U + 2700*CM_de*de

sol = sp.solve((eq1, eq2, eq3), (T, U, de))

T0 = float(sol[T])
U0 = float(sol[U])
de0 = float(sol[de])

print("=== TRIM ===")
print(f"U0 = {U0:.3f}")
print(f"T0 = {T0:.3f}")
print(f"de0 = {de0:.4f} rad ({math.degrees(de0):.2f} deg)")

# =========================
# INPUTS (CONTROL)
# =========================
u_input = np.zeros(4)  # [dT, dde, dda, ddr]

# =========================
# MODELO NO LINEAL
# =========================
def f_nl(t, x):

    u, v, w, p, q, r, phi, theta, psi = x

    dT, dde, dda, ddr = u_input

    T = T0 + dT
    de = de0 + dde
    da = dda
    dr = ddr

    Fx = T + 45*S*Cx_u*u - m*g*np.sin(theta_trim)
    Fy = S*(22.5*b*Cy_r*r + 22.5*b*CL_dr*dr)
    Fz = -S*(45*Cz_u*u + 22.5*c*Cz_q*q + 2700*Cz_de*de) \
          + m*g*np.cos(theta_trim)*np.cos(phi)

    L = S*b*(22.5*b*CL_p*p + 22.5*b*CL_r*r + 2700*CL_da*da + 2700*CL_dr*dr)
    M = S*c*(45*CM_u*u + 22.5*c*CM_q*q + 2700*CM_de*de)
    N = S*b*(22.5*b*CN_p*p + 22.5*b*CN_r*r + 2700*CN_da*da + 2700*CN_dr*dr)

    du = r*v - q*w + Fx/m
    dv = p*w - r*u + Fy/m
    dw = q*u - p*v + Fz/m

    dp = (L + (Iy - Iz)*q*r) / Ix
    dq = (M + (Iz - Ix)*p*r) / Iy
    dr = (N + (Ix - Iy)*p*q) / Iz

    dphi = p + np.tan(theta)* (q*np.sin(phi) + r*np.cos(phi))
    dtheta = q*np.cos(phi) - r*np.sin(phi)
    dpsi = (q*np.sin(phi) + r*np.cos(phi)) / np.cos(theta)

    return [du, dv, dw, dp, dq, dr, dphi, dtheta, dpsi]

# =========================
# CONDICIÓN INICIAL
# =========================
x0 = np.array([U0, 0, 0, 0, 0, 0, 0, theta_trim, 0])
u0 = np.zeros(4)

# =========================
# LINEALIZACIÓN A y B
# =========================
def linearize_AB(f, x0):
    n = len(x0)
    m = len(u0)

    A = np.zeros((n, n))
    B = np.zeros((n, m))

    eps = 1e-5

    # -------- A --------
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps

        f1 = np.array(f(0, x0 + dx))
        f2 = np.array(f(0, x0 - dx))

        A[:, i] = (f1 - f2) / (2 * eps)

    # -------- B --------
    for i in range(m):
        du = np.zeros(m)
        du[i] = eps

        u_input[:] = du
        f1 = np.array(f(0, x0))

        u_input[:] = -du
        f2 = np.array(f(0, x0))

        B[:, i] = (f1 - f2) / (2 * eps)

    u_input[:] = u0

    return A, B

A, B = linearize_AB(f_nl, x0)

np.set_printoptions(precision=4, suppress=True)

print("\n=== MATRIZ A ===")
print(A)

print("\n=== MATRIZ B ===")
print(B)

# =========================
# SIMULACIÓN NO LINEAL
# =========================
t_span = (0, 5)
t_eval = np.linspace(0, 5, 5*100)

sol_nl = solve_ivp(f_nl, t_span, x0, t_eval=t_eval)

# =========================
# MODELO LINEAL
# =========================
def f_lin(t, x):
    u = np.zeros(4)
    return A @ (x - x0) + B @ u

sol_lin = solve_ivp(f_lin, t_span, x0, t_eval=t_eval)

# =========================
# PLOTS
# =========================
states = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi']
units  = ['m/s', 'm/s', 'm/s', 'rad/s', 'rad/s', 'rad/s', 'rad', 'rad', 'rad']

plt.figure(figsize=(12, 10))

for i in range(9):
    plt.subplot(3, 3, i+1)

    plt.plot(sol_nl.t, sol_nl.y[i], label='No lineal')
    plt.plot(sol_lin.t, sol_lin.y[i], '--', label='Lineal')

    plt.title(states[i])
    plt.ylabel(units[i])  
    plt.xlabel('Tiempo [s]')

    plt.grid()
    plt.xlim(0, 5)
    plt.ylim(-20, 50)

    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

## =========================
# ANIMACIÓN 3D ACTITUD
# =========================

phi = sol_nl.y[6]
theta = sol_nl.y[7]
psi = sol_nl.y[8]
t = sol_nl.t

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Ejes del avión (body frame)
line_x, = ax.plot([], [], [], 'r', lw=2)  # eje longitudinal
line_y, = ax.plot([], [], [], 'g', lw=2)  # eje lateral
line_z, = ax.plot([], [], [], 'b', lw=2)  # eje vertical

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ax.set_title("Actitud 3D del vehículo")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# =========================
# ROTATION MATRIX
# =========================
def rotation_matrix(phi, theta, psi):

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    return R_z @ R_y @ R_x


# =========================
# UPDATE
# =========================
def update(i):

    R = rotation_matrix(phi[i], theta[i], psi[i])

    origin = np.array([0, 0, 0])

    x_axis = R @ np.array([1, 0, 0])
    y_axis = R @ np.array([0, 1, 0])
    z_axis = R @ np.array([0, 0, 1])

    # líneas de ejes del avión
    line_x.set_data([0, x_axis[0]], [0, x_axis[1]])
    line_x.set_3d_properties([0, x_axis[2]])

    line_y.set_data([0, y_axis[0]], [0, y_axis[1]])
    line_y.set_3d_properties([0, y_axis[2]])

    line_z.set_data([0, z_axis[0]], [0, z_axis[1]])
    line_z.set_3d_properties([0, z_axis[2]])

    return line_x, line_y, line_z


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(t),
    interval=30,
    blit=False
)

plt.show()