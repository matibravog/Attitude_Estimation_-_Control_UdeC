#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laboratorio 3 - Estimación de actitud, caso aeronáutico
figuras exportadas en SVG.

Vector de estado:
    x = [phi, theta, psi, u, v, w]^T

Vector de entrada:
    u_k = [p, q, r, ax, ay, az]^T   (dinámica)

Vector de salida/medición:
    y_k = [Va, v_N, v_E, v_D]^T
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# CONFIGURACIÓN
# =========================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "aircraft_data.xlsx"

SHEET_NAME = 0
START_INDEX = 0
MAX_POINTS = 10000
STRIDE = 1

G = 9.80665
G_I = np.array([0.0, 0.0, -G])

SAVE_FIGURES = True
SHOW_FIGURES = False
FIG_FORMAT = "svg"

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def wrap_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def angle_error(est, ref):
    return wrap_pi(est - ref)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def rot_matrix_bi(phi, theta, psi):
    R_phi = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi), np.sin(phi)],
        [0.0, -np.sin(phi), np.cos(phi)]
    ])
    R_theta = np.array([
        [np.cos(theta), 0.0, -np.sin(theta)],
        [0.0, 1.0, 0.0],
        [np.sin(theta), 0.0, np.cos(theta)]
    ])
    R_psi = np.array([
        [np.cos(psi), np.sin(psi), 0.0],
        [-np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0]
    ])
    return R_phi @ (R_theta @ R_psi)


def euler_from_rot_matrix_bi(R):
    theta = np.arcsin(np.clip(-R[0, 2], -1.0, 1.0))
    phi = np.arctan2(R[1, 2], R[2, 2])
    psi = np.arctan2(R[0, 1], R[0, 0])
    return np.array([phi, theta, psi])


def euler_rates(phi, theta, p, q, r):
    c_theta = np.cos(theta)
    if abs(c_theta) < 1e-6:
        c_theta = np.sign(c_theta) * 1e-6 if c_theta != 0 else 1e-6

    s_phi = np.sin(phi)
    c_phi = np.cos(phi)

    phi_dot = p + (s_phi * q + c_phi * r) * np.tan(theta)
    theta_dot = c_phi * q - s_phi * r
    psi_dot = (s_phi * q + c_phi * r) / c_theta

    return np.array([phi_dot, theta_dot, psi_dot])

# =========================================================
# TRIAD
# =========================================================
def triad_rotation(b1, b2, r1, r2):
    tb1 = normalize(b1)
    tb2 = normalize(np.cross(tb1, b2))
    tb3 = np.cross(tb1, tb2)

    tr1 = normalize(r1)
    tr2 = normalize(np.cross(tr1, r2))
    tr3 = np.cross(tr1, tr2)

    T_b = np.column_stack((tb1, tb2, tb3))
    T_i = np.column_stack((tr1, tr2, tr3))
    return T_b @ T_i.T


def estimate_triad(vgps_i, Va, acc_b):
    b1 = np.array([max(float(Va), 1e-6), 0.0, 0.0])
    b2 = np.asarray(acc_b, dtype=float)
    r1 = np.asarray(vgps_i, dtype=float)
    r2 = G_I

    if np.linalg.norm(r1) < 1e-9 or np.linalg.norm(b2) < 1e-9:
        return np.eye(3)

    return triad_rotation(b1, b2, r1, r2)

# =========================================================
# MODELO EKF
# =========================================================
def process_model_continuous(x, u_in):
    phi, theta, psi, u, v, w = x
    p, q, r, ax, ay, az = u_in

    omega_b = np.array([p, q, r])
    acc_b = np.array([ax, ay, az])
    vel_b = np.array([u, v, w])

    R_bi = rot_matrix_bi(phi, theta, psi)

    att_dot = euler_rates(phi, theta, p, q, r)
    acc_kin_b = acc_b - R_bi @ G_I
    vel_dot_b = acc_kin_b - np.cross(omega_b, vel_b)

    return np.hstack((att_dot, vel_dot_b))


def discrete_process_model(x, u_in, dt):
    x_next = x + dt * process_model_continuous(x, u_in)
    x_next[0:3] = wrap_pi(x_next[0:3])
    return x_next


def measurement_model(x):
    """
    y = [Va, v_N, v_E, v_D]^T
    """
    phi, theta, psi, u, v, w = x
    R_bi = rot_matrix_bi(phi, theta, psi)

    vel_b = np.array([u, v, w])

    Va_pred = np.array([u])
    vel_i = R_bi.T @ vel_b

    return np.hstack((Va_pred, vel_i))


def numerical_jacobian(fun, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(fun(x), dtype=float)

    J = np.zeros((y0.size, x.size))

    for j in range(x.size):
        dx = np.zeros_like(x)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step

        yp = fun(x + dx)
        ym = fun(x - dx)

        J[:, j] = (yp - ym) / (2.0 * step)

    return J


def ekf_step(x_plus, P_plus, u_in, y_meas, dt, Qd, R_meas):
    n = x_plus.size
    I = np.eye(n)

    f_disc = lambda xx: discrete_process_model(xx, u_in, dt)

    F = numerical_jacobian(f_disc, x_plus)
    x_minus = f_disc(x_plus)
    P_minus = F @ P_plus @ F.T + Qd

    h_fun = lambda xx: measurement_model(xx)
    H = numerical_jacobian(h_fun, x_minus)

    y_pred = h_fun(x_minus)
    innovation = y_meas - y_pred

    S = H @ P_minus @ H.T + R_meas
    K = P_minus @ H.T @ np.linalg.inv(S)

    x_plus = x_minus + K @ innovation
    x_plus[0:3] = wrap_pi(x_plus[0:3])

    P_plus = (I - K @ H) @ P_minus @ (I - K @ H).T + K @ R_meas @ K.T

    return x_plus, P_plus, K, innovation

# =========================================================
# CARGA DE DATOS
# =========================================================
def load_aircraft_data(filename):
    df = pd.read_excel(filename, sheet_name=SHEET_NAME, engine="openpyxl")

    i0 = START_INDEX
    i1 = None if MAX_POINTS is None else START_INDEX + MAX_POINTS * STRIDE
    df = df.iloc[i0:i1:STRIDE].copy().reset_index(drop=True)

    t = df["Tiempo_s"].to_numpy()
    t = t - t[0]

    pqr_meas = np.deg2rad(df[["p_deg_s", "q_deg_s", "r_deg_s"]].to_numpy())
    acc_meas = df[["ax_m_s2", "ay_m_s2", "az_m_s2"]].to_numpy()
    vgps_meas = df[["vxgps_m_s", "vygps_m_s", "vzgps_m_s"]].to_numpy()
    Va_meas = df["Va_m_s"].to_numpy()

    euler_true = np.deg2rad(df[["phi_deg_true", "theta_deg_true", "psi_deg_true"]].to_numpy())
    vb_true = df[["u_m_s_true", "v_m_s_true", "w_m_s_true"]].to_numpy()

    return t, pqr_meas, acc_meas, vgps_meas, Va_meas, euler_true, vb_true

# =========================================================
# MAIN
# =========================================================
def main():
    t, pqr_meas, acc_meas, vgps_meas, Va_meas, euler_true, vb_true = load_aircraft_data(DATA_FILE)

    N = len(t)

    print("Ejecutando TRIAD...")
    euler_triad = np.zeros((N, 3))

    for k in range(N):
        R_triad = estimate_triad(vgps_meas[k], Va_meas[k], acc_meas[k])
        euler_triad[k] = euler_from_rot_matrix_bi(R_triad)

    euler_triad = np.unwrap(euler_triad, axis=0)

    print("Ejecutando EKF...")

    euler_ekf = np.zeros((N, 3))
    vb_ekf = np.zeros((N, 3))
    innovations = np.zeros((N, 4))

    R0 = rot_matrix_bi(*euler_triad[0])
    vb0 = R0 @ vgps_meas[0]

    x_plus = np.hstack((euler_triad[0], vb0))

    P_plus = np.diag([
        np.deg2rad(10.0)**2,
        np.deg2rad(10.0)**2,
        np.deg2rad(15.0)**2,
        4.0**2,
        4.0**2,
        4.0**2,
    ])

    Qc = np.diag([
        (np.deg2rad(0.001))**2,
        (np.deg2rad(0.001))**2,
        (np.deg2rad(0.002))**2,
        (0.5)**2,
        (0.5)**2,
        (0.5)**2,
    ])

    R_meas = np.diag([
        (0.5)**2,
        (0.2)**2,
        (0.2)**2,
        (0.3)**2
    ])

    for k in range(N):

        dt = max(t[1] - t[0], 1e-3) if k == 0 else max(t[k] - t[k-1], 1e-3)

        u_in = np.hstack((pqr_meas[k], acc_meas[k]))

        y_meas = np.hstack((Va_meas[k], vgps_meas[k]))

        Qd = Qc * dt

        x_plus, P_plus, K, innov = ekf_step(x_plus, P_plus, u_in, y_meas, dt, Qd, R_meas)

        euler_ekf[k] = x_plus[:3]
        vb_ekf[k] = x_plus[3:]
        innovations[k] = innov

    euler_ekf = np.unwrap(euler_ekf, axis=0)
    euler_true = np.unwrap(euler_true, axis=0)

    print("\nVector de salida final:")
    print("y = [Va, v_N, v_E, v_D]^T")

    print("\nListo.")

    # =====================================================
    # GRAFICOS
    # =====================================================

    labels = ["Roll", "Pitch", "Yaw"]

    # ---------------- ATTITUDE ----------------
    fig1, ax1 = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    for i in range(3):
        ax1[i].plot(t, np.rad2deg(euler_true[:, i]), label="Real")
        ax1[i].plot(t, np.rad2deg(euler_triad[:, i]), label="TRIAD")
        ax1[i].plot(t, np.rad2deg(euler_ekf[:, i]), label="EKF")
        ax1[i].set_ylabel(labels[i] + " [deg]")
        ax1[i].grid()

    ax1[0].legend()
    ax1[0].set_title("Actitud: Real vs TRIAD vs EKF")
    ax1[-1].set_xlabel("Tiempo [s]")

    # ---------------- VELOCITY ----------------
    fig2, ax2 = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    v_labels = ["u", "v", "w"]

    for i in range(3):
        ax2[i].plot(t, vb_true[:, i], label="Real")
        ax2[i].plot(t, vb_ekf[:, i], label="EKF")
        ax2[i].set_ylabel(v_labels[i] + " [m/s]")
        ax2[i].grid()

    ax2[0].legend()
    ax2[0].set_title("Velocidades body: Real vs EKF")
    ax2[-1].set_xlabel("Tiempo [s]")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()