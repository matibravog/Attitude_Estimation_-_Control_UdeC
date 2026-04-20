import numpy as np
import control as ct

# =========================
# MATRICES ORIGINALES
# =========================

A = np.array([
    [ -0.1031,   0,        0,        0,      0,      0,      0,      0,      0 ],
    [  0,        0,       10,        0,      0, -14.276,     0,      0,      0 ],
    [ -0.6187, -10,        0,        0,  9.9969,   0,        0,      0,      0 ],
    [  0,        0,        0,   -7.2437,  0,    7.2437,     0,      0,      0 ],
    [  0.0029,   0,        0,        0, -1.0956,    0,    0,      0,      0 ],
    [  0,        0,        0,   -1.466,     0,  -1.466,    0,      0,      0 ],
    [  0,        0,        0,        1,     0,    0.1763,   0,      0,      0 ],
    [  0,        0,        0,        0,     1,    0,        0,      0,      0 ],
    [  0,        0,        0,        0,     0,    1.0154,   0,      0,      0 ]
])

B = np.array([
    [ 0.0014,   0,        0,        0 ],
    [ 0,        0,        0,     -0.2836 ],
    [ 0,      -24.7468,   0,        0 ],
    [ 0,        0,       39.511,   -9.8777 ],
    [ 0,       -8.7645,   0,        0 ],
    [ 0,        0,      -3.1986,  -15.993 ],
    [ 0,        0,        0,        0 ],
    [ 0,        0,        0,        0 ],
    [ 0,        0,        0,        0 ]
])

# =========================
# FILTRO NUMÉRICO
# =========================

def clean_matrix(M, eps=1e-4):
    M = np.array(M)
    M[np.abs(M) < eps] = 0.0
    return M

A = clean_matrix(A, eps=1e-4)
B = clean_matrix(B, eps=1e-4)

# =========================
# SALIDAS
# =========================

C_phi   = np.array([[0,0,0,0,0,0,1,0,0]])
C_theta = np.array([[0,0,0,0,0,0,0,1,0]])
C_psi   = np.array([[0,0,0,0,0,0,0,0,1]])

# =========================
# FUNCION TF
# =========================

def tf_from_input(A, B, C, input_index):
    B_i = B[:, input_index].reshape(-1,1)
    sys = ct.ss(A, B_i, C, 0)
    return ct.ss2tf(sys)

# =========================
# ROLL (φ)
# =========================
G_phi_dT = tf_from_input(A, B, C_phi, 0)
G_phi_de = tf_from_input(A, B, C_phi, 1)
G_phi_da = tf_from_input(A, B, C_phi, 2)
G_phi_dr = tf_from_input(A, B, C_phi, 3)

# =========================
# PITCH (θ)
# =========================
G_theta_dT = tf_from_input(A, B, C_theta, 0)
G_theta_de = tf_from_input(A, B, C_theta, 1)
G_theta_da = tf_from_input(A, B, C_theta, 2)
G_theta_dr = tf_from_input(A, B, C_theta, 3)

# =========================
# YAW (ψ)
# =========================
G_psi_dT = tf_from_input(A, B, C_psi, 0)
G_psi_de = tf_from_input(A, B, C_psi, 1)
G_psi_da = tf_from_input(A, B, C_psi, 2)
G_psi_dr = tf_from_input(A, B, C_psi, 3)

# =========================
# PRINT (12 TFs)
# =========================

print("\n========== ROLL (φ) ==========")
print("\nROLL / THRUST")
print(G_phi_dT)

print("\nROLL / ELEVADOR")
print(G_phi_de)

print("\nROLL / ALERON")
print(G_phi_da)

print("\nROLL / RUDDER")
print(G_phi_dr)

print("\n========== PITCH (θ) ==========")
print("\nPITCH / THRUST")
print(G_theta_dT)

print("\nPITCH / ELEVADOR")
print(G_theta_de)

print("\nPITCH / ALERON")
print(G_theta_da)

print("\nPITCH / RUDDER")
print(G_theta_dr)

print("\n========== YAW (ψ) ==========")
print("\nYAW / THRUST")
print(G_psi_dT)

print("\nYAW / ELEVADOR")
print(G_psi_de)

print("\nYAW / ALERON")
print(G_psi_da)

print("\nYAW / RUDDER")
print(G_psi_dr)