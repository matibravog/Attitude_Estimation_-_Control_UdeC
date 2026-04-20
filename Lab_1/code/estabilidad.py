import numpy as np
import control as ct

# =========================================================
# MATRICES
# =========================================================

A = np.array([
    [ -0.1031,   0,        0,        0,      0,      0,      0,      0,      0 ],
    [  0,        0,       10,        0,      0, -14.276,     0,      0,      0 ],
    [ -0.6187, -10,        0,        0,  9.9969,   0,        0,      0,      0 ],
    [  0,        0,        0,   -7.2437,  0,    7.2437,     0,      0,      0 ],
    [  0.0029,   0,        0,        0, -1.0956, 0,    0,      0,      0 ],
    [  0,        0,        0,   -1.466,  0, -1.466,    0,      0,      0 ],
    [  0,        0,        0,        1,     0,    0.1763,   0,      0,      0 ],
    [  0,        0,        0,        0,     1,    0,        0,      0,      0 ],
    [  0,        0,        0,        0,     0,    1.0154,   0,      0,      0 ]
])

B = np.array([
    [ 0.0014,   0,        0,        0 ],
    [ 0,        0,        0,   -0.2836 ],
    [ 0,      -24.7468,   0,        0 ],
    [ 0,        0,       39.511,   -9.8777 ],
    [ 0,       -8.7645,   0,        0 ],
    [ 0,        0,      -3.1986,  -15.993 ],
    [ 0,        0,        0,        0 ],
    [ 0,        0,        0,        0 ],
    [ 0,        0,        0,        0 ]
])

# =========================================================
# SALIDAS
# =========================================================

C_phi   = np.array([[0,0,0,0,0,0,1,0,0]])
C_theta = np.array([[0,0,0,0,0,0,0,1,0]])
C_psi   = np.array([[0,0,0,0,0,0,0,0,1]])

# =========================================================
# LIMPIEZA
# =========================================================

def clean(M, eps=1e-4):
    M = np.array(M)
    M[np.abs(M) < eps] = 0.0
    return M

A = clean(A)
B = clean(B)

# =========================================================
# TF
# =========================================================

def tf_from(A, B, C, input_idx):
    B_i = B[:, input_idx].reshape(-1,1)
    sys = ct.ss(A, B_i, C, 0)
    return ct.ss2tf(sys)

# =========================================================
# TODAS LAS 12 TF
# =========================================================

TFs = {

    # ROLL (φ)
    "ROLL / THRUST":   tf_from(A, B, C_phi, 0),
    "ROLL / ELEVADOR": tf_from(A, B, C_phi, 1),
    "ROLL / ALERON":   tf_from(A, B, C_phi, 2),
    "ROLL / RUDDER":   tf_from(A, B, C_phi, 3),

    # PITCH (θ)
    "PITCH / THRUST":   tf_from(A, B, C_theta, 0),
    "PITCH / ELEVADOR": tf_from(A, B, C_theta, 1),
    "PITCH / ALERON":   tf_from(A, B, C_theta, 2),
    "PITCH / RUDDER":   tf_from(A, B, C_theta, 3),

    # YAW (ψ)
    "YAW / THRUST":   tf_from(A, B, C_psi, 0),
    "YAW / ELEVADOR": tf_from(A, B, C_psi, 1),
    "YAW / ALERON":   tf_from(A, B, C_psi, 2),
    "YAW / RUDDER":   tf_from(A, B, C_psi, 3),
}

# =========================================================
# ANALISIS
# =========================================================

def analyze(tf, name):

    poles = ct.poles(tf)

    print("\n==============================")
    print(name)
    print("==============================")

    print("Polos:")
    for p in poles:
        print(f"{p:.5f}")

    stable = np.all(np.real(poles) < 0)

    if stable:
        print("✔ ESTABLE")
    else:
        print("✘ INESTABLE")

    print(f"Max Re(p): {np.max(np.real(poles)):.5f}")

# =========================================================
# EJECUCIÓN
# =========================================================

for name, tf in TFs.items():
    analyze(tf, name)