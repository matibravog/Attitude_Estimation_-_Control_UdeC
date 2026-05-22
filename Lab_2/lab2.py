# --------------------------------------------
# Para modificar condiciones iniciales de 
# velocidad de yaw, pitch, roll
# en linea : 232

# Para modificar condiciones inicales de
# torques externos en linea :   244
# --------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONSTANTES ORBITALES
# ============================================================

mu = 3.986e14                  # [m^3/s^2]
Re = 6371e3                    # [m]

h = 500e3                      # Altitud [m]
r_orbit = Re + h               # Radio orbital [m]

# Mean motion orbital
n = np.sqrt(mu / r_orbit**3)

# Periodo orbital
T_orbit = 2 * np.pi / n

# ============================================================
# TIEMPO DE SIMULACION
# ============================================================

dt = 0.1
t_final = T_orbit

t = np.arange(0, t_final, dt)
N = len(t)

# ============================================================
# TENSOR DE INERCIA
# ============================================================

# CAMBIAR AQUI EL TENSOR DE INERCIA REAL
I = np.diag([1.0, 1.0, 1.0])

I_inv = np.linalg.inv(I)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalize(v):
    norm = np.linalg.norm(v)

    if norm < 1e-12:
        return v

    return v / norm

# ------------------------------------------------------------

def quat_normalize(q):
    return q / np.linalg.norm(q)

# ------------------------------------------------------------

def quat_conjugate(q):

    return np.array([
        q[0],
        -q[1],
        -q[2],
        -q[3]
    ])

# ------------------------------------------------------------

def quat_multiply(q1, q2):

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([

        w1*w2 - x1*x2 - y1*y2 - z1*z2,

        w1*x2 + x1*w2 + y1*z2 - z1*y2,

        w1*y2 - x1*z2 + y1*w2 + z1*x2,

        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# ------------------------------------------------------------

def omega_matrix(w):

    wx, wy, wz = w

    return np.array([

        [0,   -wx, -wy, -wz],

        [wx,   0,   wz, -wy],

        [wy,  -wz,  0,   wx],

        [wz,   wy, -wx,  0 ]

    ])

# ------------------------------------------------------------

def quat_kinematics(q, w):

    return 0.5 * omega_matrix(w) @ q

# ------------------------------------------------------------

def rigid_body_dynamics(w, torque):

    return I_inv @ (
        torque - np.cross(w, I @ w)
    )

# --------------------------------------------q----------------

def dcm_to_quaternion(R):

    tr = np.trace(R)

    if tr > 0:

        S = np.sqrt(tr + 1.0) * 2

        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S

    else:

        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):

            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2

            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S

        elif R[1,1] > R[2,2]:

            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2

            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S

        else:

            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2

            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S

    q = np.array([qw, qx, qy, qz])

    return quat_normalize(q)

# ------------------------------------------------------------

def quaternion_to_dcm(q):

    qw, qx, qy, qz = q

    return np.array([

        [1 - 2*(qy**2 + qz**2),
         2*(qx*qy - qz*qw),
         2*(qx*qz + qy*qw)],

        [2*(qx*qy + qz*qw),
         1 - 2*(qx**2 + qz**2),
         2*(qy*qz - qx*qw)],

        [2*(qx*qz - qy*qw),
         2*(qy*qz + qx*qw),
         1 - 2*(qx**2 + qy**2)]

    ])

# ============================================================
# ARRAYS
# ============================================================

r_eci = np.zeros((N, 3))
v_eci = np.zeros((N, 3))

h_orbit = np.zeros((N, 3))

q_orbit = np.zeros((N, 4))

q_sat = np.zeros((N, 4))
w_sat = np.zeros((N, 3))

w_dot_hist = np.zeros((N, 3))

q_error = np.zeros((N, 4))

error_angle = np.zeros(N)

kinetic_energy = np.zeros(N)

angular_momentum_body = np.zeros((N, 3))

torques = np.zeros((N, 3))

# ============================================================
# CONDICIONES INICIALES
# ============================================================

# ------------------------------------------------------------
# VELOCIDAD ANGULAR INICIAL
# ------------------------------------------------------------
# CAMBIAR AQUI PARA:
# velocidades de roll, pitch, yaw iniciales

w_sat[0] = np.array([0.001, 0, n])

# ============================================================
# TORQUES EXTERNOS
# ============================================================

# ------------------------------------------------------------
# EJEMPLO FUTURO:
#
# for i in range(N):

#     if 1000 < t[i] < 1001:
#         torques[i] = np.array([0.01, 0, 0])

# ------------------------------------------------------------

# ============================================================
# PROPAGACION ORBITAL
# ============================================================

for i in range(N):

    theta = n * t[i]

    # --------------------------------------------------------
    # POSICION
    # --------------------------------------------------------

    r = r_orbit * np.array([
        np.cos(theta),
        np.sin(theta),
        0
    ])

    # --------------------------------------------------------
    # VELOCIDAD
    # --------------------------------------------------------

    v = np.sqrt(mu / r_orbit) * np.array([
        -np.sin(theta),
        np.cos(theta),
        0
    ])

    r_eci[i] = r
    v_eci[i] = v

    # --------------------------------------------------------
    # MOMENTUM ANGULAR ORBITAL
    # --------------------------------------------------------

    h_vec = np.cross(r, v)

    h_orbit[i] = h_vec

    # --------------------------------------------------------
    # FRAME LVLH
    # --------------------------------------------------------

    x_lvlh = -normalize(r)

    z_lvlh = normalize(h_vec)

    y_lvlh = np.cross(z_lvlh, x_lvlh)

    # --------------------------------------------------------
    # DCM LVLH -> ECI
    # --------------------------------------------------------

    R = np.column_stack((x_lvlh, y_lvlh, z_lvlh))

    q_orbit[i] = dcm_to_quaternion(R)
    
    # para arreglar FLIP
    if i > 0:

        if np.dot(q_orbit[i], q_orbit[i-1]) < 0:

            q_orbit[i] = -q_orbit[i]

q_sat[0]=q_orbit[0]

# ============================================================
# PROPAGACION DINAMICA Y CINEMATICA
# ============================================================

for i in range(N - 1):

    q = q_sat[i]
    w = w_sat[i]
    tau = torques[i]

    # --------------------------------------------------------
    # DINAMICA
    # --------------------------------------------------------

    w_dot = rigid_body_dynamics(w, tau)

    w_dot_hist[i] = w_dot

    # --------------------------------------------------------
    # CINEMATICA
    # --------------------------------------------------------

    q_dot = quat_kinematics(q, w)

    # --------------------------------------------------------
    # INTEGRACION EULER
    # --------------------------------------------------------

    w_sat[i+1] = w + w_dot * dt

    q_sat[i+1] = q + q_dot * dt

    # --------------------------------------------------------
    # NORMALIZACION
    # --------------------------------------------------------

    q_sat[i+1] = quat_normalize(q_sat[i+1])

# ============================================================
# ERROR DE ACTITUD
# ============================================================

for i in range(N):

    q_o = q_orbit[i]

    q_s = q_sat[i]

    q_error[i] = quat_multiply(
        quat_conjugate(q_o),
        q_s
    )

    q_error[i] = quat_normalize(q_error[i])

    # --------------------------------------------------------
    # ANGULO DE ERROR
    # --------------------------------------------------------

    qw = np.clip(q_error[i,0], -1.0, 1.0)

    error_angle[i] = 2 * np.arccos(qw)

    # --------------------------------------------------------
    # ENERGIA ROTACIONAL
    # --------------------------------------------------------

    kinetic_energy[i] = (
        0.5 * w_sat[i].T @ I @ w_sat[i]
    )

    # --------------------------------------------------------
    # MOMENTUM ANGULAR
    # --------------------------------------------------------

    angular_momentum_body[i] = I @ w_sat[i]

# ============================================================
# GRAFICOS
# ============================================================

# ------------------------------------------------------------
# QUATERNION ORBITAL
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(t, q_orbit[:,0], label='qw')
plt.plot(t, q_orbit[:,1], label='qx')
plt.plot(t, q_orbit[:,2], label='qy')
plt.plot(t, q_orbit[:,3], label='qz')

plt.title("Quaternion Orbital LVLH")
plt.xlabel("Tiempo [s]")
plt.ylabel("Quaternion")
plt.grid()
plt.legend()

# ------------------------------------------------------------
# QUATERNION SATELITE
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(t, q_sat[:,0], label='qw')
plt.plot(t, q_sat[:,1], label='qx')
plt.plot(t, q_sat[:,2], label='qy')
plt.plot(t, q_sat[:,3], label='qz')

plt.title("Quaternion Satélite")
plt.xlabel("Tiempo [s]")
plt.ylabel("Quaternion")
plt.grid()
plt.legend()

# ------------------------------------------------------------
# ERROR QUATERNION
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(t, q_error[:,0], label='qw')
plt.plot(t, q_error[:,1], label='qx')
plt.plot(t, q_error[:,2], label='qy')
plt.plot(t, q_error[:,3], label='qz')

plt.title("Quaternion Error")
plt.xlabel("Tiempo [s]")
plt.ylabel("Quaternion")
plt.grid()
plt.legend()

# ------------------------------------------------------------
# VELOCIDAD ANGULAR
# ------------------------------------------------------------

# plt.figure(figsize=(10,6))

# plt.plot(t, np.rad2deg(w_sat[:,0]), label='wx')
# plt.plot(t, np.rad2deg(w_sat[:,1]), label='wy')
# plt.plot(t, np.rad2deg(w_sat[:,2]), label='wz')

# plt.title("Velocidad Angular")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("deg/s")
# plt.grid()
# plt.legend()
# ------------------------------------------------------------
# ORBITA 3D
# ------------------------------------------------------------

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(
#     r_eci[:,0]/1e3,
#     r_eci[:,1]/1e3,
#     r_eci[:,2]/1e3
# )

# ax.set_title("Orbita en ECI")
# ax.set_xlabel("X [km]")
# ax.set_ylabel("Y [km]")
# ax.set_zlabel("Z [km]")

# ------------------------------------------------------------
# ANGULO DE ERROR
# ------------------------------------------------------------

# plt.figure(figsize=(10,6))

# plt.plot(t, np.rad2deg(error_angle))

# plt.title("Angulo de Error")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("deg")
# plt.grid()

# ------------------------------------------------------------
# MOMENTUM ANGULAR
# ------------------------------------------------------------

# plt.figure(figsize=(10,6))

# plt.plot(t, angular_momentum_body[:,0], label='Hx')
# plt.plot(t, angular_momentum_body[:,1], label='Hy')
# plt.plot(t, angular_momentum_body[:,2], label='Hz')

# plt.title("Momentum Angular")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Nms")
# plt.grid()
# plt.legend()

# ------------------------------------------------------------
# POSICION
# ------------------------------------------------------------

# plt.figure(figsize=(10,6))

# plt.plot(t, r_eci[:,0]/1e3, label='x')
# plt.plot(t, r_eci[:,1]/1e3, label='y')
# plt.plot(t, r_eci[:,2]/1e3, label='z')

# plt.title("Posicion ECI")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("km")
# plt.grid()
# plt.legend()

# # ------------------------------------------------------------
# # VELOCIDAD
# # ------------------------------------------------------------

# plt.figure(figsize=(10,6))

# plt.plot(t, v_eci[:,0], label='vx')
# plt.plot(t, v_eci[:,1], label='vy')
# plt.plot(t, v_eci[:,2], label='vz')

# plt.title("Velocidad ECI")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("m/s")
# plt.grid()
# plt.legend()

# ------------------------------------------------------------
# NORMA QUATERNION
# ------------------------------------------------------------

# plt.figure(figsize=(10,6))

# plt.plot(t, np.linalg.norm(q_sat, axis=1))

# plt.title("Norma Quaternion")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Norma")
# plt.grid()

plt.show()
