import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONSTANTES ORBITALES
# =========================
mu = 3.986e14          # [m^3/s^2] Tierra
Re = 6371e3            # [m]
h = 500e3              # altitud LEO
r_mag = Re + h

n = np.sqrt(mu / r_mag**3)  # mean motion

# tiempo de simulación
T_orbit = 2 * np.pi / n
t = np.linspace(0, T_orbit, 500)

# =========================
# ÓRBITA CIRCULAR EN ECI
# =========================
theta = n * t

r = np.zeros((len(t), 3))
v = np.zeros((len(t), 3))

for i in range(len(t)):
    r[i] = r_mag * np.array([
        np.cos(theta[i]),
        np.sin(theta[i]),
        0
    ])

    v[i] = np.sqrt(mu / r_mag) * np.array([
        -np.sin(theta[i]),
        np.cos(theta[i]),
        0
    ])

# =========================
# LVLH FRAME
# =========================
def normalize(x):
    return x / np.linalg.norm(x)

q = np.zeros((len(t), 4))  # quaternion ECI->LVLH

for i in range(len(t)):
    r_hat = normalize(r[i])
    v_hat = normalize(v[i])

    z_LVLH = normalize(np.cross(r[i], v[i]))
    x_LVLH = -r_hat
    y_LVLH = np.cross(z_LVLH, x_LVLH)

    R = np.vstack([x_LVLH, y_LVLH, z_LVLH]).T

    # matriz a quaternion
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

    q[i] = np.array([qw, qx, qy, qz])

# =========================
# RESULTADOS
# =========================
plt.figure()
plt.plot(t, q[:,0], label='qw')
plt.plot(t, q[:,1], label='qx')
plt.plot(t, q[:,2], label='qy')
plt.plot(t, q[:,3], label='qz')
plt.legend()
plt.title("Quaternion ECI → LVLH")
plt.xlabel("Tiempo [s]")
plt.grid()
plt.show()