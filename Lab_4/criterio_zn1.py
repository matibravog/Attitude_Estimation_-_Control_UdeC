import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Dinámica del satélite
# ==========================================================

J = np.diag([2.22e-3, 2.22e-3, 2.22e-3])
Jinv = np.linalg.inv(J)

def dynamics(w, L):
    return Jinv @ (L - np.cross(w, J @ w))

# ==========================================================
# Simulación en lazo abierto
# ==========================================================

dt = 0.01
tf = 40
t = np.arange(0, tf, dt)

# Estado inicial
w = np.zeros(3)

# Escalón de torque (eje X)
L = np.array([1e-4, 0, 0])

whist = np.zeros((len(t),3))
thetahist = np.zeros(len(t))

theta = 0

for k in range(len(t)):

    wdot = dynamics(w, L)

    # Euler
    w += wdot*dt
    theta += w[0]*dt

    whist[k] = w
    thetahist[k] = theta

# ==========================================================
# Gráficas
# ==========================================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(t, whist[:,0])
plt.grid()
plt.xlabel("Tiempo [s]")
plt.ylabel(r"$\omega_x$ [rad/s]")
plt.title("Respuesta en velocidad")

plt.subplot(1,2,2)
plt.plot(t, np.rad2deg(thetahist))
plt.grid()
plt.xlabel("Tiempo [s]")
plt.ylabel("Ángulo [deg]")
plt.title("Respuesta en posición")

plt.tight_layout()
plt.show()