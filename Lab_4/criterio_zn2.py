import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Parámetros
# =====================================================

J = 2.22e-3          # Inercia eje X [kg·m²]

dt = 0.01
tf = 1000
t = np.arange(0, tf, dt)

# =====================================================
# Ganancia proporcional (ir variándola)
# =====================================================

Kp = 0.000001          # Cambiar este valor

torque_max = 2e-4    # Saturación del actuador

# =====================================================
# Condiciones iniciales
# =====================================================

theta = np.deg2rad(20)     # Error inicial
omega = 0

theta_hist = []
omega_hist = []
torque_hist = []

# =====================================================
# Simulación
# =====================================================

for _ in t:

    # Control proporcional
    torque = -Kp * theta

    # Saturación
    torque = np.clip(torque,
                     -torque_max,
                      torque_max)

    # Dinámica
    omega_dot = torque / J

    # Euler
    omega += omega_dot * dt
    theta += omega * dt

    theta_hist.append(theta)
    omega_hist.append(omega)
    torque_hist.append(torque)

theta_hist = np.array(theta_hist)
omega_hist = np.array(omega_hist)
torque_hist = np.array(torque_hist)

# =====================================================
# Resultados
# =====================================================

plt.figure(figsize=(12,8))

# plt.plot()
plt.plot(t, np.rad2deg(theta_hist))
plt.grid(True)
plt.ylabel("Ángulo [deg]")
plt.title(f"Respuesta con Kp = {Kp}")

# plt.subplot(312)
# plt.plot(t, omega_hist)
# plt.grid(True)
# plt.ylabel("Velocidad [rad/s]")

# plt.subplot(313)
# plt.plot(t, torque_hist)
# plt.grid(True)
# plt.ylabel("Torque [N·m]")
# plt.xlabel("Tiempo [s]")

plt.tight_layout()
plt.show()