import numpy as np
import control as ct
import matplotlib.pyplot as plt

# =========================
# FUNCIÓN DE TRANSFERENCIA ORIGINAL
# =========================

num_original = [
    -8.882e-15,
    38.95,
    67.14,
    3069,
    6628,
    -8.255e4,
    -8578,
    0
]

den = [
    1,
    9.908,
    110.2,
    858.4,
    1003,
    -1.324e4,
    -1375,
    0
]

# quitar ceros finales automáticos
num_original = np.trim_zeros(num_original, 'b')
den = np.trim_zeros(den, 'b')

# sistema original
sys_original = ct.tf(num_original, den)

# =========================
# SISTEMA FILTRADO
# (se elimina el término 1e-15)
# =========================

num_filtered = [
    38.95,
    67.14,
    3069,
    6628,
    -8.255e4,
    -8578,
    0
]

num_filtered = np.trim_zeros(num_filtered, 'b')

sys_filtered = ct.tf(num_filtered, den)

# =========================
# SIMULACIÓN STEP
# =========================

t = np.linspace(0, 1, 100)

t1, y1 = ct.step_response(sys_original, t)
t2, y2 = ct.step_response(sys_filtered, t)

# =========================
# GRÁFICA
# =========================

plt.figure(figsize=(10,6))

plt.plot(t1, y1, label="Original (con 1e-15)", linewidth=2)
plt.plot(t2, y2, '--', label="Filtrada (sin 1e-15)", linewidth=2)

plt.title("Respuesta al escalón - comparación de TF")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()

plt.show()