import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Funciones auxiliares de cuaterniones
# =============================================================================

def qnorm(q):
    # Normaliza un cuaternión.
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Cuaternión con norma cero.")
    return q/n


def qconj(q):
    # Conjugado de cuaternión q = [v, s] -> q* = [-v, s].
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def qmul(p, q):
    # Producto de cuaterniones con escalar al final.
    pv, ps = p[:3], p[3]
    qv, qs = q[:3], q[3]

    v = qs*pv + ps*qv - np.cross(pv, qv)
    s = ps*qs - np.dot(pv, qv)

    return qnorm(np.r_[v, s])


def Omega(w):
    # Matriz Omega(w) para qdot = 0.5 Omega(w) q.
    wx, wy, wz = w

    return np.array([
        [0.0,  wz,  -wy,  wx],
        [-wz, 0.0,   wx,  wy],
        [ wy, -wx,  0.0,  wz],
        [-wx, -wy,  -wz, 0.0]
    ], dtype=float)


def qerr(qd, q):
    # Error de actitud qe = qd otimes q^{-1}. Se fuerza qe[3] >= 0.
    qe = qmul(qd, qconj(q))
    if qe[3] < 0.0:
        qe = -qe
    return qnorm(qe)


def angle_error(qe):
    # Error angular principal: theta = 2 acos(qe_4).
    return 2*np.arccos(np.clip(qe[3], -1.0, 1.0))

# =============================================================================
# Modelo dinámico e integrador
# =============================================================================

def dyn(q, w, L, J, Jinv):
    wdot = Jinv @ (L - np.cross(w, J @ w))
    qdot = 0.5 * Omega(w) @ q
    return qdot, wdot


def rk4(q, w, L, J, Jinv, dt):
    k1q, k1w = dyn(q, w, L, J, Jinv)
    k2q, k2w = dyn(qnorm(q + 0.5*dt*k1q), w + 0.5*dt*k1w, L, J, Jinv)
    k3q, k3w = dyn(qnorm(q + 0.5*dt*k2q), w + 0.5*dt*k2w, L, J, Jinv)
    k4q, k4w = dyn(qnorm(q + dt*k3q), w + dt*k3w, L, J, Jinv)

    qn = qnorm(q + (dt/6.0)*(k1q + 2*k2q + 2*k3q + k4q))
    wn = w + (dt/6.0)*(k1w + 2*k2w + 2*k3w + k4w)

    return qn, wn

# =============================================================================
# Parámetros físicos del CubeSat 1U
# =============================================================================

# Inercia [kg m^2]: cubo solido ~1.33 kg, arista 0.10 m -> (1/6) m a^2
J = np.diag([2.22e-3, 2.22e-3, 2.22e-3])
Jinv = np.linalg.inv(J)

# Condicion inicial de actitud: rotacion de 60 deg en eje arbitrario
theta0 = np.deg2rad(60.0)
w = np.array([0.05, -0.03, 0.04])       # velocidad angular inicial [rad/s]
qd = np.array([0.0, 0.0, 0.0, 1.0])     # actitud deseada: identidad

e0 = np.array([1.0, 1.0, 0.0])
e0 = e0/np.linalg.norm(e0)
q = qnorm(np.r_[e0*np.sin(theta0/2.0), np.cos(theta0/2.0)])

# =============================================================================
# SINTONIA DE GANANCIAS - Metodologia: asignacion de polos en cascada
# =============================================================================
#
# Lazo interno (velocidad angular -> torque):
#   Planta exacta (J isotropica -> termino giroscopico se anula): Js*wdot = L
#   PID en planta 1/(Js s): ecuacion caracteristica exacta de 2do orden
#       (Js + Kd) s^2 + Kp s + Ki = 0
#   Se iguala a s^2 + 2*xi*wn*s + wn^2 = 0 (con Kd=0 -> Jeff=Js)
#       Kp = 2*shi*wn*Js      Ki = wn^2*Js
#   omega_n acotado por saturacion de torque:
#       wn <= torque_max / (2*shi*Js*|ew|_max)
#   Con torque_max=2e-4 N m, ew_max~0.17 rad/s, shi=0.8 -> wn_max=0.331 rad/s
#   Se usa 70% de margen -> wn_rate = 0.2318 rad/s
#
# Lazo externo (error de actitud -> velocidad angular comandada):
#   Aproximacion de separacion de escalas (interno 7x mas rapido que externo):
#       eq_dot ~ 0.5*wc  (cinematica linealizada, b=0.5)
#   Mismo metodo algebraico, con "Jeff_att" = 1/b = 2:
#       Kp_att = 2*shi_att*wn_att*Jeff_att
#       Ki_att = wn_att^2*Jeff_att
#   wn_att = wn_rate / 7  (separacion de escalas de tiempo)
#
# NOTA: se descarto Ziegler-Nichols (metodo 2 / ganancia ultima) porque el
# lazo en cascada, al estar bien amortiguado y ser rapido internamente, solo
# oscila de forma sostenida en el regimen LINEAL con Ku~734 (fuera de todo
# rango fisico) y, con saturacion real activa, el "Ku" detectado (~123) es
# en realidad un ciclo limite de rele impuesto por wcmd_max, no una frontera
# de estabilidad util para sintonizar. Aplicar la formula de ZN con esos
# valores da ganancias que saturan el actuador el 100% del tiempo.
# =============================================================================

# Js = 2.22e-3
# shi_rate, wn_rate = 0.8, 0.2318      # lazo interno (ver Etapa 3)
# shi_att,  K_sep    = 0.8, 7.0        # lazo externo, separacion de escalas

# =========================
# PARÁMETROS FÍSICOS
# =========================
Js = 2.22e-3
b = 0.5
Jeff_att = 1.0 / b

# =========================
# ESPECIFICACIONES DE DISEÑO
# =========================

# amortiguamientos (robustos típicos)
xi_rate = 1
xi_att  = 1

# tiempos de establecimiento deseados (AJUSTABLES)
Ts_rate = 1.0   # [s] lazo interno (velocidad angular)
K_sep   = 10.0    # separación de escalas

# =========================
# DERIVACIÓN DE FRECUENCIAS NATURALES
# =========================

# relación segundo orden:
# Ts ≈ 4 / (xi * wn)
wn_rate = 4.0 / (xi_rate * Ts_rate)
wn_att = wn_rate / K_sep

print(f"wn_rate={wn_rate:.4f} rad/s  wn_att={wn_att:.4f} rad/s")

# PID externo: actitud -> velocidad angular comandada
Kp_att = 2*xi_att*wn_att*Jeff_att
Ki_att = wn_att**2 * Jeff_att
Kd_att = 0.0

# PID interno: velocidad angular -> torque
Kp_rate = np.diag([2*xi_rate*wn_rate*Js]*3)
Ki_rate = np.diag([wn_rate**2 * Js]*3)
Kd_rate = np.diag([0.0]*3)

print(f"Kp_att={Kp_att:.5f}  Ki_att={Ki_att:.5f}  Kd_att={Kd_att:.5f}")
print(f"Kp_rate={Kp_rate[0,0]:.6e}  Ki_rate={Ki_rate[0,0]:.6e}  Kd_rate={Kd_rate[0,0]:.6e}")

# Saturaciones (actuador realista para 1U: ruedas de reaccion micro)
torque_max = 1e-2       # [N m] saturacion por eje
wcmd_max = 0.5          # [rad/s] saturacion por eje
int_att_max = 0.5
int_rate_max = 0.5

# Tiempo: extendido para que el lazo externo (mas lento) alcance asentamiento
dt = 0.02
tf = 100.0
n_steps = int(tf/dt)

# =============================================================================
# Simulación del PID en cascada
# =============================================================================

Iatt = np.zeros(3)
Irate = np.zeros(3)
eq_prev = np.zeros(3)
ew_prev = np.zeros(3)

time = np.zeros(n_steps)
qhist = np.zeros((n_steps, 4))
whist = np.zeros((n_steps, 3))
Lhist = np.zeros((n_steps, 3))
wchist = np.zeros((n_steps, 3))
errhist = np.zeros(n_steps)

for k in range(n_steps):

    t = k*dt

    qe = qerr(qd, q)
    eq = qe[:3]

    # PID externo
    Iatt = np.clip(Iatt + eq*dt, -int_att_max, int_att_max)
    deq = (eq - eq_prev)/dt
    wc = Kp_att*eq + Ki_att*Iatt + Kd_att*deq
    wc = np.clip(wc, -wcmd_max, wcmd_max)
    eq_prev = eq.copy()

    # PID interno
    ew = wc - w
    Irate = np.clip(Irate + ew*dt, -int_rate_max, int_rate_max)
    dew = (ew - ew_prev)/dt
    L = Kp_rate@ew + Ki_rate@Irate + Kd_rate@dew
    L = np.clip(L, -torque_max, torque_max)
    ew_prev = ew.copy()

    q, w = rk4(q, w, L, J, Jinv, dt)

    time[k] = t
    qhist[k] = q
    whist[k] = w
    Lhist[k] = L
    wchist[k] = wc
    errhist[k] = angle_error(qe)

# =============================================================================
# Métricas cuantitativas
# =============================================================================

errdeg = np.rad2deg(errhist)
wnorm = np.linalg.norm(whist, axis=1)
Lnorm = np.linalg.norm(Lhist, axis=1)

idx_1deg = np.where(errdeg < 1.0)[0]
t_settle = time[idx_1deg[0]] if len(idx_1deg) > 0 else np.nan

print("================= RESULTADOS =================")
print(f"Error angular inicial       : {errdeg[0]:.4f} deg")
print(f"Error angular final         : {errdeg[-1]:.6f} deg")
print(f"Tiempo asentamiento (<1 deg): {t_settle:.2f} s")
print(f"Norma velocidad final       : {wnorm[-1]:.6e} rad/s")
print(f"Torque máximo aplicado      : {np.max(Lnorm):.6e} N m  (limite {torque_max:.1e})")
print(f"wc máximo comandado         : {np.max(np.abs(wchist)):.6f} rad/s  (limite {wcmd_max:.2f})")
print(f"Norma final del cuaternión  : {np.linalg.norm(qhist[-1]):.12f}")
print("==============================================")

# =============================================================================
# Paleta de colores consistente (mismos colores para cada eje/canal en TODAS
# las figuras)
# =============================================================================
colors_xyz  = {'x': '#1f77b4', 'y': '#ff7f0e', 'z': '#2ca02c'}   # azul, naranja, verde
limit_color = '#d62728'                                          # rojo -> líneas de límite/saturación
settle_color = 'black'                                           # línea de tiempo de asentamiento

# =============================================================================
# NOTA / SUPUESTO IMPORTANTE:
# Se asume que "qhist" es el CUATERNIÓN DE ERROR de actitud (converge a
# [0,0,0,1]), con convención escalar-al-final: qhist[:,0:3] = parte vectorial,
# qhist[:,3] = parte escalar. Si tu convención es escalar-primero
# (q4 = qhist[:,0]) o si qhist es la actitud absoluta (no el error), ajusta
# los índices de "quat2euler_321" más abajo.
# =============================================================================

def quat2euler_321(q1, q2, q3, q4):
    """
    Convierte cuaternión (escalar al final) a ángulos de Euler secuencia 3-2-1
    (yaw-pitch-roll). Devuelve roll, pitch, yaw en RADIANES.
    """
    roll  = np.arctan2(2*(q4*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2*(q4*q2 - q3*q1), -1.0, 1.0))
    yaw   = np.arctan2(2*(q4*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    return roll, pitch, yaw

roll_err, pitch_err, yaw_err = quat2euler_321(
    qhist[:,0], qhist[:,1], qhist[:,2], qhist[:,3]
)

roll_deg  = np.degrees(roll_err)
pitch_deg = np.degrees(pitch_err)
yaw_deg   = np.degrees(yaw_err)

euler_deg = {'x': roll_deg, 'y': pitch_deg, 'z': yaw_deg}
euler_labels = {'x': 'Roll', 'y': 'Pitch', 'z': 'Yaw'}

#  =============================================================================
# Figura 1 - Ángulo de actitud (Euler: roll, pitch, yaw) con banda de
# asentamiento y línea de tiempo de asentamiento
# =============================================================================
 
tol_deg = 1.0  # <-- tolerancia de la banda de asentamiento en grados (ajustar)
 
t_settle_per_axis = np.full(3, np.nan)
for idx, axis in enumerate(['x','y','z']):
    señal = euler_deg[axis]
    dentro = np.abs(señal) <= tol_deg
    fuera_idx = np.where(~dentro)[0]
    t_settle_per_axis[idx] = time[fuera_idx[-1]] if fuera_idx.size > 0 else time[0]
 
t_settle_euler = np.nanmax(t_settle_per_axis)
 
# -----------------------------------------------------------------------
# Sobreimpulso por eje.
# Definición: tras el primer cruce por cero (el ángulo pasa del signo
# inicial al signo contrario), se busca el valor pico y se compara con
# la magnitud inicial:
#     OS[%] = |pico tras cruce| / |valor inicial| * 100
# Si la señal nunca cruza el cero, se considera sobreimpulso = 0.
# -----------------------------------------------------------------------
overshoot_val = {}
overshoot_pct = {}
for axis in ['x','y','z']:
    señal = euler_deg[axis]
    signo0 = np.sign(señal[0]) if señal[0] != 0 else 1.0
    cruce_idx = np.where(np.sign(señal) != signo0)[0]
 
    if cruce_idx.size > 0 and señal[0] != 0:
        inicio = cruce_idx[0]
        pico_idx = inicio + np.argmax(np.abs(señal[inicio:]))
        pico_val = señal[pico_idx]
        overshoot_val[axis] = pico_val
        overshoot_pct[axis] = abs(pico_val) / abs(señal[0]) * 100
    else:
        overshoot_val[axis] = 0.0
        overshoot_pct[axis] = 0.0
 
plt.figure(figsize=(10,6))
 
for axis in ['x','y','z']:
    plt.plot(time, euler_deg[axis],
             color=colors_xyz[axis],
             linewidth=2,
             label=euler_labels[axis])
 
# Banda de asentamiento
plt.axhline(tol_deg, color='gray', linestyle=':', linewidth=1.8, label=f'Banda ±{tol_deg:.1f}°')
plt.axhline(-tol_deg, color='gray', linestyle=':', linewidth=1.8)
 
# Tiempo de asentamiento
plt.axvline(t_settle_euler, color=settle_color, linestyle='--', linewidth=2,
            label=f'$T_s$ = {t_settle_euler:.2f} s')
 
# Sobreimpulso por eje (línea horizontal en el valor pico, color del eje)
for axis in ['x','y','z']:
    plt.axhline(overshoot_val[axis],
                color=colors_xyz[axis],
                linestyle='-.',
                linewidth=1.5,
                label=f'OS {euler_labels[axis]} = {overshoot_val[axis]:.2f}° ({overshoot_pct[axis]:.1f}%)')
 
plt.grid(alpha=0.3)
 
plt.xlabel('Tiempo [s]', fontsize=15)
plt.ylabel('Ángulo de error [°]', fontsize=15)
plt.title('Ángulo de actitud (Euler)', fontsize=18)
 
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
 
plt.legend()
 
plt.tight_layout()
plt.show()
 
# =============================================================================
# Figura 2 - Velocidad angular del cuerpo
# =============================================================================

plt.figure(figsize=(10,6))

for label in ['x','y','z']:
    i = {'x':0,'y':1,'z':2}[label]
    plt.plot(time,
             whist[:,i],
             color=colors_xyz[label],
             linewidth=2,
             label=rf'$\omega_{label}$')

# plt.axhline(0.5, color=lor=limimit_color, linestyle=':', linewidth=2, label='Límite ±0.5')
# plt.axhline(-0.5, colit_color, linestyle=':', linewidth=2)

plt.grid(alpha=0.3)

plt.xlabel('Tiempo [s]', fontsize=15)
plt.ylabel(r'$\omega$ [rad/s]', fontsize=15)
plt.title('Velocidad angular del cuerpo', fontsize=18)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.legend(ncol=2)

plt.tight_layout()
plt.show()

# =============================================================================
# Figura 3 - Velocidad angular comandada
# =============================================================================

plt.figure(figsize=(10,6))

plt.plot(time, wchist[:,0], color=colors_xyz['x'], linewidth=2, label=r'$\omega_{c,x}$')
plt.plot(time, wchist[:,1], color=colors_xyz['y'], linewidth=2, label=r'$\omega_{c,y}$')
plt.plot(time, wchist[:,2], color=colors_xyz['z'], linewidth=2, label=r'$\omega_{c,z}$')

plt.axhline(0.5, color=limit_color, linestyle=':', linewidth=2, label='Límite ±0.5')
plt.axhline(-0.5, color=limit_color, linestyle=':', linewidth=2)

plt.grid(alpha=0.3)

plt.xlabel('Tiempo [s]', fontsize=15)
plt.ylabel(r'$\omega_c$ [rad/s]', fontsize=15)
plt.title('Velocidad angular comandada', fontsize=18)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# Figura 4 - Torque comandado
# =============================================================================

plt.figure(figsize=(10,6))

plt.plot(time, Lhist[:,0], color=colors_xyz['x'], linewidth=2, label='$L_x$')
plt.plot(time, Lhist[:,1], color=colors_xyz['y'], linewidth=2, label='$L_y$')
plt.plot(time, Lhist[:,2], color=colors_xyz['z'], linewidth=2, label='$L_z$')

plt.axhline(torque_max, color=limit_color, linestyle=':', linewidth=2, label='Límite')
plt.axhline(-torque_max, color=limit_color, linestyle=':', linewidth=2)

plt.grid(alpha=0.3)

plt.xlabel('Tiempo [s]', fontsize=15)
plt.ylabel('Torque [Nm]', fontsize=15)
plt.title('Torque comandado', fontsize=18)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.legend()

plt.tight_layout()
plt.show()


print("="*70)
print("MÉTRICAS - ÁNGULO DE ACTITUD (Euler)")
print("="*70)
for axis in ['x','y','z']:
    print(f"{euler_labels[axis]:>6s}: "
          f"error inicial = {euler_deg[axis][0]:8.3f}°   "
          f"error final = {euler_deg[axis][-1]:8.4f}°   "
          f"Ts (banda ±{tol_deg:.1f}°) = {t_settle_per_axis[{'x':0,'y':1,'z':2}[axis]]:6.2f} s   "
          f"Overshoot = {overshoot_val[axis]:7.3f}° ({overshoot_pct[axis]:5.1f}%)")
print(f"\nTs global (peor eje) = {t_settle_euler:.2f} s")
 
print("\n" + "="*70)
print("MÉTRICAS - VELOCIDAD ANGULAR DEL CUERPO (whist)")
print("="*70)
w_limit = 0.5
for label in ['x','y','z']:
    i = {'x':0,'y':1,'z':2}[label]
    max_abs = np.max(np.abs(whist[:,i]))
    rms = np.sqrt(np.mean(whist[:,i]**2))
    sat_pct = np.mean(np.abs(whist[:,i]) >= w_limit) * 100
    print(f"ω_{label}: máx = {max_abs:8.4f} rad/s   RMS = {rms:8.4f} rad/s   "
          f"saturado = {sat_pct:5.2f}%")
 
print("\n" + "="*70)
print("MÉTRICAS - VELOCIDAD ANGULAR COMANDADA (wchist)")
print("="*70)
for label in ['x','y','z']:
    i = {'x':0,'y':1,'z':2}[label]
    max_abs = np.max(np.abs(wchist[:,i]))
    sat_pct = np.mean(np.abs(wchist[:,i]) >= w_limit) * 100
    print(f"ω_c,{label}: máx = {max_abs:8.4f} rad/s   saturado = {sat_pct:5.2f}%")
 
print("\n" + "="*70)
print("MÉTRICAS - TORQUE COMANDADO (Lhist)")
print("="*70)
for label in ['x','y','z']:
    i = {'x':0,'y':1,'z':2}[label]
    max_abs = np.max(np.abs(Lhist[:,i]))
    sat_pct = np.mean(np.abs(Lhist[:,i]) >= torque_max) * 100
    print(f"L_{label}: máx = {max_abs:10.4e} Nm   límite = {torque_max:10.4e} Nm   "
          f"saturado = {sat_pct:5.2f}%")
 
print("\n" + "="*70)
print("MÉTRICAS GENERALES ADICIONALES")
print("="*70)
IAE_euler = {axis: np.trapezoid(np.abs(euler_deg[axis]), time) for axis in ['x','y','z']}
ISE_euler = {axis: np.trapezoid(euler_deg[axis]**2, time) for axis in ['x','y','z']}
for axis in ['x','y','z']:
    print(f"{euler_labels[axis]:>6s}: IAE = {IAE_euler[axis]:10.3f}   ISE = {ISE_euler[axis]:10.3f}")
 
print(f"\nDuración total de la simulación: {time[-1]:.2f} s")
print(f"Norma final del cuaternión de error ||q|| = {np.linalg.norm(qhist[-1]):.6f}")