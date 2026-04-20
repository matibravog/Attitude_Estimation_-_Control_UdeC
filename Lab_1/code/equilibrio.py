import sympy as sp
import math

# Variables simbólicas
U, T, de = sp.symbols('U T de')

# Constantes dadas
Cx_u = -0.1
Cz_u = 0.6
CM_de = -0.5
CM_u = 0.01

# Parámetros físicos
m = 507+200 # kg
g = 9.81 # m/s^2
S = 16.2  # m^2

# Ángulos
phi = 0
theta = math.radians(0)

# Ecuaciones
eq1 = T + 45*S*Cx_u*U - m*g*math.cos(theta)
eq2 = m*g*math.cos(phi)*math.cos(theta) - (45*S*Cz_u*U + 2700*S*CM_de*de)
eq3 = de + (45*CM_u*U)/(2700*CM_de)

# Resolver sistema
sol = sp.solve((eq1, eq2, eq3), (T, U, de))

# Mostrar resultados
T_sol = float(sol[T])
U_sol = float(sol[U])
de_sol = float(sol[de])

print(f"U = {U_sol:.4f} m/s")
print(f"U = {U_sol*3.6:.4f} km/h")
print(f"T = {T_sol:.4f}")
print(f"T = {T_sol/9.81:.4f} kg")
print(f"de = {de_sol:.4f} rad ({math.degrees(de_sol):.2f} deg)")