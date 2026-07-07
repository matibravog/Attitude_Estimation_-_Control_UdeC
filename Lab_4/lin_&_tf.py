import sympy as sp

# ============================================================
# VARIABLES SIMBÓLICAS
# ============================================================

# Estados
q1, q2, q3 = sp.symbols('q1 q2 q3')
wx, wy, wz = sp.symbols('wx wy wz')

# Entradas (torques)
Lx, Ly, Lz = sp.symbols('Lx Ly Lz')

# Inercias
Jx, Jy, Jz = sp.symbols('Jx Jy Jz', positive=True)

# Variable de Laplace
s = sp.symbols('s')

# ============================================================
# MODELO LINEALIZADO
# ============================================================
#
# qdot = 1/2 w
#
# wdot = J^-1 L
#
# x=[q1 q2 q3 wx wy wz]^T
#
# ============================================================

x = sp.Matrix([q1, q2, q3, wx, wy, wz])
u = sp.Matrix([Lx, Ly, Lz])

f = sp.Matrix([
    wx/2,
    wy/2,
    wz/2,
    Lx/Jx,
    Ly/Jy,
    Lz/Jz
])

A = f.jacobian(x)
B = f.jacobian(u)

print("\n==================== MATRIZ A ====================")
sp.pprint(A)

print("\n==================== MATRIZ B ====================")
sp.pprint(B)

# ============================================================
# FUNCIONES DE TRANSFERENCIA
# ============================================================

print("\n")
print("="*60)
print("FUNCIONES DE TRANSFERENCIA")
print("="*60)

# ----------- Lazo interno -----------------

Gwx = sp.simplify((1/Jx)/s)
Gwy = sp.simplify((1/Jy)/s)
Gwz = sp.simplify((1/Jz)/s)

print("\nTorque -> Velocidad angular")

print("\nEje X")
print("Gwx(s)=")
sp.pprint(Gwx)

print("\nEje Y")
print("Gwy(s)=")
sp.pprint(Gwy)

print("\nEje Z")
print("Gwz(s)=")
sp.pprint(Gwz)

# ----------- Lazo externo -----------------

Gqx = sp.simplify(1/(2*Jx*s**2))
Gqy = sp.simplify(1/(2*Jy*s**2))
Gqz = sp.simplify(1/(2*Jz*s**2))

print("\n")
print("Torque -> Actitud")

print("\nEje X")
print("Gqx(s)=")
sp.pprint(Gqx)

print("\nEje Y")
print("Gqy(s)=")
sp.pprint(Gqy)

print("\nEje Z")
print("Gqz(s)=")
sp.pprint(Gqz)

# ============================================================
# NUMERADOR Y DENOMINADOR
# ============================================================

print("\n")
print("="*60)
print("COEFICIENTES")
print("="*60)

TF = {
    "Gwx": Gwx,
    "Gwy": Gwy,
    "Gwz": Gwz,
    "Gqx": Gqx,
    "Gqy": Gqy,
    "Gqz": Gqz,
}

for nombre, G in TF.items():

    num, den = sp.fraction(sp.together(G))

    print(f"\n{nombre}")

    print("Numerador:")
    sp.pprint(sp.Poly(num, s).all_coeffs())

    print("Denominador:")
    sp.pprint(sp.Poly(den, s).all_coeffs())