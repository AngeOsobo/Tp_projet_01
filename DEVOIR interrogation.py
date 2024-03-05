import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Paramètres du système
m = 10  # kg
a = 20  # Ns/m
k = 4000  # N/m
X0_1 = 0.01  # m
X0_2 = 0  # m
F0 = 100  # N
omega = 10  # rad/s

# Définition de l'équation différentielle
def model(X, t):
    x, v = X
    dxdt = v
    dvdt = (F0 * np.cos(omega * t) - a * v - k * x) / m
    return [dxdt, dvdt]

# Cas (a) : Oscillations libres
t_a = np.linspace(0, 10, 1000)
solution_a_1 = odeint(model, [X0_1, 0], t_a)
solution_a_2 = odeint(model, [X0_2, 0], t_a)

# Cas (b) : Force extérieure
t_b = np.linspace(0, 10, 1000)
solution_b_1 = odeint(model, [X0_1, 0], t_b)
solution_b_2 = odeint(model, [X0_2, 0], t_b)

# Calcul de l'énergie cinétique et potentielle pour le cas (a)
kinetic_energy_a = 0.5 * m * solution_a_1[:, 1]**2
potential_energy_a = 0.5 * k * solution_a_1[:, 0]**2

# Plot pour le cas (a)
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_a, solution_a_1[:, 0], label='X0=0.01 m')
plt.plot(t_a, solution_a_2[:, 0], label='X0=0 m')
plt.title('Oscillations libres (F(t)=0)')
plt.xlabel('Temps (s)')
plt.ylabel('Déplacement (m)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_a, kinetic_energy_a, label='énergie cinétique')
plt.plot(t_a, potential_energy_a, label='énergie potentielle')
plt.title('énergies cinétique et potentielle (F(t)=0)')
plt.xlabel('Temps (s)')
plt.ylabel('énergie')
plt.legend()

plt.tight_layout()
plt.show()