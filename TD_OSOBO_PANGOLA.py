import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Ressort:
    def __init__(self, masse, coefficient_frottement, constante_raideur, position_initiale, force_exterieure, frequence_angulaire):
        self.masse = masse
        self.coefficient_frottement = coefficient_frottement
        self.constante_raideur = constante_raideur
        self.position_initiale = position_initiale
        self.force_exterieure = force_exterieure
        self.frequence_angulaire = frequence_angulaire

    def equation_differentielle(self, X, temps):
        x, v = X
        dxdt = v
        dvdt = (self.force_exterieure * np.cos(self.frequence_angulaire * temps) - self.coefficient_frottement * v - self.constante_raideur * x) / self.masse
        return [dxdt, dvdt]

    def resoudre_equation_differentielle(self, position_initiale):
        temps = np.linspace(0, 10, 1000)
        return odeint(self.equation_differentielle, [position_initiale, 0], temps)

    def calculer_energie_mecanique(self, solution):
        energie_cinetique = 0.5 * self.masse * solution[:, 1]**2
        energie_potentielle = 0.5 * self.constante_raideur * solution[:, 0]**2
        energie_mecanique = energie_cinetique + energie_potentielle
        return energie_mecanique

    def tracer_solution(self, solution, titre):
        temps = np.linspace(0, 10, 1000)
        energie_mecanique = self.calculer_energie_mecanique(solution)

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(temps, solution[:, 0], label=f'Position initiale={self.position_initiale} m')
        plt.title(titre)
        plt.xlabel('Temps (s)')
        plt.ylabel('Déplacement (m)')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(temps, energie_mecanique, label='énergie mécanique')
        plt.title('énergie mécanique')
        plt.xlabel('Temps (s)')
        plt.ylabel('énergie')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(temps, solution[:, 1], label='Vitesse')
        plt.title('Vitesse')
        plt.xlabel('Temps (s)')
        plt.ylabel('Vitesse')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Paramères du système
masse = 10  # kg
coefficient_frottement = 20  # Ns/m
constante_raideur = 4000  # N/m
position_initiale = 0.01  # m
force_exterieure = 100  # N
frequence_angulaire = 10  # rad/s

# Création de l'objet Ressort
ressort = Ressort(masse, coefficient_frottement, constante_raideur, position_initiale, force_exterieure, frequence_angulaire)

# Cas (a) : Oscillations libres
solution_a = ressort.resoudre_equation_differentielle(position_initiale)
ressort.tracer_solution(solution_a, 'Oscillations libres (F(t)=0)')

# Cas (b) : Force extérieure
solution_b = ressort.resoudre_equation_differentielle(0)
ressort.tracer_solution(solution_b, 'Force extérieure (F(t)=F0*cos(ωt))')