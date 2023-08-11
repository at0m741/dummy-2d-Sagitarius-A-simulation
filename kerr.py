import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Circle
import math

G = 6.67430e-11  # m^3/kg.s^2
c = 299792458  # m/s
Msun = 1.98847e30  # kg
R_g = 6.957e8 
M = 4.31 * Msun
a = 0.5 * 10**9 * G * Msun / c**2
Rs = 2*G*M/c**2
distance_sgr_a = 8000  # Distance estimée de Sagittarius A* en parsecs
r = distance_sgr_a * 3.086 * 10**16


def kerr_gravity(r, theta):
    Delta = r**2 - 2*M*r + a**2
    A = (r**2 + a**2)**2 - a**2*Delta*np.sin(theta)**2
    B = 2*r*(r**2 + a**2)*np.sin(theta)**2
    C = Delta - a**2*np.sin(theta)**2
    F = (r**2 + a**2*np.cos(theta)**2)**2 - Delta*a**2*np.sin(theta)**2
    gx = -2*G*M*r*A/(F**2 - B**2)
    gy = -2*G*M*r*C*np.sin(theta)**2/(F**2 - B**2)
    return gx, gy

def plot_particle_trajectories(ax, r0, theta0, phi0, E, Lz, T):
    t0 = 0
    p0 = np.array([r0, theta0, phi0, 0, Lz, E])

def f(t, p):
    r, theta, phi, pr, ptheta, pphi = p
    Delta = r**2 - 2*M*r + a**2
    A = (r**2 + a**2)**2 - a**2*Delta*np.sin(theta)**2
    B = 2*r*(r**2 + a**2)*np.sin(theta)**2
    C = Delta - a**2*np.sin(theta)**2
    F = (r**2 + a**2*np.cos(theta)**2)**2 - Delta*a**2*np.sin(theta)**2
    
    fr = pr**2 + kerr_gravity(r, theta)[0] - (Lz - a*E*np.sin(theta))**2/(np.sin(theta)**2)
    ftheta = ptheta**2 + (Lz**2/np.sin(theta)**2 - a**2*E**2*np.sin(theta)**2 - kerr_gravity(r, theta)[1])
    fphi = 2*(Lz/E)*pr*np.sin(theta)**2
    
    G = (r**2 + a**2 - 2*M*r)*E**2 - 2*a*Lz*E + pr**2 + ptheta**2/np.sin(theta)**2
    
    frdot = (-1/F)*((r - M)*(r**2 + a**2)*E**2 - 2*M*r*a*Lz*E*np.sin(theta)**2 + pr*(r**2 + a**2 - 2*M*r)*(pr**2 + G))
    fthetadot = (-1/(2*F))*(A*(r**2 + a**2 - 2*M*r)*np.sin(2*theta) + 2*C*M*r*(pr**2 + G)*np.sin(theta)**2 - ptheta**2*(r**2 + a**2 - 2*M*r)*np.sin(2*theta)/np.sin(theta)**4)
    fphidot = (1/F)*(-2*a*M*r*E + 2*a*Lz*(r - M)*np.sin(theta)**2 + 2*pr*Lz*np.sin(theta)**2/np.sin(theta)**2)
    print(pr, ptheta, pphi, frdot, fthetadot, fphidot)
    return [pr, ptheta, pphi, frdot, fthetadot, fphidot] 

def acceleration(x, y):
    r = np.sqrt(x**2 + y**2)
    a = -G * M / (r**2)
    ax = a * x / r
    ay = a * y / r
    return ax, ay



def kerr_geodesic(x, y, z, vx, vy, vz):
    m = 10e-54
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    g00 = -(1 - (2 * G * M) / (c**2 * R_g))  # Expression de g00 en fonction de M et R_g
    g03 = -((2 * a * G * M * np.sin(theta)**2) / (c**2 * R_g))  # Expression de g03 en fonction de a, M, R_g et theta
    g33 = (r**2 + a**2 + (2 * G * M * a**2 * np.sin(theta)**2) / (c**2 * R_g)) * np.sin(theta)**2  # Expression de g33 en fonction de r, a, M, R_g et theta
    g22 = r**2 + a**2 * np.cos(theta)**2  # Expression de g22 en fonction de r, a et theta
    Delta = r**2 - 2*M*r + a**2
    A = (r**2 + a**2)**2 - a**2*Delta*np.sin(theta)**2
    B = 2*r*(r**2 + a**2)*np.sin(theta)**2
    C = Delta - a**2*np.sin(theta)**2
    F = (r**2 + a**2*np.cos(theta)**2)**2 - Delta*a**2*np.sin(theta)**2

    E = -g00*c**2*vz - g03*c*vy + g33*vx
    Lz = g03*c**2*vz + g33*c*vy - g22*vx

    pr = (1/F)*(-A*E + Lz**2*np.sin(theta)**2 + C*M*r*E/Delta - (r**2 + a**2)*(E**2 - Delta*(Lz**2 + (r**2 + a**2)*(E**2 - Delta*(Lz**2 + 1/4*(m**2 + (1/np.sin(theta))**2))))))
    ptheta = (1/F)*(-1*np.sqrt(A)*np.sqrt(-C)*Lz*np.cos(theta) + np.sqrt(-A)*np.sqrt(C)*E*np.sin(theta)*np.sqrt(F/Delta) - pr*np.cos(theta)/np.sin(theta)*(r**2 + a**2)*E**2 + (r**2 + a**2)**2*pr*(E**2 - Delta*(Lz**2/pr + 1/4*(m**2 + (1/np.sin(theta))**2))))
    fphi = 2*(Lz/E)*pr*np.sin(theta)**2
    fpr = -(1/2)*((A+C*np.cos(theta)**2)*E**2 - 2*pr*(M*r*E - a*Lz*np.sin(theta))/Delta)
    fptheta = (1/2)*(A*np.cos(theta)*np.sin(theta)**2 - C*np.sin(theta)*np.cos(theta)*E - ptheta*np.cos(theta)/np.sin(theta)*(r**2 + a**2)*E**2 + (r**2 + a**2)**2*ptheta*(E**2 - Delta*(Lz**2/pr + 1/4*(m**2 + (1/np.sin(theta))**2))))
    fx = np.sin(theta)*np.cos(phi)*vx + np.cos(theta)*np.cos(phi)*vy - np.sin(phi)*vz
    fy = np.sin(theta)*np.sin(phi)*vx + np.cos(theta)*np.sin(phi)*vy + np.cos(phi)*vz
    fz = np.cos(theta)*vx - np.sin(theta)*vy
    print(fx, fy, fz, fphi, fpr, fptheta)
    return fx, fy, fz, fphi, fpr, fptheta


# Fonction qui calcule la trajectoire d'une particule à partir de ses coordonnées initiales
def compute_trajectory(x0, y0, vx0, vy0, dt, n_steps):
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    vx = np.zeros(n_steps)
    vy = np.zeros(n_steps)

    # Initialisation des variables
    x[0] = x0
    y[0] = y0
    vx[0] = vx0
    vy[0] = vy0

    # Calcul de la trajectoire
    for i in range(1, n_steps):
        ax, ay = acceleration(x[i-1], y[i-1])
        vx[i] = vx[i-1] + ax * dt
        vy[i] = vy[i-1] + ay * dt
        x[i] = x[i-1] + vx[i-1] * dt  # Utilisation de vx[i-1] à la place de vx[i]
        y[i] = y[i-1] + vy[i-1] * dt  # Utilisation de vy[i-1] à la place de vy[i]

    return x, y


# Fonction qui trace les géodésiques
def plot_geodesics(x0_list, y0_list, vx0_list, vy0_list, dt, n_steps):
    # Création de la figure
    fig, ax = plt.subplots()
    R = 2*G*M/c**2

    # Définition de la grille pour le tracé des géodésiques
    x_grid = np.linspace(-R, R, 100)
    y_grid = np.linspace(-R, R, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Calcul des accélérations sur la grille
    ax_grid, ay_grid = acceleration(X, Y)

    # Tracé des géodésiques
    for x0, y0, vx0, vy0 in zip(x0_list, y0_list, vx0_list, vy0_list):
        x, y = compute_trajectory(x0, y0, vx0, vy0, dt, n_steps)
        ax.plot(x, y)
    
    # Tracé des accélérations
    ax.quiver(X, Y, ax_grid, ay_grid)

    # Configuration de la figure
    ax.set_aspect('equal')
    ax.set_xlim([-R, R])
    ax.set_ylim([-R, R])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Géodésiques dans le champ gravitationnel terrestre')

    # Affichage de la figure
    plt.show()


# def generate_magnetic_perturbations(radius, strength, num_points, num_lines):
#     # Génération des coordonnées radiales
#     r = np.linspace(-radius, radius, num_points)

#     # Génération des perturbations magnétiques
#     magnetic_field = np.zeros((num_lines, num_points))
#     for i in range(num_lines):
#         phase = np.random.uniform(-2*np.pi, 2*np.pi)
#         frequency = np.random.uniform(-2, 5)
#         amplitude = np.random.uniform(-1, 5)
#         perturbation = amplitude * np.sin(frequency * r + phase)
#         magnetic_field[i] = strength * perturbation

#     # Ajout de bruit aléatoire aux perturbations magnétiques
#     noise = np.random.normal(0, 0.1 * strength, (num_lines, num_points))
#     magnetic_field += noise

#     return r, magnetic_field





# radius = 6  # Rayon maximal
# strength = 9000  # Intensité du champ magnétique
# num_points = 100000  # Nombre de points
# num_lines = 5  # Nombre de lignes distordues
# # Génération des perturbations magnétiques
# r, magnetic_field = generate_magnetic_perturbations(radius, strength, num_points, num_lines)
# print('Magnetic field \n', magnetic_field)
# print(len(magnetic_field))


r = np.linspace(0, 10*G*Msun/c**2, 320)
theta = np.linspace(0, np.pi, 320)
phi = np.linspace(0, 2*np.pi, 320)
R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')


N = 120 
X, Y, Z = np.zeros((N,N,N)), np.zeros((N,N,N)), np.zeros((N,N,N))

for i in range(N):
    for j in range(N):
        for k in range (N):
            Theta[i,j] = i*np.pi/N
            Phi[i,j] = j*2*np.pi/N
            X[i,j,k] = Rs*np.sqrt(2*(1-np.cos(Theta[i,j,k])))*np.sin(Phi[i,j,k])
            Y[i,j,k] = Rs*np.sqrt(2*(1-np.cos(Theta[i,j,k])))*np.cos(Phi[i,j,k])
            Z[i,j,k] = Rs*np.sqrt((2*G*M*R_g)/Rs) + 0.2*Rs*np.cos(Theta[i,j,k])

Req = np.zeros_like(X[:,:,0])
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if ((G*M/c**2)**2 - a**2*np.cos(Theta[i,j])**2 < 0).any():
            Req[i,j] = np.nan
        else:
            Req[i,j] = 2*G*M/c**2 + 2*np.sqrt((G*M/c**2)**2 - a**2*np.cos(Theta[i,j])**2)[0]

R_in = 1.5 * Rs
R_out = 25 * Rs 
h = Rs 
rho0 = 1e-5 
# Définir les conditions initiales
x0 = 10.0  # position initiale x
y0 = 10.0  # position initiale y
z0 = 10.0  # position initiale z
vx0 = 1.0  # vitesse initiale en x
vy0 = 1.0  # vitesse initiale en y
vz0 = 1.0  # vitesse initiale en z

# Définir les pas de temps et le nombre d'itérations
dt = 0.10  # pas de temps
num_iterations = 5000

# Tableaux pour stocker les coordonnées des trajectoires
x_traj = [x0]
y_traj = [y0]

# Boucle d'itération pour calculer les trajectoires
for i in range(num_iterations):
    # Calculer les accélérations
    ax, ay, _, _, _, _ = kerr_geodesic(x_traj[-1], y_traj[-1], z0, vx0, vy0, vz0)
    
    # Mettre à jour les positions et les vitesses
    x_new = x_traj[-1] + vx0 * dt + 0.5 * ax * dt**2
    y_new = y_traj[-1] + vy0 * dt + 0.5 * ay * dt**2
    
    vx_new = vx0 + ax * dt
    vy_new = vy0 + ay * dt
    
    # Ajouter les nouvelles positions aux tableaux des trajectoires
    x_traj.append(x_new)
    y_traj.append(y_new)
    
    # Mettre à jour les vitesses initiales
    vx0 = vx_new
    vy0 = vy_new

r_disk = np.linspace(R_in, R_out, 160)
theta_disk = np.linspace(0, 2*np.pi, 120)
r_disk, theta_disk = np.meshgrid(r_disk, theta_disk)

sigma_disk = rho0 * (Rs/r_disk)**(3/2)

x_disk = r_disk * np.sin(theta_disk)
y_disk = r_disk * np.cos(theta_disk)
z_disk = np.zeros_like(x_disk)
theta_sing, phi_sing = np.meshgrid(np.linspace(0, np.pi, 25), np.linspace(0, 2*np.pi, 100))
r_sing = 2*M*(1 + np.cos(theta_sing))
x_sing = r_sing*np.sin(theta_sing)*np.cos(phi_sing)
y_sing = r_sing*np.sin(theta_sing)*np.sin(phi_sing)
z_sing = r_sing*np.cos(theta_sing)

def kerr_ring_singularity(a, theta, phi):
    rho = np.sqrt(a**2 + np.sin(theta)**2)
    r = rho + np.sqrt(rho**2 - a**2*np.sin(theta)**2)
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    x += X
    y += Y
    z += Z
    return fx, fy, fz

# Ajout de la singularité dans le graphique

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
plt.plot(theta, phi, x_sing, y_sing)


    
disk = ax.contourf(x_disk, y_disk, sigma_disk, 400, cmap='inferno', label='accretion')
ergo_contour = ax.contour(X[:,:,0], Y[:,:,0], Z[:,:,1], zdir='z', levels=[0], alpha=1, colors='white', label='ergosphere')
bh_contour = ax.contour(X[:,:,0], Y[:,:,0], Z[:,:,0], zdir='z', levels=[0], alpha=1, colors='red')
sing_contour = ax.contour(X[:,:,0], Y[:,:,0], Z[:,:,1], zdir='z', levels=[0], alpha=0.5, colors='black', label='singularity')
# geod = ax.plot(x_traj, y_traj, 'b', label='particles trajectory', linestyle='--')
circle = Circle((0, 0), Rs, color='black', fill=False, label='Black hole')
ax.add_patch(circle)
print('Rs = ', Rs)
plt.legend()
# Ajouter une grille
ax.grid(True)
# add legend
# sing_legend.legendHandles[0].set_color('black')

# combine legends into one

# ax.add_artist(sing_legend)
# for i in range(num_lines):
#     ax.plot(magnetic_field[i], label='magneto perturbations {i+1}', alpha=0.7, color='white')
ax.plot(r, label='r')
ax.set_xlabel('RS X axis in meters')
ax.set_ylabel('RS Y axis in meters')
ax.legend()
ax.set_xlim(0, 50*G*Msun/c**2)
ax.set_ylim(-70*G*Msun/c**2, 70*G*Msun/c**2)
ax.set_title('Sagittarius A* 2D sim with potentials magnetic perturbations', size=20)
ax.set_label('graphique de représentation de l\'érgosphere et du rayon de Schwarzschild de Sagittarius A*')
plt.show()

