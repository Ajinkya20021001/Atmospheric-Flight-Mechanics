import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

# Function for equations of motion
def equations_of_motion(t, y, rho_sl, S, C_D0, K, W, m, g):
    v, gamma, Xe, h = y  # Velocity, flight path angle, horizontal position, altitude
    
    # Air density as a function of altitude
    beta = 9296
    rho = rho_sl * np.exp(-h / beta)
    
    # Calculate lift coefficient
    Cl = np.sqrt(C_D0 / K)
    
    # Calculate drag force
    D = 0.5 * rho * v**2 * S * (C_D0 + K * Cl**2)
    L = 0.5 * rho * v**2 * S * Cl
    
    # Equations of motion
    dvdt = (-D - W * np.sin(gamma)) / m
    dgamma_dt = (L - W * np.cos(gamma)) / m
    dXedt = v * np.cos(gamma)
    dhdt = v * np.sin(gamma)
    
    return [dvdt, dgamma_dt, dXedt, dhdt]

# Aircraft Parameters
c_bar = 1.211  # Mean Aerodynamic Chord (m)
b = 10.47  # Wing Span (m)
AR = 8.8  # Aspect Ratio
e = 0.9  # Oswald efficiency factor
S = 12.47  # Wing Area (m^2)
m = 750  # Mass (kg)
g = 9.81  # Gravitational acceleration (m/s^2)
W = m * g  # Weight (N)
C_D0 = 0.036  # Drag Coefficient at zero lift
C_Lmax = 1.8  # Maximum lift coefficient
K = 1 / (np.pi * e * AR)  # Induced drag constant

# Sea level density (rho_sl)
rho_sl = 1.225  # Air density at sea level (kg/m^3)

# Altitude range
altitudes = np.arange(0, 7000, 100)  # Altitudes from 0 to 10000 m

# Pre-allocate arrays for results
Rmax = np.zeros_like(altitudes, dtype=float)  # Maximum range
Emax = np.zeros_like(altitudes, dtype=float)  # Maximum endurance
V = np.zeros_like(altitudes, dtype=float)  # Velocity
gamma = np.zeros_like(altitudes, dtype=float)  # Flight path angle

# Loop over different altitudes
for i, h in enumerate(altitudes):
    beta = 9296  # Scale height (m)
    rho = rho_sl * np.exp(-h / beta)  # Air density at altitude
    
    # Rmax (Maximum Range)
    Rmax[i] = h / (2 * np.sqrt(C_D0 * K))
    
    # Cl and Cd for endurance
    Clprime = np.sqrt(C_D0 / K) * np.sqrt(3)
    Cdprime = C_D0 * 4
    
    # Emax (Maximum Endurance)
    Emax[i] = (((1 / np.sqrt(2 * (W / S))) * ((Clprime ** (3/2)) / Cdprime) * (2 * beta * (1 -  np.exp(-h / (2 * beta)))))) 
    
   
    #Cl and Cd for velocity and flight path angle
    Cl = np.sqrt(C_D0 / K)
    Cd = C_D0 + K * Cl**2
    # Velocity V
    V[i] = np.sqrt(2 * W / (rho * S * Cl)) 
    
    # Flight path angle (gamma)
    tan_gamma = -1 / (Cl / Cd)
    gamma[i] = np.arctan(tan_gamma)

# Display results
print('Altitude (m) | Range (m) | Endurance (hr) | Velocity (m/s) | Flight Path Angle (degrees)')
for i, h in enumerate(altitudes):
    print(f'{h:4.0f}\t\t{Rmax[i]:.2f}\t\t{Emax[i] / 3600:.2f}\t\t{V[i]:.2f}\t\t{np.degrees(gamma[i]):.2f}')

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot Maximum Range
axs[0, 0].plot(altitudes, Rmax, '-b', linewidth=1.5)
axs[0, 0].set_xlabel('Altitude (m)')
axs[0, 0].set_ylabel('Maximum Range (m)')
axs[0, 0].set_title('Maximum Range vs Altitude')
axs[0, 0].grid(True)

# Plot Maximum Endurance
axs[0, 1].plot(altitudes, Emax, '-b', linewidth=1.5)
axs[0, 1].set_xlabel('Altitude (m)')
axs[0, 1].set_ylabel('Maximum Endurance (hr)')
axs[0, 1].set_title('Maximum Endurance vs Altitude')
axs[0, 1].grid(True)

# Plot Velocity
axs[1, 0].plot(altitudes, V, '-b', linewidth=1.5)
axs[1, 0].set_xlabel('Altitude (m)')
axs[1, 0].set_ylabel('Velocity (m/s)')
axs[1, 0].set_title('Velocity vs Altitude')
axs[1, 0].grid(True)

# Plot Flight Path Angle
axs[1, 1].plot(altitudes, np.degrees(gamma), '-b', linewidth=1.5)
axs[1, 1].set_xlabel('Altitude (m)')
axs[1, 1].set_ylabel('Flight Path Angle (degrees)')
axs[1, 1].set_title('Flight Path Angle vs Altitude')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Initial conditions for numerical integration: [v, gamma, Xe, h]
initial_conditions = [V[0], gamma[0], 0, altitudes[0]]

# Time span for simulation
tspan = [0, 1000]  # Simulation time (s)

# Solve using solve_ivp
solution = solve_ivp(
    equations_of_motion, tspan, initial_conditions, args=(rho_sl, S, C_D0, K, W, m, g), dense_output=True
)

# Extract results
t = solution.t
y = solution.y

# Plot results of numerical integration
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(t, y[0])  # Velocity
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Velocity (m/s)')
axs[0, 0].set_title('Velocity vs Time')

axs[0, 1].plot(t, y[1])  # Flight path angle
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Flight Path Angle (rad)')
axs[0, 1].set_title('Flight Path Angle vs Time')

axs[1, 0].plot(t, y[2])  # Horizontal position
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Horizontal Position (m)')
axs[1, 0].set_title('Horizontal Position vs Time')

axs[1, 1].plot(t, y[3])  # Altitude
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Altitude (m)')
axs[1, 1].set_title('Altitude vs Time')

plt.tight_layout()
plt.show()

