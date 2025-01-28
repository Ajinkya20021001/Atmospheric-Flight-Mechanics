import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # Gravity (m/s^2)
rho0 = 1.225  # Air density at sea level (kg/m^3)
P_SL = 74570  # Maximum shaft power at sea level (W)
eta_p = 0.8  # Propeller efficiency
C_D0 = 0.036  # Zero-lift drag coefficient
e = 0.8  # Oswald efficiency factor
AR = 8.8  # Aspect ratio
K = 1 / (np.pi * e * AR)  # Induced drag factor
S = 12.47  # Wing area (m^2)
W = 750 * g  # Aircraft weight (N)
velocities = np.linspace(1, 60, 50)  # Velocity range (m/s)
altitudes = np.arange(0, 5001, 500)  # Altitudes (m)
target_roc = 0.5  # Target rate of climb (m/s)

# Initialize Result Arrays
roc_values = np.zeros_like(altitudes, dtype=float)
pa_values = np.zeros_like(altitudes, dtype=float)
excess_power = np.zeros((len(altitudes), len(velocities)))
roc = np.zeros((len(altitudes), len(velocities)))
max_roc_velocity = np.zeros(len(altitudes))

# Altitude vs Rate of Climb Analysis
for i, h in enumerate(altitudes):
    rho = rho0 * (1 - h / 44330) ** 4.256
    Pa = eta_p * P_SL * (rho / rho0) ** (1/3)
    pa_values[i] = Pa

    for j, V in enumerate(velocities):
        CL = W / (0.5 * rho * V**2 * S)
        CD = C_D0 + K * CL**2
        D = 0.5 * rho * V**2 * S * CD
        Pr = D * V
        excess_power[i, j] = Pa - Pr

        if excess_power[i, j] > 0:
            roc[i, j] = excess_power[i, j] / W

    roc_values[i] = np.max(roc[i, :])
    max_roc_velocity[i] = velocities[np.argmax(roc[i, :])]
    

# Find Ceilings
absolute_ceiling = altitudes[np.where(roc_values <= 0)[0][0]] if np.any(roc_values <= 0) else np.nan
service_ceiling = altitudes[np.where(roc_values >= target_roc)[0][-1]] if np.any(roc_values >= target_roc) else np.nan

# Create UI for parameter input
import tkinter as tk
from tkinter import ttk

def get_parameters():
    global g, rho0, P_SL, eta_p, C_D0, e, AR, K, S, W

    def apply_parameters():
        try:
            # Update global parameters with user input
            g = float(g_entry.get())
            rho0 = float(rho_entry.get()) 
            P_SL = float(power_entry.get())
            eta_p = float(eff_entry.get())
            C_D0 = float(cd0_entry.get())
            e = float(e_entry.get())
            AR = float(ar_entry.get())
            K = 1 / (np.pi * e * AR)
            S = float(s_entry.get())
            W = float(weight_entry.get()) * g
            window.destroy()
        except ValueError:
            error_label.config(text="Please enter valid numbers")

    window = tk.Tk()
    window.title("Aircraft Parameters")
    
    # Create and pack widgets
    tk.Label(window, text="Enter aircraft parameters:").grid(row=0, column=0, columnspan=2, pady=5)
    
    # Gravity
    tk.Label(window, text="Gravity (m/s²):").grid(row=1, column=0)
    g_entry = ttk.Entry(window)
    g_entry.insert(0, str(g))
    g_entry.grid(row=1, column=1)
    
    # Air density
    tk.Label(window, text="Air density at sea level (kg/m³):").grid(row=2, column=0)
    rho_entry = ttk.Entry(window)
    rho_entry.insert(0, str(rho0))
    rho_entry.grid(row=2, column=1)
    
    # Power
    tk.Label(window, text="Max shaft power at sea level (W):").grid(row=3, column=0)
    power_entry = ttk.Entry(window)
    power_entry.insert(0, str(P_SL))
    power_entry.grid(row=3, column=1)
    
    # Propeller efficiency
    tk.Label(window, text="Propeller efficiency:").grid(row=4, column=0)
    eff_entry = ttk.Entry(window)
    eff_entry.insert(0, str(eta_p))
    eff_entry.grid(row=4, column=1)
    
    # Zero-lift drag coefficient
    tk.Label(window, text="Zero-lift drag coefficient:").grid(row=5, column=0)
    cd0_entry = ttk.Entry(window)
    cd0_entry.insert(0, str(C_D0))
    cd0_entry.grid(row=5, column=1)
    
    # Oswald efficiency
    tk.Label(window, text="Oswald efficiency factor:").grid(row=6, column=0)
    e_entry = ttk.Entry(window)
    e_entry.insert(0, str(e))
    e_entry.grid(row=6, column=1)
    
    # Aspect ratio
    tk.Label(window, text="Aspect ratio:").grid(row=7, column=0)
    ar_entry = ttk.Entry(window)
    ar_entry.insert(0, str(AR))
    ar_entry.grid(row=7, column=1)
    
    # Wing area
    tk.Label(window, text="Wing area (m²):").grid(row=8, column=0)
    s_entry = ttk.Entry(window)
    s_entry.insert(0, str(S))
    s_entry.grid(row=8, column=1)
    
    # Weight
    tk.Label(window, text="Aircraft mass (kg):").grid(row=9, column=0)
    weight_entry = ttk.Entry(window)
    weight_entry.insert(0, str(W/g))
    weight_entry.grid(row=9, column=1)
    
    # Error label
    error_label = tk.Label(window, text="", fg="red")
    error_label.grid(row=10, column=0, columnspan=2)
    
    # Buttons
    ttk.Button(window, text="Use Default", command=window.destroy).grid(row=11, column=0, pady=10)
    ttk.Button(window, text="Apply", command=apply_parameters).grid(row=11, column=1, pady=10)
    
    window.mainloop()

# Show parameter input dialog
get_parameters()


# Plot: Rate of Climb vs Altitude
plt.figure()
plt.plot(roc_values, altitudes, 'b-', label='Rate of Climb', linewidth=1.5)
plt.axvline(target_roc, color='r', linestyle='--', label='Target ROC', linewidth=1.2)
if not np.isnan(absolute_ceiling):
    plt.axhline(absolute_ceiling, color='g', linestyle='--', label='Absolute Ceiling', linewidth=1.2)
if not np.isnan(service_ceiling):
    plt.axhline(service_ceiling, color='m', linestyle='--', label='Service Ceiling', linewidth=1.2)
plt.ylabel('Altitude (m)')
plt.xlabel('Rate of Climb (m/s)')
plt.title('Rate of Climb vs Altitude')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Plot: Excess Power vs Velocity
plt.figure()
for i, h in enumerate(altitudes):
    valid_indices = excess_power[i, :] >= 0
    plt.plot(velocities[valid_indices], excess_power[i, valid_indices], label=f'Altitude: {h} m')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Excess Power (W)')
plt.title('Excess Power vs Velocity')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Plot: Time to Climb vs Altitude
if not np.isnan(absolute_ceiling) and not np.isnan(service_ceiling):
    time_to_climb = np.linspace(0, (absolute_ceiling / roc_values[0]) * \
                               np.log(absolute_ceiling / (absolute_ceiling - service_ceiling)), 100)
    altitudes_climb = absolute_ceiling * (1 - np.exp(-roc_values[0] * time_to_climb / absolute_ceiling))

    plt.figure()
    plt.plot(time_to_climb, altitudes_climb, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Time to Climb vs Altitude')
    plt.grid(True)
    plt.show()

# Plot: Climb Schedule
plt.figure()
for i, h in enumerate(altitudes):
    plt.plot(velocities, roc[i, :], label=f'h = {h} m')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Rate of Climb (m/s)')
plt.title('Climb Schedule')
plt.legend(loc='best')
plt.grid(True)
plt.show()
print(excess_power)
print(roc_values)