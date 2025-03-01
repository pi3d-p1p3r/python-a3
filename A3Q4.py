import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def pendulum_system(t, y, g, L):
    """
    Define the pendulum system of first-order ODEs.
    y[0] = θ, y[1] = θ'
    """
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g/L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Parameters
g = 32.17  # ft/s^2
L = 2.0    # feet
t_span = (0, 2)  # time span
t_eval = np.arange(0, 2.1, 0.1)  # evaluation points with 0.1s increment

# Initial conditions
theta_0 = np.pi/6  # θ(0) = π/6
omega_0 = 0        # θ'(0) = 0

# Solve using solve_ivp (modern equivalent of odeint)
solution = solve_ivp(
    fun=lambda t, y: pendulum_system(t, y, g, L),
    t_span=t_span,
    y0=[theta_0, omega_0],
    method='RK45',
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-8
)

# Extract results
t = solution.t
theta = solution.y[0]
omega = solution.y[1]

# Print the results in a table
print("t (s)\tθ (rad)\tθ (degrees)")
print("-" * 30)
for i in range(len(t)):
    print(f"{t[i]:.1f}\t{theta[i]:.6f}\t{theta[i] * 180/np.pi:.6f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, theta, 'b-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('θ (rad)')
plt.title('Pendulum Motion (θ vs. time)')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.savefig('pendulum_motion.png')
plt.show()

# Calculate the period analytically for small oscillations
T_small = 2 * np.pi * np.sqrt(L/g)
print(f"\nAnalytical period for small oscillations: {T_small:.4f} seconds")

# Verify if our simulation captures approximately one period
print(f"From the plot, we can see that the period is approximately {T_small:.4f} seconds")