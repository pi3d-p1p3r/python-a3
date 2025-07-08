import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# Problem i)
# Define the first ODE: y' = t*e^3t - 2y; y(0) = 0
def ode1(y, t):
    return t * np.exp(3*t) - 2*y

# Exact solution for problem i)
def exact_sol1(t):
    return (1/5) * t * np.exp(3*t) - (1/25) * np.exp(3*t) + (1/25) * np.exp(-2*t)

# Problem ii)
# Define the second ODE: y' = 1 + (t - y)^2; y(2) = 1
def ode2(y, t):
    return 1 + (t - y)**2

# Since solve_ivp expects the function with arguments (t, y), we need a different version
def ode2_ivp(t, y):
    return 1 + (t - y[0])**2

# Exact solution for problem ii)
def exact_sol2(t):
    return t + 1/(1-t)

# Solve problem i) using odeint and solve_ivp
t1 = np.linspace(0, 1, 100)  # Time points from 0 to 1
y1_odeint = odeint(ode1, 0, t1)  # Solve using odeint

# For solve_ivp, we need to reshape the result
sol1_ivp = solve_ivp(lambda t, y: t * np.exp(3*t) - 2*y, [0, 1], [0], t_eval=t1)
y1_ivp = sol1_ivp.y.T

# Calculate exact solution
y1_exact = exact_sol1(t1)

# Solve problem ii) using odeint and solve_ivp
t2 = np.linspace(2, 3, 100)  # Time points from 2 to 3
y2_odeint = odeint(ode2, 1, t2)  # Solve using odeint

# For solve_ivp, we need different handling due to the initial point
sol2_ivp = solve_ivp(ode2_ivp, [2, 3], [1], t_eval=t2)
y2_ivp = sol2_ivp.y.T

# Calculate exact solution
y2_exact = exact_sol2(t2)

# Calculate errors for problem i)
error1_odeint = np.abs(y1_odeint.flatten() - y1_exact)
error1_ivp = np.abs(y1_ivp.flatten() - y1_exact)

# Calculate errors for problem ii)
error2_odeint = np.abs(y2_odeint.flatten() - y2_exact)
error2_ivp = np.abs(y2_ivp.flatten() - y2_exact)

# Create plots for problem i)
plt.figure(figsize=(15, 10))

# Plot solutions for problem i)
plt.subplot(2, 2, 1)
plt.plot(t1, y1_odeint, 'b-', label='odeint')
plt.plot(t1, y1_ivp, 'g--', label='solve_ivp')
plt.plot(t1, y1_exact, 'r:', label='Exact')
plt.title("Problem i) Solutions")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)

# Plot errors for problem i)
plt.subplot(2, 2, 2)
plt.semilogy(t1, error1_odeint, 'b-', label='odeint error')
plt.semilogy(t1, error1_ivp, 'g--', label='solve_ivp error')
plt.title("Problem i) Errors")
plt.xlabel('t')
plt.ylabel('Absolute Error (log scale)')
plt.legend()
plt.grid(True)

# Plot solutions for problem ii)
plt.subplot(2, 2, 3)
plt.plot(t2, y2_odeint, 'b-', label='odeint')
plt.plot(t2, y2_ivp, 'g--', label='solve_ivp')
plt.plot(t2, y2_exact, 'r:', label='Exact')
plt.title("Problem ii) Solutions")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)

# Plot errors for problem ii)
plt.subplot(2, 2, 4)
plt.semilogy(t2, error2_odeint, 'b-', label='odeint error')
plt.semilogy(t2, error2_ivp, 'g--', label='solve_ivp error')
plt.title("Problem ii) Errors")
plt.xlabel('t')
plt.ylabel('Absolute Error (log scale)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print maximum errors
print(f"Problem i) Maximum Errors:")
print(f"  odeint: {np.max(error1_odeint):.2e}")
print(f"  solve_ivp: {np.max(error1_ivp):.2e}")
print(f"\nProblem ii) Maximum Errors:")
print(f"  odeint: {np.max(error2_odeint):.2e}")
print(f"  solve_ivp: {np.max(error2_ivp):.2e}")
