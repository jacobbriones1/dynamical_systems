import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

# Define the ODE system for the damped harmonic oscillator
def damped_oscillator(t, y, omega_0, beta):
    x, v = y
    dxdt = v
    dvdt = -2 * beta * v - omega_0**2 * x
    return [dxdt, dvdt]

# Solve the ODE with given parameters
def solve_damped_oscillator(omega_0, beta, x0, v0):
    t_span = [0, 20]
    t_eval = np.linspace(0, 20, 500)
    sol = solve_ivp(damped_oscillator, t_span, [x0, v0], args=(omega_0, beta), t_eval=t_eval)
    return sol.t, sol.y[0], sol.y[1]  # Return time, position, velocity

# Initialize parameters
initial_omega_0 = 1.0
initial_beta = 0.5
initial_x0 = 1.0
initial_v0 = 0.0

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the initial data
num_trajectories = 15
colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
lines1 = [ax1.plot([], [], lw=2, color=colors[i])[0] for i in range(num_trajectories)]
points1 = [ax1.plot([], [], 'ko',markersize=5)[0] for _ in range(num_trajectories)]

lines2 = [ax2.plot([], [], lw=2, color=colors[i])[0] for i in range(num_trajectories)]
points2 = [ax2.plot([], [], 'ko',markersize=5)[0] for _ in range(num_trajectories)]

ax1.set_xlim(0, 20)
ax1.set_ylim(-5, 5)
ax1.set_xlabel('Time $t$')
ax1.set_ylabel('Position $x(t)$')
ax1.set_title('Position vs Time')
ax1.grid(True)

ax2.set_xlim(-6,6)
ax2.set_ylim(-6, 6)
ax2.set_xlabel('Position $x$')
ax2.set_ylabel('Velocity $v$')
ax2.set_title('Phase Diagram')
ax2.grid(True)

# Initialize the plot lines
def init():
    for line in lines1:
        line.set_data([], [])
    for point in points1:
        point.set_data([], [])
    for line in lines2:
        line.set_data([], [])
    for point in points2:
        point.set_data([], [])
    return lines1 + points1 + lines2 + points2

# Update function for animation
def animate(i):
    # Update position vs. time plot
    for j in range(num_trajectories):
        x0 = slider_x0.val + ((-1)**j) * 0.5  # Different initial conditions
        v0 = slider_v0.val +((-1)**j) 
        t, x, v = solve_damped_oscillator(omega_0, beta, x0, v0)
        lines1[j].set_data(t[:i], x[:i])
        points1[j].set_data([t[i]], [x[i]])
        
        # Update phase diagram
        lines2[j].set_data(x[:i], v[:i])
        points2[j].set_data([x[i]], [v[i]])
    
    # Update parameter labels
    return lines1 + points1 + lines2 + points2

# Create sliders for parameters
ax_omega_0 = plt.axes([0.1, 0.01, 0.7, 0.03], facecolor='lightgoldenrodyellow')
ax_beta = plt.axes([0.2, 0.06, 0.7, 0.03], facecolor='lightgoldenrodyellow')
ax_x0 = plt.axes([0.2, 0.11, 0.7, 0.03], facecolor='lightgoldenrodyellow')
ax_v0 = plt.axes([0.2, 0.16, 0.7, 0.03], facecolor='lightgoldenrodyellow')

slider_omega_0 = Slider(ax_omega_0, 'ω₀ (Frequency)', 0.1, 5.0, valinit=initial_omega_0)
slider_beta = Slider(ax_beta, 'β (Damping Coefficient)', 0.0, 6.0, valinit=initial_beta)
slider_x0 = Slider(ax_x0, 'x₀ (Initial Position)', -4.0, 4.0, valinit=initial_x0)
slider_v0 = Slider(ax_v0, 'v₀ (Initial Velocity)', -4.0, 4.0, valinit=initial_v0)

# Function to update the solution when sliders change
def update(val):
    global omega_0, beta, x0, v0
    omega_0 = slider_omega_0.val
    beta = slider_beta.val
    x0 = slider_x0.val
    v0 = slider_v0.val
    ax1.legend()
    ax2.legend()
    ani.event_source.stop()
    ani.event_source.start()

# Link the sliders to the update function
slider_omega_0.on_changed(update)
slider_beta.on_changed(update)
slider_x0.on_changed(update)
slider_v0.on_changed(update)

# Initialize parameters
omega_0 = slider_omega_0.val
beta = slider_beta.val
x0 = slider_x0.val
v0 = slider_v0.val

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=500, init_func=init, interval=30, blit=True)

# Adjust the layout to fit the sliders below the plot
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.25)

plt.show()
