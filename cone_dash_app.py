import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # Gravitational acceleration (m/s^2)
MIN_RADIUS = 0.1  # Minimum radius to prevent the particle from reaching the cone's apex
MIN_L = 0.01  # Small minimum angular momentum to prevent degenerate case

# Differential equations for motion on the cone surface
def equations(t, y, L, alpha, m):
    r, r_dot, phi = y
    r = max(r, MIN_RADIUS)  # Enforce minimum radius to stay on the cone surface
    r_double_dot = (L**2 / (m**2 * r**3 * np.sin(alpha)**2)) - g * np.cos(alpha)
    phi_dot = L / (m * r**2 * np.sin(alpha)**2)  # Angular velocity (constant in this ideal model)
    return [r_dot, r_double_dot, phi_dot]

# Real-time simulation function with fixed small time step
def simulate_step(r0, v_r0, phi0, L, alpha, m, t_step=0.009):
    t_eval = [0, t_step]  # Small, fixed time step for smooth real-time tracing
    sol = solve_ivp(equations, [0, t_step], [r0, v_r0, phi0], args=(L, alpha, m), t_eval=t_eval, atol=1e-8, rtol=1e-8)
    r = max(sol.y[0][-1], MIN_RADIUS)  # Prevent particle from going below minimum radius
    phi = sol.y[2][-1]
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = r / np.tan(alpha)  # Constrain z to be on the cone
    return x, y, z, r, sol.y[1][-1], phi  # Return updated position and velocities

# Create figure and initialize plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.3)

# Initial parameters
r0, v_r0, phi0, L, alpha, m = 1, 0, 0, 1, np.pi / 6, 1
x, y, z, r, v_r, phi = r0, 0, r0 / np.tan(alpha), r0, v_r0, phi0  # Initial conditions
path, = ax.plot([x], [y], [z], 'b-', lw=1)  # Real-time trajectory
mass_marker, = ax.plot([x], [y], [z], 'o', color='black', markersize=5)  # Particle marker

# Initial cone surface
cone_height = 5  # Fixed height for visualization
def update_cone_surface(alpha):
    global surface
    r_cone = np.linspace(MIN_RADIUS, cone_height * np.tan(alpha), 50)
    phi_cone = np.linspace(0, 2 * np.pi, 50)
    R_cone, PHI_cone = np.meshgrid(r_cone, phi_cone)
    X_cone = R_cone * np.cos(PHI_cone)
    Y_cone = R_cone * np.sin(PHI_cone)
    Z_cone = R_cone / np.tan(alpha)
    # Remove old surface if it exists and plot a new one
    if 'surface' in globals():
        surface.remove()
    surface = ax.plot_surface(X_cone, Y_cone, Z_cone, color='yellow', alpha=0.3)

update_cone_surface(alpha)  # Initial cone plot

# Set axis limits
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 5)

# Remove the bounding box for a cleaner look
ax.set_axis_off()

# Sliders for initial conditions and parameters
slider_r0 = Slider(plt.axes([0.2, 0.2, 0.65, 0.03]), 'Initial Radius (r0)', 0.1, 2, valinit=r0)
slider_v_r0 = Slider(plt.axes([0.2, 0.15, 0.65, 0.03]), 'Initial Velocity (v_r0)', -2, 2, valinit=v_r0)
slider_phi0 = Slider(plt.axes([0.2, 0.1, 0.65, 0.03]), 'Initial Angle (phi0)', 0, 2 * np.pi, valinit=phi0)
slider_L = Slider(plt.axes([0.2, 0.05, 0.65, 0.03]), 'Angular Momentum (L)', MIN_L, 5, valinit=L)
slider_alpha = Slider(plt.axes([0.2, 0, 0.65, 0.03]), 'Angle Alpha (alpha)', 0.2, np.pi / 3, valinit=alpha)

# Update function for slider changes
def slider_update(val):
    global r0, v_r0, phi0, L, alpha, r, v_r, phi, x, y, z
    r0 = slider_r0.val
    v_r0 = slider_v_r0.val
    phi0 = slider_phi0.val
    L = max(slider_L.val, MIN_L)
    alpha = slider_alpha.val
    r, v_r, phi = r0, v_r0, phi0
    x, y, z = r0 * np.cos(phi0), r0 * np.sin(phi0), r0 / np.tan(alpha)
    path.set_data([x], [y])
    path.set_3d_properties([z])
    mass_marker.set_data([x], [y])
    mass_marker.set_3d_properties([z])
    # Update the cone surface when alpha is changed
    update_cone_surface(alpha)

# Connect sliders to update function
slider_r0.on_changed(slider_update)
slider_v_r0.on_changed(slider_update)
slider_phi0.on_changed(slider_update)
slider_L.on_changed(slider_update)
slider_alpha.on_changed(slider_update)

# Animation function to trace the trajectory in real-time
def animate(i):
    global x, y, z, r, v_r, phi
    x, y, z, r, v_r, phi = simulate_step(r, v_r, phi, L, alpha, m)  # Update position
    
    # Get existing path data and extend it with the new point
    x_data, y_data, z_data = path.get_data_3d()
    path.set_data_3d(np.append(x_data, x), np.append(y_data, y), np.append(z_data, z))
    
    # Update particle marker position
    mass_marker.set_data([x], [y])
    mass_marker.set_3d_properties([z])
    return path, mass_marker

# Run animation
ani = FuncAnimation(fig, animate, frames=200, interval=8, blit=True, repeat=True)
plt.show()
