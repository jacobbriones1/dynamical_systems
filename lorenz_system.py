import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the Lorenz system
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Function to compute the trajectory
def compute_trajectory(sigma, rho, beta, initial_state, t_span, t_steps):
    sol = solve_ivp(
        lorenz, t_span, initial_state, args=(sigma, rho, beta), 
        t_eval=np.linspace(t_span[0], t_span[1], t_steps), 
        method='RK45'
    )
    return sol.y

# Initial parameters
initial_sigma = 10.0
initial_rho = 28.0
initial_beta = 8/3

initial_state = [1.0, 1.0, 1.0]
t_span = (0, 40)
t_steps = 10000

# Compute initial trajectory
trajectory = compute_trajectory(initial_sigma, initial_rho, initial_beta, initial_state, t_span, t_steps)

# Compute mirror trajectory for symmetry
mirror_trajectory = compute_trajectory(initial_sigma, initial_rho, initial_beta, [-initial_state[0], -initial_state[1], initial_state[2]], t_span, t_steps)

# Set up the plot with a larger figure size
fig = plt.figure(figsize=(16, 12))  # Increased size
ax = fig.add_subplot(111, projection='3d')

# Adjust subplot to make room for sliders and tooltips
plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.95)

# Set dark background for the figure
fig.patch.set_facecolor('#2E2E2E')  # Dark gray
ax.set_facecolor('#1E1E1E')        # Even darker for the axes

# Plot the trajectory lines
line, = ax.plot(
    trajectory[0], trajectory[1], trajectory[2], 
    lw=0.5, color='gray', alpha=0.5
)
mirror_line, = ax.plot(
    mirror_trajectory[0], mirror_trajectory[1], mirror_trajectory[2],
    lw=0.5, color='gray', alpha=0.5
)

# Initialize particles for both trajectories
particle1, = ax.plot([], [], [], 'o', color='cyan', markersize=6)
particle2, = ax.plot([], [], [], 'o', color='magenta', markersize=6)
mirror_particle1, = ax.plot([], [], [], 'o', color='yellow', markersize=6)
mirror_particle2, = ax.plot([], [], [], 'o', color='green', markersize=6)

# Initialize trails for both trajectories
trail_length = 100  # Number of previous points to show
trail1, = ax.plot([], [], [], lw=1, color='cyan', alpha=0.7)
trail2, = ax.plot([], [], [], lw=1, color='magenta', alpha=0.7)
mirror_trail1, = ax.plot([], [], [], lw=1, color='yellow', alpha=0.7)
mirror_trail2, = ax.plot([], [], [], lw=1, color='green', alpha=0.7)

# Highlight the z-axis
ax.plot([0, 0], [0, 0], [np.min(trajectory[2]), np.max(trajectory[2])], 
        color='white', lw=1, linestyle='--')

# Setting the axes properties
ax.set_xlim((np.min(trajectory[0]), np.max(trajectory[0])))
ax.set_ylim((np.min(trajectory[1]), np.max(trajectory[1])))
ax.set_zlim((np.min(trajectory[2]), np.max(trajectory[2])))

# Remove grid lines
ax.grid(False)

# Remove axis panes and spines for a cleaner look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Set axis lines to white
ax.xaxis.line.set_color('white')
ax.yaxis.line.set_color('white')
ax.zaxis.line.set_color('white')

# Set labels with lighter colors
ax.set_xlabel("X Axis", color='white', fontsize=12)
ax.set_ylabel("Y Axis", color='white', fontsize=12)
ax.set_zlabel("Z Axis", color='white', fontsize=12)
ax.set_title("Lorenz System Simulation with Dynamic Motion", color='white', fontsize=15, pad=20)

# Set tick labels to white
ax.tick_params(colors='white', which='both')

# Slider axes positions
ax_sigma = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor='#4E4E4E')  # Darker slider background
ax_rho = plt.axes([0.1, 0.18, 0.8, 0.03], facecolor='#4E4E4E')
ax_beta = plt.axes([0.1, 0.11, 0.8, 0.03], facecolor='#4E4E4E')

# Create sliders without invalid keyword arguments
slider_sigma = Slider(
    ax_sigma, 'σ (Sigma)', 0.0, 30.0, valinit=initial_sigma, 
    color='#AAAAAA'
)
slider_rho = Slider(
    ax_rho, 'ρ (Rho)', 0.0, 50.0, valinit=initial_rho, 
    color='#AAAAAA'
)
slider_beta = Slider(
    ax_beta, 'β (Beta)', 0.0, 10.0, valinit=initial_beta, 
    color='#AAAAAA'
)

# Adjust slider label font sizes
slider_sigma.label.set_fontsize(10)
slider_rho.label.set_fontsize(10)
slider_beta.label.set_fontsize(10)

# Create text annotations for tooltips
tooltip_sigma = plt.text(0.5, 0.3, f'σ: {initial_sigma:.2f}', 
                         transform=fig.transFigure, color='white', fontsize=10, ha='center')
tooltip_rho = plt.text(0.5, 0.23, f'ρ: {initial_rho:.2f}', 
                       transform=fig.transFigure, color='white', fontsize=10, ha='center')
tooltip_beta = plt.text(0.5, 0.16, f'β: {initial_beta:.2f}', 
                        transform=fig.transFigure, color='white', fontsize=10, ha='center')

# Animation variables
current_index = 0
max_index = t_steps
num_particles = 2  # Number of particles to animate per trajectory
particle_indices = [0, int(t_steps / 2)]  # Starting points for particles

# Initialize trail data
trail1_x, trail1_y, trail1_z = [], [], []
trail2_x, trail2_y, trail2_z = [], [], []
mirror_trail1_x, mirror_trail1_y, mirror_trail1_z = [], [], []
mirror_trail2_x, mirror_trail2_y, mirror_trail2_z = [], [], []

# Update trajectory and reset animation
def update_trajectory(sigma, rho, beta):
    global trajectory, mirror_trajectory, max_index, current_index, particle_indices
    # Recompute trajectory
    trajectory = compute_trajectory(sigma, rho, beta, initial_state, t_span, t_steps)
    mirror_initial_state = [-initial_state[0], -initial_state[1], initial_state[2]]
    mirror_trajectory = compute_trajectory(sigma, rho, beta, mirror_initial_state, t_span, t_steps)
    
    # Update trajectory lines
    line.set_data(trajectory[0], trajectory[1])
    line.set_3d_properties(trajectory[2])
    mirror_line.set_data(mirror_trajectory[0], mirror_trajectory[1])
    mirror_line.set_3d_properties(mirror_trajectory[2])
    
    # Reset animation variables
    current_index = 0
    particle_indices = [0, int(t_steps / 2)]
    
    # Update axes limits
    ax.set_xlim((np.min(trajectory[0]), np.max(trajectory[0])))
    ax.set_ylim((np.min(trajectory[1]), np.max(trajectory[1])))
    ax.set_zlim((np.min(trajectory[2]), np.max(trajectory[2])))
    
    # Clear trails
    trail1_x.clear()
    trail1_y.clear()
    trail1_z.clear()
    trail2_x.clear()
    trail2_y.clear()
    trail2_z.clear()
    mirror_trail1_x.clear()
    mirror_trail1_y.clear()
    mirror_trail1_z.clear()
    mirror_trail2_x.clear()
    mirror_trail2_y.clear()
    mirror_trail2_z.clear()
    
    # Reset trail lines
    trail1.set_data([], [])
    trail1.set_3d_properties([])
    trail2.set_data([], [])
    trail2.set_3d_properties([])
    mirror_trail1.set_data([], [])
    mirror_trail1.set_3d_properties([])
    mirror_trail2.set_data([], [])
    mirror_trail2.set_3d_properties([])

# Update function for sliders
def slider_update(val):
    sigma = slider_sigma.val
    rho = slider_rho.val
    beta = slider_beta.val
    update_trajectory(sigma, rho, beta)
    # Update tooltip texts
    tooltip_sigma.set_text(f'σ: {sigma:.2f}')
    tooltip_rho.set_text(f'ρ: {rho:.2f}')
    tooltip_beta.set_text(f'β: {beta:.2f}')

# Connect sliders to update function
slider_sigma.on_changed(slider_update)
slider_rho.on_changed(slider_update)
slider_beta.on_changed(slider_update)

# Animation function
def animate(frame):
    global current_index, particle_indices
    if current_index >= max_index:
        current_index = 0  # Reset to start

    # Update particles for the main trajectory
    for i in range(num_particles):
        idx = (particle_indices[i] + 1) % max_index
        particle_indices[i] = idx
        x = trajectory[0][idx]
        y = trajectory[1][idx]
        z = trajectory[2][idx]
        if i == 0:
            # Update particle1 position
            particle1.set_data([x], [y])  # Wrap x and y in lists
            particle1.set_3d_properties([z])  # Wrap z in a list
            # Update trail1
            trail1_x.append(x)
            trail1_y.append(y)
            trail1_z.append(z)
            if len(trail1_x) > trail_length:
                trail1_x.pop(0)
                trail1_y.pop(0)
                trail1_z.pop(0)
            trail1.set_data(trail1_x, trail1_y)
            trail1.set_3d_properties(trail1_z)
        elif i == 1:
            # Update particle2 position
            particle2.set_data([x], [y])
            particle2.set_3d_properties([z])
            # Update trail2
            trail2_x.append(x)
            trail2_y.append(y)
            trail2_z.append(z)
            if len(trail2_x) > trail_length:
                trail2_x.pop(0)
                trail2_y.pop(0)
                trail2_z.pop(0)
            trail2.set_data(trail2_x, trail2_y)
            trail2.set_3d_properties(trail2_z)
    
    # Update particles for the mirror trajectory
    for i in range(num_particles):
        idx = (particle_indices[i] + 1) % max_index
        mirror_particle_idx = (idx + t_steps//2) % max_index  # Offset for mirror
        x = mirror_trajectory[0][idx]
        y = mirror_trajectory[1][idx]
        z = mirror_trajectory[2][idx]
        if i == 0:
            # Update mirror_particle1 position
            mirror_particle1.set_data([x], [y])
            mirror_particle1.set_3d_properties([z])
            # Update mirror_trail1
            mirror_trail1_x.append(x)
            mirror_trail1_y.append(y)
            mirror_trail1_z.append(z)
            if len(mirror_trail1_x) > trail_length:
                mirror_trail1_x.pop(0)
                mirror_trail1_y.pop(0)
                mirror_trail1_z.pop(0)
            mirror_trail1.set_data(mirror_trail1_x, mirror_trail1_y)
            mirror_trail1.set_3d_properties(mirror_trail1_z)
        elif i == 1:
            # Update mirror_particle2 position
            mirror_particle2.set_data([x], [y])
            mirror_particle2.set_3d_properties([z])
            # Update mirror_trail2
            mirror_trail2_x.append(x)
            mirror_trail2_y.append(y)
            mirror_trail2_z.append(z)
            if len(mirror_trail2_x) > trail_length:
                mirror_trail2_x.pop(0)
                mirror_trail2_y.pop(0)
                mirror_trail2_z.pop(0)
            mirror_trail2.set_data(mirror_trail2_x, mirror_trail2_y)
            mirror_trail2.set_3d_properties(mirror_trail2_z)
    
    current_index += 1
    return particle1, particle2, mirror_particle1, mirror_particle2, trail1, trail2, mirror_trail1, mirror_trail2

# Create animation
ani = FuncAnimation(fig, animate, frames=max_index, interval=20, blit=True)

plt.show()
