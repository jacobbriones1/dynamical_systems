import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as colors

# ---------------------------- Constants ----------------------------
g = 9.81  # acceleration due to gravity (m/s^2)
l = 1.0   # length of pendulum arms (m)
m = 1.0   # mass of pendulum bobs (kg)
dt = 0.02 # time step for simulation (seconds)
max_phase_points = 500  # Maximum number of points in phase space plots

# ----------------------- Equations of Motion -----------------------
def double_pendulum_eq(t, y):
    """
    Computes the derivatives for the double pendulum system.

    Parameters:
    - t: Current time (unused as the system is autonomous).
    - y: Current state vector [theta1, z1, theta2, z2].

    Returns:
    - dydt: Derivatives [d(theta1)/dt, d(z1)/dt, d(theta2)/dt, d(z2)/dt].
    """
    theta1, z1, theta2, z2 = y
    delta = theta1 - theta2

    den1 = (2 - np.cos(2 * delta))
    den2 = den1  # Since masses and lengths are equal

    # Prevent division by zero
    den1 = den1 if den1 != 0 else 1e-6

    theta1_dot = z1
    theta2_dot = z2

    theta1_ddot = (-2 * g * np.sin(theta1) - np.sin(delta) * (z2**2 + z1**2 * np.cos(delta))) / den1
    theta2_ddot = (2 * np.sin(delta) * (2 * z1**2 + 2 * g * np.cos(theta1) + z2**2 * np.cos(delta))) / den2

    return [theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]

# --------------------------- Simulator Class ---------------------------
class PendulumSimulator:
    """
    Simulates the double pendulum dynamics using SciPy's ODE integrator.

    Attributes:
    - state: Current state vector [theta1, z1, theta2, z2].
    - solver: ODE integrator instance.
    """

    def __init__(self, y0, t0=0):
        """
        Initializes the simulator with initial conditions.

        Parameters:
        - y0: Initial state vector [theta1, z1, theta2, z2].
        - t0: Initial time.
        """
        self.state = y0
        self.solver = ode(double_pendulum_eq).set_integrator('dopri5')
        self.solver.set_initial_value(self.state, t0)

    def step(self, dt):
        """
        Advances the simulation by a time step dt.

        Parameters:
        - dt: Time step to advance (seconds).

        Returns:
        - new_state: Updated state vector after stepping.
        """
        if self.solver.successful():
            self.solver.integrate(self.solver.t + dt)
            self.state = self.solver.y
            return self.state
        else:
            raise RuntimeError("ODE solver failed.")

    def reset(self, y0, t0=0):
        """
        Resets the simulation with new initial conditions.

        Parameters:
        - y0: New state vector [theta1, z1, theta2, z2].
        - t0: Reset time.
        """
        self.state = y0
        self.solver = ode(double_pendulum_eq).set_integrator('dopri5')
        self.solver.set_initial_value(self.state, t0)

# -------------------------- Initialization --------------------------
# Initial Conditions: [theta1, z1, theta2, z2]
# Example 1: Regular Motion
# y0 = [np.pi / 2, 0, np.pi / 2, 0]

# Example 2: Chaotic Motion
y0 = [np.pi / 2, 0, np.pi / 4, 0]

# Initialize the simulator
simulator = PendulumSimulator(y0)

# Time Span for Simulation (Real-Time)
t_start = 0
t_end = 60  # seconds
num_frames = int(t_end / dt)

# Data Storage for Phase Space and Trajectories
theta1_history = []
z1_history = []
theta2_history = []
z2_history = []
x1_history = []
y1_history = []
x2_history = []
y2_history = []

# Compute Initial Positions
theta1_initial, z1_initial, theta2_initial, z2_initial = y0
x1_initial = l * np.sin(theta1_initial)
y1_initial = -l * np.cos(theta1_initial)
x2_initial = x1_initial + l * np.sin(theta2_initial)
y2_initial = y1_initial - l * np.cos(theta2_initial)

# Append initial positions to histories
theta1_history.append(theta1_initial)
z1_history.append(z1_initial)
theta2_history.append(theta2_initial)
z2_history.append(z2_initial)
x1_history.append(x1_initial)
y1_history.append(y1_initial)
x2_history.append(x2_initial)
y2_history.append(y2_initial)

# --------------------------- Plot Setup -----------------------------
# Create Figure and Axes
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.4)

# ---------------- Phase Space Projection: theta1 vs z1 ----------------
ax1 = axs[0]
ax1.set_xlabel(r'$\theta_1$ (rad)')
ax1.set_ylabel(r'$\dot{\theta}_1$ (rad/s)')
ax1.set_title(r'Phase Space: $\theta_1$ vs. $\dot{\theta}_1$')
ax1.grid(True)

# Initialize scatter plot for phase space with colormap
cmap1 = plt.get_cmap('viridis')  # Updated to use plt.get_cmap
norm = colors.Normalize(vmin=0, vmax=num_frames)
scatter1 = ax1.scatter([], [], c=[], cmap=cmap1, norm=norm, s=10)
point1, = ax1.plot([], [], 'ko')  # Current state point in black

# Initialize line trace for phase space with LineCollection
line1_collection = LineCollection([], cmap='viridis', norm=norm, alpha=0.6)
ax1.add_collection(line1_collection)

# ---------------- Phase Space Projection: theta2 vs z2 ----------------
ax2 = axs[1]
ax2.set_xlabel(r'$\theta_2$ (rad)')
ax2.set_ylabel(r'$\dot{\theta}_2$ (rad/s)')
ax2.set_title(r'Phase Space: $\theta_2$ vs. $\dot{\theta}_2$')
ax2.grid(True)

# Initialize scatter plot for phase space with colormap
cmap2 = plt.get_cmap('plasma')  # Using a different colormap for distinction
scatter2 = ax2.scatter([], [], c=[], cmap=cmap2, norm=norm, s=10)
point2, = ax2.plot([], [], 'ko')  # Current state point in black

# Initialize line trace for phase space with LineCollection
line2_collection = LineCollection([], cmap='plasma', norm=norm, alpha=0.6)
ax2.add_collection(line2_collection)

# -------------- Physical Motion of the Double Pendulum --------------
ax3 = axs[2]
ax3.set_xlim(-2.2 * l, 2.2 * l)
ax3.set_ylim(-2.2 * l, 2.2 * l)
ax3.set_xlabel('X Position (m)')
ax3.set_ylabel('Y Position (m)')
ax3.set_title('Double Pendulum Physical Motion')
ax3.grid(True)
ax3.set_aspect('equal')  # Ensure the subplot is square

# Initialize Lines and Bobs
line_pendulum, = ax3.plot([], [], lw=2, color='black')

# Trajectory lines for tracing with color gradients using LineCollection
trajectory1_collection = LineCollection([], cmap='viridis', norm=norm, alpha=0.6)
trajectory2_collection = LineCollection([], cmap='plasma', norm=norm, alpha=0.6)
ax3.add_collection(trajectory1_collection)
ax3.add_collection(trajectory2_collection)

# Initialize Bobs
bob1 = Circle((x1_initial, y1_initial), 0.05, fc='blue', zorder=5, picker=True)
bob2 = Circle((x2_initial, y2_initial), 0.05, fc='red', zorder=5, picker=True)
ax3.add_patch(bob1)
ax3.add_patch(bob2)

# --------------------- Interactive Dragging ---------------------
# Variables to track dragging state
dragging = False
dragged_bob = None

def on_press(event):
    """
    Handles mouse button press events for dragging pendulum masses.
    """
    global dragging, dragged_bob
    if event.inaxes != ax3:
        return
    contains1, _ = bob1.contains(event)
    contains2, _ = bob2.contains(event)
    if contains1:
        dragging = True
        dragged_bob = bob1
    elif contains2:
        dragging = True
        dragged_bob = bob2

def on_motion(event):
    """
    Handles mouse motion events for dragging pendulum masses.
    """
    global dragging
    if not dragging or dragged_bob is None:
        return
    if event.inaxes != ax3:
        return
    # Update the position of the dragged bob
    x, y = event.xdata, event.ydata
    dragged_bob.center = (x, y)
    # Update the connected lines and angles
    if dragged_bob == bob1:
        # Recompute theta1 based on new position
        new_theta1 = np.arctan2(x, -y)
        # Update simulator's initial state
        new_y0 = [new_theta1, 0, simulator.state[2], 0]
    elif dragged_bob == bob2:
        # Recompute theta2 based on new position relative to bob1
        x1_new, y1_new = bob1.center  # Updated bob1 position
        new_theta2 = np.arctan2(x - x1_new, -(y - y1_new))
        # Update simulator's initial state
        new_y0 = [simulator.state[0], 0, new_theta2, 0]
    # Reset the simulation with new initial conditions
    simulator.reset(new_y0, t0=simulator.solver.t)
    # Clear phase space and trajectory histories
    theta1_history.clear()
    z1_history.clear()
    theta2_history.clear()
    z2_history.clear()
    x1_history.clear()
    y1_history.clear()
    x2_history.clear()
    y2_history.clear()
    # Append new initial positions
    theta1_history.append(new_y0[0])
    z1_history.append(new_y0[1])
    theta2_history.append(new_y0[2])
    z2_history.append(new_y0[3])
    x1_initial_new = l * np.sin(new_y0[0])
    y1_initial_new = -l * np.cos(new_y0[0])
    x2_initial_new = x1_initial_new + l * np.sin(new_y0[2])
    y2_initial_new = y1_initial_new - l * np.cos(new_y0[2])
    x1_history.append(x1_initial_new)
    y1_history.append(y1_initial_new)
    x2_history.append(x2_initial_new)
    y2_history.append(y2_initial_new)
    # Reset trajectory lines
    trajectory1_collection.set_segments([])
    trajectory2_collection.set_segments([])
    # Reset phase space scatter plots
    scatter1.set_offsets(np.empty((0, 2)))  # Corrected to 2D empty array
    scatter2.set_offsets(np.empty((0, 2)))  # Corrected to 2D empty array
    # Reset line traces
    line1_collection.set_segments([])
    line2_collection.set_segments([])

def on_release(event):
    """
    Handles mouse button release events to stop dragging.
    """
    global dragging, dragged_bob
    dragging = False
    dragged_bob = None

# Connect event handlers
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# --------------------- Animation Function ---------------------
def init_anim():
    """
    Initializes the animation by clearing all plot data.
    """
    scatter1.set_offsets(np.empty((0, 2)))  # Corrected to 2D empty array
    scatter2.set_offsets(np.empty((0, 2)))  # Corrected to 2D empty array
    point1.set_data([], [])
    point2.set_data([], [])
    line_pendulum.set_data([], [])
    trajectory1_collection.set_segments([])
    trajectory2_collection.set_segments([])
    line1_collection.set_segments([])
    line2_collection.set_segments([])
    # Set initial positions of bobs
    bob1.center = (x1_initial, y1_initial)
    bob2.center = (x2_initial, y2_initial)
    return scatter1, scatter2, point1, point2, line_pendulum, trajectory1_collection, trajectory2_collection, line1_collection, line2_collection, bob1, bob2

def animate(frame):
    """
    Updates the animation for each frame.

    Parameters:
    - frame: Frame number (unused).

    Returns:
    - Updated plot elements.
    """
    # Advance the simulation by dt
    try:
        state = simulator.step(dt)
    except RuntimeError:
        # Stop the animation if the solver fails
        anim.event_source.stop()
        return scatter1, scatter2, point1, point2, line_pendulum, trajectory1_collection, trajectory2_collection, line1_collection, line2_collection, bob1, bob2

    theta1_current, z1_current, theta2_current, z2_current = state

    # Append current state to histories
    theta1_history.append(theta1_current)
    z1_history.append(z1_current)
    theta2_history.append(theta2_current)
    z2_history.append(z2_current)

    # Compute current positions
    x1_current = l * np.sin(theta1_current)
    y1_current = -l * np.cos(theta1_current)
    x2_current = x1_current + l * np.sin(theta2_current)
    y2_current = y1_current - l * np.cos(theta2_current)

    x1_history.append(x1_current)
    y1_history.append(y1_current)
    x2_history.append(x2_current)
    y2_history.append(y2_current)

    # Update Phase Space Projections
    # Limit the number of points to max_phase_points
    if len(theta1_history) > max_phase_points:
        theta1_display = theta1_history[-max_phase_points:]
        z1_display = z1_history[-max_phase_points:]
        theta2_display = theta2_history[-max_phase_points:]
        z2_display = z2_history[-max_phase_points:]
    else:
        theta1_display = theta1_history
        z1_display = z1_history
        theta2_display = theta2_history
        z2_display = z2_history

    # Assign colors based on the sequence
    # Normalize the sequence for color mapping
    current_len = len(theta1_display)
    colors_seq = np.linspace(0, 1, current_len)
    scatter1.set_offsets(np.c_[theta1_display, z1_display])
    scatter1.set_array(colors_seq)
    scatter1.set_clim(0, 1)  # Update color limits

    scatter2.set_offsets(np.c_[theta2_display, z2_display])
    scatter2.set_array(colors_seq)
    scatter2.set_clim(0, 1)  # Update color limits

    # Update Current State Points
    point1.set_data([theta1_current], [z1_current])
    point2.set_data([theta2_current], [z2_current])

    # Dynamic Axes Ranges for Phase Space Plots
    # Update ax1 limits based on theta1 and z1 data
    ax1.set_xlim(min(theta1_display) - 0.5, max(theta1_display) + 0.5)
    ax1.set_ylim(min(z1_display) - 1, max(z1_display) + 1)

    # Update ax2 limits based on theta2 and z2 data
    ax2.set_xlim(min(theta2_display) - 0.5, max(theta2_display) + 0.5)
    ax2.set_ylim(min(z2_display) - 1, max(z2_display) + 1)

    # Update Physical Pendulum Positions
    line_pendulum.set_data([0, x1_current, x2_current], [0, y1_current, y2_current])
    bob1.center = (x1_current, y1_current)
    bob2.center = (x2_current, y2_current)

    # Update Trajectory Traces with LineCollection
    # Create segments for trajectory1 and trajectory2
    if len(x1_history) > 1:
        segments1 = np.array([ [(x1_history[i], y1_history[i]),
                                 (x1_history[i+1], y1_history[i+1])]
                             for i in range(len(x1_history)-1) ])
        segments2 = np.array([ [(x2_history[i], y2_history[i]),
                                 (x2_history[i+1], y2_history[i+1])]
                             for i in range(len(x2_history)-1) ])
        trajectory1_collection.set_segments(segments1)
        trajectory2_collection.set_segments(segments2)
        # Assign colors based on sequence
        trajectory1_collection.set_array(np.linspace(0, 1, len(segments1)))
        trajectory2_collection.set_array(np.linspace(0, 1, len(segments2)))
    else:
        trajectory1_collection.set_segments([])
        trajectory2_collection.set_segments([])

    # Update Phase Space Line Traces with LineCollection
    if len(theta1_display) > 1:
        segments_phase1 = np.array([ [(theta1_display[i], z1_display[i]),
                                      (theta1_display[i+1], z1_display[i+1])]
                                  for i in range(len(theta1_display)-1) ])
        line1_collection.set_segments(segments_phase1)
        line1_collection.set_array(np.linspace(0, 1, len(segments_phase1)))
    else:
        line1_collection.set_segments([])

    if len(theta2_display) > 1:
        segments_phase2 = np.array([ [(theta2_display[i], z2_display[i]),
                                      (theta2_display[i+1], z2_display[i+1])]
                                  for i in range(len(theta2_display)-1) ])
        line2_collection.set_segments(segments_phase2)
        line2_collection.set_array(np.linspace(0, 1, len(segments_phase2)))
    else:
        line2_collection.set_segments([])

    return scatter1, scatter2, point1, point2, line_pendulum, trajectory1_collection, trajectory2_collection, line1_collection, line2_collection, bob1, bob2

# ------------------------- Run Animation --------------------------
# Create the animation object
anim = FuncAnimation(fig, animate, init_func=init_anim,
                     frames=num_frames, interval=dt*1000, blit=True)

# ------------------------ Display Animation -----------------------
plt.show()

# ----------------------- Poincaré Section --------------------------
# Optional: Plotting the Poincaré section after the animation
# Uncomment the following lines if you wish to see the Poincaré section

# Define a Poincaré Section: theta2 crosses zero with positive velocity
crossings_theta1 = []
crossings_z1 = []

for i in range(1, len(theta2_history)):
    if theta2_history[i-1] < 0 and theta2_history[i] >= 0 and z2_history[i] > 0:
        crossings_theta1.append(theta1_history[i])
        crossings_z1.append(z1_history[i])

# Create Figure for Poincaré Section
fig_poincare, ax_poincare = plt.subplots(figsize=(7, 7))  # Square figure
ax_poincare.scatter(crossings_theta1, crossings_z1, s=10, color='black', alpha=0.6)
ax_poincare.set_xlabel(r'$\theta_1$ (rad)')
ax_poincare.set_ylabel(r'$\dot{\theta}_1$ (rad/s)')
ax_poincare.set_title(r'Poincaré Section: $\theta_2 = 0$, $\dot{\theta}_2 > 0$')
ax_poincare.grid(True)
ax_poincare.set_aspect('equal')  # Ensure the plot is square
plt.show()
