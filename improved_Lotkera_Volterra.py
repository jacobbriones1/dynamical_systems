# Improved Predator-Prey Model with Irreversible Extinction

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

# Define the predator-prey model with extinction conditions
def predator_prey_model(t, y, a, K, b, c, d, e):
    R, F = y

    # Extinction thresholds
    extinction_threshold = 1e-6

    # Ensure populations are non-negative
    R = max(R, 0)
    F = max(F, 0)

    # Check for extinction
    if R <= extinction_threshold:
        R = 0
        dRdt = 0
    else:
        # Prey growth with logistic term
        dRdt = a * R * (1 - R / K) - (b * R * F) / (1 + e * R)

    if F <= extinction_threshold:
        F = 0
        dFdt = 0
    else:
        # Predator growth
        if R > extinction_threshold:
            dFdt = -c * F + (d * R * F) / (1 + e * R)
        else:
            dFdt = -c * F  # Predators decline without prey

    return [dRdt, dFdt]

# Initial parameters
a_init = 1.0      # Prey intrinsic growth rate
K_init = 500.0    # Prey carrying capacity
b_init = 0.1      # Maximum predation rate coefficient
c_init = 0.5      # Predator mortality rate
d_init = 0.075    # Predator reproduction efficiency
e_init = 0.01     # Half-saturation constant (handling time)

R0_init = 50.0    # Initial prey population
F0_init = 10.0    # Initial predator population

# Time span
t_max = 200
t_span = [0, t_max]
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Set up the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)
line1, = ax.plot([], [], lw=2, label='Trajectory in Phase Plane')
ax.set_xlabel('Prey Population')
ax.set_ylabel('Predator Population')
ax.set_title('Phase Plane of the Improved Predator-Prey Model')
ax.grid(True)
ax.legend()

# Add text annotation for real-time population display
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top',
               fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Define sliders for parameters and initial conditions
axcolor = 'lightgoldenrodyellow'
ax_a = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_K = plt.axes([0.25, 0.26, 0.65, 0.03], facecolor=axcolor)
ax_b = plt.axes([0.25, 0.22, 0.65, 0.03], facecolor=axcolor)
ax_c = plt.axes([0.25, 0.18, 0.65, 0.03], facecolor=axcolor)
ax_d = plt.axes([0.25, 0.14, 0.65, 0.03], facecolor=axcolor)
ax_e = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_R0 = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
ax_F0 = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor=axcolor)

s_a = Slider(ax_a, 'a (Prey Growth Rate)', 0.0, 3.0, valinit=a_init, valstep=0.01)
s_K = Slider(ax_K, 'K (Carrying Capacity)', 100.0, 1000.0, valinit=K_init, valstep=10)
s_b = Slider(ax_b, 'b (Predation Rate)', 0.0, 1.0, valinit=b_init, valstep=0.01)
s_c = Slider(ax_c, 'c (Predator Mortality)', 0.0, 1.0, valinit=c_init, valstep=0.01)
s_d = Slider(ax_d, 'd (Predator Efficiency)', 0.0, 0.5, valinit=d_init, valstep=0.005)
s_e = Slider(ax_e, 'e (Handling Time)', 0.0, 0.1, valinit=e_init, valstep=0.001)
s_R0 = Slider(ax_R0, 'Initial Prey (R0)', 0.0, 200.0, valinit=R0_init, valstep=1)
s_F0 = Slider(ax_F0, 'Initial Predator (F0)', 0.0, 50.0, valinit=F0_init, valstep=1)

# Variables to control animation state
is_paused = False
frame_number = [0]  # List to make frame_number mutable in nested functions

# Function to create and start the animation
def start_animation():
    global ani
    ani = animation.FuncAnimation(fig, animate, frames=len(t_eval), interval=25, blit=False)
    plt.draw()

# Animation function
def animate(i):
    if is_paused:
        return line1, text
    frame = frame_number[0]
    line1.set_data([R[:frame+1]], [F[:frame+1]])
    current_R = max(R[frame], 0)
    current_F = max(F[frame], 0)
    text.set_text(f'Prey (R): {int(np.round(current_R))}\nPredator (F): {int(np.round(current_F))}')
    frame_number[0] = frame + 1
    if frame_number[0] >= len(t_eval):
        frame_number[0] = len(t_eval) - 1  # Stop at the last frame
    return line1, text

# Update function for sliders
def update(val):
    global R, F, ani, frame_number, is_paused
    # Stop the animation
    if 'ani' in globals():
        ani.event_source.stop()
    is_paused = False
    button_start_stop.label.set_text('Stop')
    frame_number[0] = 0  # Reset frame number

    # Get new parameters
    a = s_a.val
    K = s_K.val
    b = s_b.val
    c = s_c.val
    d = s_d.val
    e = s_e.val
    R0 = s_R0.val
    F0 = s_F0.val
    y0 = [R0, F0]
    params = (a, K, b, c, d, e)

    # Recompute the solution
    solution = solve_ivp(
        predator_prey_model, t_span, y0, t_eval=t_eval, args=params,
        method='RK45', rtol=1e-8, atol=1e-10
    )
    R = solution.y[0]
    F = solution.y[1]

    # Enforce extinction thresholds
    R[R < 1e-6] = 0
    F[F < 1e-6] = 0

    # Clear the line data and text
    line1.set_data([], [])
    text.set_text('')

    # Update plot limits
    ax.set_xlim(0, max(max(R), 1)*1.1)
    ax.set_ylim(0, max(max(F), 1)*1.1)

    # Restart the animation
    start_animation()

# Connect the sliders to the update function
s_a.on_changed(update)
s_K.on_changed(update)
s_b.on_changed(update)
s_c.on_changed(update)
s_d.on_changed(update)
s_e.on_changed(update)
s_R0.on_changed(update)
s_F0.on_changed(update)

# Reset button
resetax = plt.axes([0.7, 0.92, 0.08, 0.04])
button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_a.reset()
    s_K.reset()
    s_b.reset()
    s_c.reset()
    s_d.reset()
    s_e.reset()
    s_R0.reset()
    s_F0.reset()
button_reset.on_clicked(reset)

# Start/Stop button
start_stop_ax = plt.axes([0.81, 0.92, 0.08, 0.04])
button_start_stop = Button(start_stop_ax, 'Stop', color=axcolor, hovercolor='0.975')

def start_stop(event):
    global is_paused
    if is_paused:
        is_paused = False
        button_start_stop.label.set_text('Stop')
    else:
        is_paused = True
        button_start_stop.label.set_text('Start')
button_start_stop.on_clicked(start_stop)

# Initial computation and animation
update(None)
plt.show()
