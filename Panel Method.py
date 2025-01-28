import math
import numpy
from scipy import integrate
from matplotlib import pyplot as plt
import numpy as np

# Commented out IPython magic to ensure Python compatibility.
# reads of the geometry from a data file
import csv
import numpy as np

# Initialize empty lists for x and y coordinates
x, y = [], []

# Open the CSV file and read data
with open('fxs21158.csv', 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    
    for row in reader:
        # Remove empty strings and convert to floats
        row = [float(value) for value in row if value]
        
        # Ensure each row has at least two values (x and y)
        if len(row) >= 2:
            x.append(row[0])
            y.append(row[1])

# Convert lists to numpy arrays
x = np.array(x)
y = np.array(y)



# plots the geometry
# %matplotlib inline

val_x, val_y = 0.1, 0.2
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
x_start, x_end = x_min-val_x*(x_max-x_min), x_max+val_x*(x_max-x_min)
y_start, y_end = y_min-val_y*(y_max-y_min), y_max+val_y*(y_max-y_min)

size = 10
plt.figure(figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
plt.gca().set_aspect('equal')
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.plot(x, y, color='k', linestyle='-', linewidth=2)

class Panel:
    """Contains information related to one panel."""
    def __init__(self, xa, ya, xb, yb):
        """Creates a panel.

        Arguments
        ---------
        xa, ya -- Cartesian coordinates of the first end-point.
        xb, yb -- Cartesian coordinates of the second end-point.
        """
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb

        self.xc, self.yc = (xa+xb)/2, (ya+yb)/2       # control-point (center-point)
        self.length = math.sqrt((xb-xa)**2+(yb-ya)**2)     # length of the panel

        # orientation of the panel (angle between x-axis and panel's normal)
        if xb-xa <= 0.:
            self.beta = math.acos((yb-ya)/self.length)
        elif xb-xa > 0.:
            self.beta = math.pi + math.acos(-(yb-ya)/self.length)

        # location of the panel
        if self.beta <= math.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'

        self.sigma = 0.                             # source strength
        self.vt = 0.                                # tangential velocity
        self.cp = 0.                                # pressure coefficient

def define_panels(x, y, N=100):
    """Discretizes the geometry into panels using 'cosine' method.

    Arguments
    ---------
    x, y -- Cartesian coordinates of the geometry (1D arrays).
    N - number of panels.

    Returns
    -------
    panels -- Numpy array of panels.
    """
    R = (x.max()-x.min())/2                                    # radius of the circle
    x_center = (x.max()+x.min())/2                             # x-coord of the center
    x_circle = x_center + R*numpy.cos(numpy.linspace(0, 2*math.pi, N+1))  # x-coord of the circle points

    x_ends = numpy.copy(x_circle)      # projection of the x-coord on the surface
    y_ends = numpy.empty_like(x_ends)  # initialization of the y-coord Numpy array

    x, y = numpy.append(x, x[0]), numpy.append(y, y[0])    # extend arrays using numpy.append

    # computes the y-coordinate of end-points
    I = 0
    for i in range(N):
        while I < len(x)-1:
            if (x[I] <= x_ends[i] <= x[I+1]) or (x[I+1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I+1]-y[I])/(x[I+1]-x[I])
        b = y[I+1] - a*x[I+1]
        y_ends[i] = a*x_ends[i] + b
    y_ends[N] = y_ends[0]

    panels = numpy.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i+1], y_ends[i+1])

    return panels

N = 100                           # number of panels
panels = define_panels(x, y, N)  # discretizes of the geometry into panels

# plots the geometry and the panels
val_x, val_y = 0.1, 0.2
x_min, x_max = min( panel.xa for panel in panels ), max( panel.xa for panel in panels )
y_min, y_max = min( panel.ya for panel in panels ), max( panel.ya for panel in panels )
x_start, x_end = x_min-val_x*(x_max-x_min), x_max+val_x*(x_max-x_min)
y_start, y_end = y_min-val_y*(y_max-y_min), y_max+val_y*(y_max-y_min)

size = 10
plt.figure(figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
plt.gca().set_aspect('equal')
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.plot(x, y, color='k', linestyle='-', linewidth=2)
plt.plot(numpy.append([panel.xa for panel in panels], panels[0].xa),
         numpy.append([panel.ya for panel in panels], panels[0].ya),
         linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305');

class Freestream:
    """Freestream conditions."""
    def __init__(self, u_inf=1.0, alpha=0.0):
        """Sets the freestream conditions.

        Arguments
        ---------
        u_inf -- Farfield speed (default 1.0).
        alpha -- Angle of attack in degrees (default 0.0).
        """
        self.u_inf = u_inf
        self.alpha = alpha*math.pi/180          # degrees --> radians

# defines and creates the object freestream
u_inf = 1
alpha = 0                                  # freestream spee
freestream = Freestream(u_inf, alpha)      # instantiation of the object freestream

def integral(x, y, panel, dxdz, dydz):
    """Evaluates the contribution of a panel at one point.

    Arguments
    ---------
    x, y -- Cartesian coordinates of the point.
    panel -- panel which contribution is evaluated.
    dxdz -- derivative of x in the z-direction.
    dydz -- derivative of y in the z-direction.

    Returns
    -------
    Integral over the panel of the influence at one point.
    """
    def func(s):
        return ( ((x - (panel.xa - math.sin(panel.beta)*s))*dxdz
                  + (y - (panel.ya + math.cos(panel.beta)*s))*dydz)
                / ((x - (panel.xa - math.sin(panel.beta)*s))**2
                   + (y - (panel.ya + math.cos(panel.beta)*s))**2) )
    return integrate.quad(lambda s:func(s), 0., panel.length)[0]

def source_matrix(panels):
    """Builds the source matrix.

    Arguments
    ---------
    panels -- array of panels.

    Returns
    -------
    A -- NxN matrix (N is the number of panels).
    """
    N = len(panels)
    A = numpy.empty((N, N), dtype=float)
    numpy.fill_diagonal(A, 0.5)

    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                A[i,j] = 0.5/math.pi*integral(p_i.xc, p_i.yc, p_j, math.cos(p_i.beta), math.sin(p_i.beta))

    return A

def vortex_array(panels):
    """Builds the vortex array.

    Arguments
    ---------
    panels - array of panels.

    Returns
    -------
    a -- 1D array (Nx1, N is the number of panels).
    """
    a = numpy.zeros(len(panels), dtype=float)

    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                a[i] -= 0.5/math.pi*integral(p_i.xc, p_i.yc, p_j, +math.sin(p_i.beta), -math.cos(p_i.beta))

    return a


def kutta_array(panels):
    """Builds the Kutta-condition array.

    Arguments
    ---------
    panels -- array of panels.

    Returns
    -------
    a -- 1D array (Nx1, N is the number of panels).
    """
    N = len(panels)
    a = numpy.zeros(N+1, dtype=float)

    a[0] = 0.5/math.pi*integral(panels[N-1].xc, panels[N-1].yc, panels[0],
                           -math.sin(panels[N-1].beta), +math.cos(panels[N-1].beta))
    a[N-1] = 0.5/math.pi*integral(panels[0].xc, panels[0].yc, panels[N-1],
                             -math.sin(panels[0].beta), +math.cos(panels[0].beta))

    for i, panel in enumerate(panels[1:N-1]):
        a[i] = 0.5/math.pi*(integral(panels[0].xc, panels[0].yc, panel,
                               -math.sin(panels[0].beta), +math.cos(panels[0].beta))
                     + integral(panels[N-1].xc, panels[N-1].yc, panel,
                               -math.sin(panels[N-1].beta), +math.cos(panels[N-1].beta)) )

        a[N] -= 0.5/math.pi*(integral(panels[0].xc, panels[0].yc, panel,
                               +math.cos(panels[0].beta), +math.sin(panels[0].beta))
                     + integral(panels[N-1].xc, panels[N-1].yc, panel,
                               +math.cos(panels[N-1].beta), +math.sin(panels[N-1].beta)) )

    return a

def build_matrix(panels):
    """Builds the matrix of the linear system.

    Arguments
    ---------
    panels -- array of panels.

    Returns
    -------
    A -- (N+1)x(N+1) matrix (N is the number of panels).
    """
    N = len(panels)
    A = numpy.empty((N+1, N+1), dtype=float)

    AS = source_matrix(panels)
    av = vortex_array(panels)
    ak = kutta_array(panels)

    A[0:N,0:N], A[0:N,N], A[N,:] = AS[:,:], av[:], ak[:]

    return A

def build_rhs(panels, freestream):
    """Builds the RHS of the linear system.

    Arguments
    ---------
    panels -- array of panels.
    freestream -- farfield conditions.

    Returns
    -------
    b -- 1D array ((N+1)x1, N is the number of panels).
    """
    N = len(panels)
    b = numpy.empty(N+1,dtype=float)

    for i, panel in enumerate(panels):
        b[i] = - freestream.u_inf * numpy.cos(freestream.alpha - panel.beta)
    b[N] = -freestream.u_inf*( math.sin(freestream.alpha-panels[0].beta)
                              +math.sin(freestream.alpha-panels[N-1].beta) )

    return b

A = build_matrix(panels)                  # calculates the singularity matrix
b = build_rhs(panels, freestream)         # calculates the freestream RHS

len(b)

def gauss_seidel(A, b, x0=None, max_iterations=1000, tolerance=1e-10):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    for it in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            sum_ = b[i]
            for j in range(n):
                if j != i:
                    sum_ -= A[i][j] * x[j]
            x[i] = sum_ / A[i][i]

        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            break

    return x

# solves the linear system
variables = gauss_seidel(A, b)

for i, panel in enumerate(panels):
	panel.sigma = variables[i]
gamma = variables[-1]

def get_tangential_velocity(panels, freestream, gamma):
    """Computes the tangential velocity on the surface.

    Arguments
    ---------
    panels -- array of panels.
    freestream -- farfield conditions.
    gamma -- circulation density.
    """
    N = len(panels)
    A = numpy.empty((N, N+1), dtype=float)
    numpy.fill_diagonal(A, 0.0)

    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                A[i,j] = 0.5/math.pi*integral(p_i.xc, p_i.yc, p_j, -math.sin(p_i.beta), +math.cos(p_i.beta))
                A[i,N] -= 0.5/math.pi*integral(p_i.xc, p_i.yc, p_j, +math.cos(p_i.beta), +math.sin(p_i.beta))

    b = freestream.u_inf * numpy.sin([freestream.alpha - panel.beta for panel in panels])

    var = numpy.append([panel.sigma for panel in panels], gamma)

    vt = numpy.dot(A, var) + b
    for i, panel in enumerate(panels):
        panel.vt = vt[i]

# computes the tangential velocity at each panel center.
get_tangential_velocity(panels, freestream, gamma)

def get_pressure_coefficient(panels, freestream):
    """Computes the surface pressure coefficients.

    Arguments
    ---------
    panels -- array of panels.
    freestream -- farfield conditions.
    """
    for panel in panels:
        panel.cp = 1.0 - (panel.vt/freestream.u_inf)**2

# Define specific angles of attack
alpha_values = [0, 5]  # Angles of attack in degrees
cl_at_zero_alpha = []
# cl_at_5alpha = []
for alpha in alpha_values:
    # Update freestream conditions for each specified angle
    freestream = Freestream(u_inf=1.0, alpha=alpha)
    
    # Build the matrix A and the RHS b for the linear system
    A = build_matrix(panels)
    b = build_rhs(panels, freestream)
    
    # Solve the linear system using Gauss-Seidel method
    variables = gauss_seidel(A, b)
    
    # Update the source strengths and circulation density (gamma)
    for i, panel in enumerate(panels):
        panel.sigma = variables[i]
    gamma = variables[-1]
    
    # Compute tangential velocities and pressure coefficients
    get_tangential_velocity(panels, freestream, gamma)
    get_pressure_coefficient(panels, freestream)
    
    # Plot the surface pressure coefficient for each angle of attack
    val_x, val_y = 0.1, 0.2
    x_min, x_max = min(panel.xa for panel in panels), max(panel.xa for panel in panels)
    cp_min, cp_max = min(panel.cp for panel in panels), max(panel.cp for panel in panels)
    x_start, x_end = x_min - val_x * (x_max - x_min), x_max + val_x * (x_max - x_min)
    y_start, y_end = cp_min - val_y * (cp_max - cp_min), cp_max + val_y * (cp_max - cp_min)

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('$C_p$', fontsize=16)
    plt.plot([panel.xc for panel in panels if panel.loc == 'upper'],
             [panel.cp for panel in panels if panel.loc == 'upper'],
             color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
    plt.plot([panel.xc for panel in panels if panel.loc == 'lower'],
             [panel.cp for panel in panels if panel.loc == 'lower'],
             color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
    plt.legend(['upper', 'lower'], loc='best', prop={'size':14})
    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)
    plt.gca().invert_yaxis()
    plt.title(f'Pressure Coefficient $C_p$ at $\\alpha = {alpha}^\\circ$')
    plt.show()
    
    # Calculate and print the lift coefficient for current alpha
    cl = gamma * sum(panel.length for panel in panels) / (0.5 * freestream.u_inf * (x_max - x_min))
    cl_at_zero_alpha.append(cl)
    # cl_at_zero_alpha = min(cl)
    # cl_at_5alpha = max(cl)


# calculates the accuracy
accuracy = sum([panel.sigma*panel.length for panel in panels])
# print ('strengths:', accuracy)

# Set up the angle of attack range and store the results
alpha_values = np.linspace(-10, 15, 30)  # Alpha values from -10 to 15 degrees
cl_values = []                           # List to store Cl values

for alpha in alpha_values:
    # Update freestream conditions with current angle of attack
    freestream = Freestream(u_inf=1.0, alpha=alpha)
    
    # Build the matrix A and the RHS b for the linear system
    A = build_matrix(panels)
    b = build_rhs(panels, freestream)
    
    # Solve the linear system using Gauss-Seidel method
    variables = gauss_seidel(A, b)
    
    # Update the source strengths and circulation density (gamma)
    for i, panel in enumerate(panels):
        panel.sigma = variables[i]
    gamma = variables[-1]
    
    # Compute the tangential velocities and pressure coefficients
    get_tangential_velocity(panels, freestream, gamma)
    get_pressure_coefficient(panels, freestream)
    
    # Calculate lift coefficient for current alpha
    x_min, x_max = min(panel.xa for panel in panels), max(panel.xa for panel in panels)
    cl = gamma * sum(panel.length for panel in panels) / (0.5 * freestream.u_inf * (x_max - x_min))
    
    # Store the lift coefficient
    cl_values.append(cl)

# Plotting Cl vs Alpha
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, cl_values, marker='o', linestyle='-', color='b')
plt.xlabel(r'$\alpha$ (degrees)', fontsize=16)
plt.ylabel(r'$C_l$', fontsize=16)
plt.title(r'$C_l$ vs $\alpha$', fontsize=18)
plt.grid(True)
plt.show()

 

slope, intercept = np.polyfit(alpha_values, cl_values, 1)
# Print the slope
print(f"Slope of Cl vs Alpha curve : {slope}per Degree")
alpha_cl_zero = -intercept / slope
print(f"zero lift angle of attack: {alpha_cl_zero:.2f} degrees")
print(f"Cl at zero angle of attack: {min(cl_at_zero_alpha)} degrees")
N_panels = len(x) - 1



#moment coefficient
x_mid = [(x[i] + x[i + 1]) / 2 for i in range(N_panels)] 

#calculate moment coefficient
def moment_coefficient(cl: float, x_mid: list[float], N: int): # Change cl type hint to float
    CM_quater = 0
    CM_le = 0
    for i in range(N):
        CM_quater += ((cl * (x_mid[i] - 0.25))/100)  # Use cl directly instead of cl[i]
        CM_le += ((cl * (x_mid[i]))/100)
    print(f"Moment Coefficient at Quater chord [CM_quater(c/4)]: {CM_quater}")
    print(f"Moment Coefficient at Leading Edge [CM_le]: {CM_le}")
    # print("center of pressure:" (CM_le/cl))
    return CM_quater

moment_coefficient(cl, x_mid, N_panels) # Pass cl, not cl_values

# Center of pressure calculation with minimum CM tracking
def center_of_pressure(cl: float, x_mid: list[float], N: int):
    min_CM = float('inf')  # Start with a very large number
    best_j = 0  # Variable to store the corresponding value of j

    for j in [i * 0.001 for i in range(1001)]:  # j goes from 0 to 1 in steps of 0.001
        CM = 0
        for i in range(N):
            CM += (cl * (x_mid[i] - j)) / 100  # Calculate CM for each j

        # Track minimum CM and corresponding j
        if abs(CM) < abs(min_CM):  # Use absolute value of CM for comparison
            min_CM = CM
            best_j = j
        
        # Debugging output to track CM and j
        # print(f"j = {j}, CM = {CM}")

    # Print the value of j where CM is minimum
    print(f"The center of pressure is located at: {best_j}c")
    
    return best_j, min_CM

# Call the function (assuming cl, x_mid, and N_panels are defined)
center_of_pressure(cl, x_mid, N_panels)


def calculate_moment_coefficients(cl: float, x_mid: list[float], x_ref: float = 0): 
    CM = [(cl * (x_mid[i] - x_ref)) for i in range(len(x_mid))]                         #As chord length is normalized to 1
    return CM
CM = calculate_moment_coefficients(cl, x_mid)



# # Aerodynamic Center
# Define lists for storage
mo = []

# Function to calculate the moment coefficient slope
def momentslope(cl_values: list[float], x_mid: list[float], N: int):
    for cl in cl_values:  # Loop over each lift coefficient
        moquater = 0
        for i in range(N):
            # Calculate moment coefficient around quarter-chord point
            moquater += ((cl * (x_mid[i] - 0.25)) / 100)
        # Append each calculated moquater to the mo list
        mo.append(moquater)

# # Example lists for alpha values and cl values (replace with your actual data)
# alpha_values = [0, 2, 4, 6, 8]  # Angle of attack in degrees
# cl_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Lift coefficient values at each alpha
# x_mid = [0.1, 0.2, 0.3, 0.4, 0.5]  # Chordwise midpoints (example values)
N = len(x_mid)

# Calculate moment coefficients for each alpha
momentslope(cl_values, x_mid, N)

# Fit a linear line to the moment coefficients vs. alpha values
slope, intercept = np.polyfit(alpha_values, mo, 1)
print(f"Slope of Cm vs alpha is: {slope}")

#We know
aerodynamic_center = (-slope + 0.25)                                                #page 366 Anderson
print(f"Aerodynamic center is at: {aerodynamic_center}c")