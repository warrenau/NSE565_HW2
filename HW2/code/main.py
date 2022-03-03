# ---------------------------------#
# -- main script for NSE 565 HW2 --#
# --------- Austin Warren ---------#
# ---------- Winter 2022 ----------#
# ---------------------------------#

import numpy as np
import matplotlib.pyplot as plt

# define functions for spatial and temporal discretizations
# - coefficient generation
# - time step

def UDS_positve(num_volumes, tot_length, density, velocity, diffusion, left, right):
    """ Upwind scheme solver for positive velocity
    """
    dx = tot_length/num_volumes
    phi = np.zeros(num_volumes)
    A = np.zeros((num_volumes,num_volumes))
    Q = np.zeros(num_volumes)

    

    for j in range(num_volumes):

        if j == 0:
            A[j,0] = density*velocity + 2*diffusion/dx
            A[j,1] = diffusion/dx
            Q[j] = density*velocity*left + diffusion*left/dx

        elif j == num_volumes-1:
            A[j,j-1] = -diffusion/dx
            A[j,j] = -density*velocity - 2*diffusion/dx
            Q[j] = -density*velocity*right + diffusion*right/dx

        else:
            A[j,j-1] = -density*velocity - diffusion/dx
            A[j,j] = density*velocity + 2*diffusion/dx
            A[j,j+1] = diffusion/dx
            Q[j] = 0

    phi = np.linalg.solve(A,Q)
    return phi


def EE_time_step(dt, dx, phi_in, density, velocity, diffusion, left, right):
    """ Explicit Euler solve for next time step
    """
    phi_out = np.zeros(len(phi_in))

    for j in range(len(phi_in)):
        if j==0:
            a = (velocity*dt/dx + 2*diffusion*dt/density/dx**2)
            b = (1-velocity*dt/dx-3*diffusion*dt/density/dx**2)
            c = (diffusion*dt/density/dx**2)
            phi_out[j] = a*left + b*phi_in[j] + c*phi_in[j+1]
        elif j==len(phi_in)-1:
            a = (velocity*dt/dx + diffusion*dt/density/dx**2)
            b = (1-velocity*dt/dx-3*diffusion*dt/density/dx**2)
            c = (2*diffusion*dt/density/dx**2)
            phi_out[j] = a*phi_in[j-1] + b*phi_in[j] + c*right
        else:
            a = (velocity*dt/dx + diffusion*dt/density/dx**2)
            b = (1-velocity*dt/dx-2*diffusion*dt/density/dx**2)
            c = (diffusion*dt/density/dx**2)
            phi_out[j] = a*phi_in[j-1] + b*phi_in[j] + c*phi_in[j+1]
    return phi_out


def IE_time_step(dt, dx, phi_in, density, velocity, diffusion, left, right):
    """ Implicit Euler solve for next time step
    """
    phi_out = np.zeros(len(phi_in))
    A = np.zeros((len(phi_in), len(phi_in)))
    Q = np.zeros(len(phi_in))

    for j in range(len(phi_in)):
        if j == 0:
            A[0,0] = (1 + velocity*dt/dx + 3*diffusion*dt/density/dx**2)
            A[0,1] = -diffusion*dt/density/dx**2
            Q[0] = phi_in[0] + (velocity*dt/dx + 2*diffusion*dt/density/dx**2)*left
        elif j==len(phi_in)-1:
            A[j,j-1] = (-velocity*dt/dx - diffusion*dt/density/dx**2)
            A[j,j] = (1 + velocity*dt/dx + 3*diffusion*dt/density/dx**2)
            Q[j] = phi_in[j] + (2*diffusion*dt/density/dx**2)*right
        else:
            A[j,j-1] = (-velocity*dt/dx - diffusion*dt/density/dx**2)
            A[j,j] = (1 + velocity*dt/dx + 2*diffusion*dt/density/dx**2)
            A[j,j+1] = (-diffusion*dt/density/dx**2)
            Q[j] = phi_in[j]

    phi_out = np.linalg.solve(A,Q)
    return phi_out

# define central difference scheme function to use for each case
def cds_ss(num_volumes, tot_length, velocity, density, diffusion, left, right):
    """Function to perform central difference scheme to solve one-dimensional steady state transport with convection and diffusion.

    Parameters
    ----------
    num_volumes : float
        The number of discretized volumes.
    tot_length : float
        The total length of the pipe in meters.
    velocity : float
        The average velocity of the flow in meters per second.
    density : float
        The density of the flow in kilograms per cubic meter.
    diffusion : float
        The diffusion coefficient in kilogram-seconds per meter.
    left : float
        The left boundary condition.
    right : float
        The right boundary condition.

    Returns
    -------
    phi : numpy.ndarray
        Solved flux profile.
    """
    dx = tot_length/num_volumes
    phi = np.zeros(num_volumes)
    A = np.zeros((num_volumes,num_volumes))
    Q = np.zeros(num_volumes)

    

    for j in range(num_volumes):

        if j == 0:
            A[j,0] = density*velocity/2 + 3*diffusion/dx
            A[j,1] = density*velocity/2 - diffusion/dx
            Q[j] = density*velocity*left + 2*diffusion*left/dx

        elif j == num_volumes-1:
            A[j,j-1] = -density*velocity/2 - diffusion/dx
            A[j,j] = -density*velocity/2 + 3*diffusion/dx
            Q[j] = -density*velocity*right + 2*diffusion*right/dx

        else:
            A[j,j-1] = -density*velocity/2 - diffusion/dx
            A[j,j] = 2*diffusion/dx
            A[j,j+1] = density*velocity/2 - diffusion/dx
            Q[j] = 0

    phi = np.linalg.solve(A,Q)
    return phi



# variables
tot_length = 1.0
density = 1.0
diffusion = 0.1
left = 100
right = 50
velocity = 2.5
num_volumes = 20

phi_init = np.zeros(num_volumes)
phi_init[:] = 50


K = np.array([0.2, 2.0, 20])
dx = tot_length/num_volumes
x = np.linspace(dx/2, tot_length-dx/2,num_volumes)
#volume = dx
dt = K*dx/velocity
max_iter = 256

# steady state CDS
phi_ss = cds_ss(num_volumes, tot_length, velocity, density, diffusion, left, right)


# explicit euler
phi_in_ee = np.zeros(num_volumes)
error_ee = np.zeros(len(dt))

for j in range(len(dt)):
    phi_in_ee[:] = phi_init[:]
    m=0
    phi_plot_ee = np.zeros((5,num_volumes))
    
    for n in range(max_iter):
        phi_out_ee = EE_time_step(dt[j], dx, phi_in_ee, density, velocity, diffusion, left, right)
        if n==0 or n==4 or n==16 or n==64:
            phi_plot_ee[m,:] = phi_in_ee[:]
            m += 1
        elif n==255:
            phi_plot_ee[m,:] = phi_out_ee[:]
        phi_in_ee[:] = phi_out_ee[:]

    # plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.figure(facecolor='w', edgecolor='k', dpi=300)
    plt.plot(x, phi_plot_ee[0,:], '-k', label='n=0')
    plt.plot(x, phi_plot_ee[1,:], '-r', label='n=4')
    plt.plot(x, phi_plot_ee[2,:], '-b', label='n=16')
    plt.plot(x, phi_plot_ee[3,:], '-g', label='n=64')
    plt.plot(x, phi_plot_ee[4,:], '-m', label='n=256')
    plt.xlabel('x (m)')
    plt.ylabel(r'$\phi$')
    plt.figlegend(bbox_to_anchor=(1.0,0.9))
    plt.grid(b=True, which='major', axis='both')
    plt.savefig('HW2/plots/graph_EE_case'+str(j+1)+'.pdf',transparent=True)

    # compare transient to steady
    error_ee[j] = np.sum(np.absolute(phi_plot_ee[4,:]-phi_ss)) / num_volumes


# implicit euler
phi_in_ie = np.zeros(num_volumes)
error_ie = np.zeros(len(dt))
for h in range(len(dt)):
    # reset inputs
    phi_in_ie[:] = phi_init[:]
    g=0
    phi_plot_ie = np.zeros((5,num_volumes))
    
    for k in range(max_iter):
        phi_out_ie = IE_time_step(dt[h], dx, phi_in_ie, density, velocity, diffusion, left, right)
        if k==0 or k==4 or k==16 or k==64:
            phi_plot_ie[g,:] = phi_in_ie[:]
            g += 1
        elif k==255:
            phi_plot_ie[g,:] = phi_out_ie[:]
        phi_in_ie[:] = phi_out_ie[:]

    # plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.figure(facecolor='w', edgecolor='k', dpi=300)
    plt.plot(x, phi_plot_ie[0,:], '-k', label='n=0')
    plt.plot(x, phi_plot_ie[1,:], '-r', label='n=4')
    plt.plot(x, phi_plot_ie[2,:], '-b', label='n=16')
    plt.plot(x, phi_plot_ie[3,:], '-g', label='n=64')
    plt.plot(x, phi_plot_ie[4,:], '-m', label='n=256')
    plt.xlabel('x (m)')
    plt.ylabel(r'$\phi$')
    plt.figlegend(bbox_to_anchor=(1.0,0.9))
    plt.grid(b=True, which='major', axis='both')
    plt.savefig('HW2/plots/graph_IE_case'+str(h+1)+'.pdf',transparent=True)

    # compare transient to steady
    error_ie[h] = np.sum(np.absolute(phi_plot_ie[4,:]-phi_ss)) / num_volumes


# generate latex table for error
out_file = open('HW2/tabs/error_tab_ee.tex','w')
out_file.write(
                '\\begin{table}[htbp]\n'+
                '\t \centering\n'+
                '\t \caption{Norm values for Explicit Euler.}\n'+
                '\t \\begin{tabular}{cc}\n'+
                '\t\t \\toprule\n'+
                '\t\t $K$ & Norm \\\ \n'+
                '\t\t \midrule \n'+
                '\t\t 0.2 & '+str(error_ee[0])+' \\\ \n'+
                '\t\t 2.0 & '+str(error_ee[1])+' \\\ \n'+
                '\t\t 20.0 & '+str(error_ee[2])+' \\\ \n'+
                '\t\t \\bottomrule \n'+
                '\t \end{tabular} \n'+
                '\t \label{tab:error ee} \n'+
                '\end{table}'
)

out_file = open('HW2/tabs/error_tab_ie.tex','w')
out_file.write(
                '\\begin{table}[htbp]\n'+
                '\t \centering\n'+
                '\t \caption{Norm values for Implicit Euler.}\n'+
                '\t \\begin{tabular}{cc}\n'+
                '\t\t \\toprule\n'+
                '\t\t $K$ & Norm \\\ \n'+
                '\t\t \midrule \n'+
                '\t\t 0.2 & '+str(error_ie[0])+' \\\ \n'+
                '\t\t 2.0 & '+str(error_ie[1])+' \\\ \n'+
                '\t\t 20.0 & '+str(error_ie[2])+' \\\ \n'+
                '\t\t \\bottomrule \n'+
                '\t \end{tabular} \n'+
                '\t \label{tab:error ie} \n'+
                '\end{table}'
)