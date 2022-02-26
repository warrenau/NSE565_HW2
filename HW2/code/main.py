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


def EE_time_step(dt, num_volumes, tot_length, phi_in, density, volume, velocity, diffusion, left, right):
    """ Explicit Euler solve for next time step
    """
    dx = tot_length/num_volumes
    phi_out = np.zeros(len(phi_in))

    for j in range(len(phi_in)):
        if j==0:
            phi_out[j] = velocity*dt/volume*left + (1-velocity*dt/volume-2*diffusion*dt/dx/density/volume)*phi_in[j] + diffusion*dt/dx/density/volume*phi_in[j+1]
        elif j==len(phi_in)-1:
            phi_out[j] = velocity*dt/volume*phi_in[j-1] + (1-velocity*dt/volume-2*diffusion*dt/dx/density/volume)*phi_in[j] + diffusion*dt/dx/density/volume*right
        else:
            phi_out[j] = velocity*dt/volume*phi_in[j-1] + (1-velocity*dt/volume-2*diffusion*dt/dx/density/volume)*phi_in[j] + diffusion*dt/dx/density/volume*phi_in[j+1]
    return phi_out




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
volume = dx*1*1
dt = K*dx/velocity
max_iter = 256


for j in range(len(dt)):
    t = np.linspace(0,dt[j]*256,256)
    phi_in = phi_init
    m=0
    phi_plot = np.zeros((5,num_volumes))
    for n in range(len(t)):
        if n==0 or n==4 or n==16 or n==64 or n==256:
            phi_plot[m,:] = phi_in[:]
            m += 1
        phi_out = EE_time_step(dt[j], num_volumes, tot_length, phi_in, density, volume, velocity, diffusion, left, right)
        phi_in[:] = phi_out[:]

# plot
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.figure(facecolor='w', edgecolor='k', dpi=300)
plt.plot(x, phi_plot[0,:], '-k', label='n=0')
plt.plot(x, phi_plot[1,:], '-r', label='n=4')
plt.plot(x, phi_plot[2,:], '-b', label='n=16')
plt.plot(x, phi_plot[3,:], '-g', label='n=64')
plt.plot(x, phi_plot[4,:], '-m', label='n=256')
plt.xlabel('x (m)')
plt.ylabel(r'$\phi$')
plt.figlegend(loc='right', bbox_to_anchor=(0.4,0.2))
plt.grid(b=True, which='major', axis='both')
plt.savefig('HW2/plots/graph_EE_case1.pdf',transparent=True)
#plt.savefig('HW1/plots/graph_case'+str(k+1)+'.svg',transparent=True)
