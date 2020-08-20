import math      as m
import functions as fun
import numpy     as np
import sys

natom     = 64
mass      = 48
lbox      = 4.232317 
rc        = 2.08 
dt        = 0.032
nstep     = 3000
mass_inv  = 1.0/mass
T_i       = 0.728
#T_t       = 0.728
T_t_a     = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5]
for i in range(0,len(T_t_a)):

	r         = np.zeros((natom,3))
	R         = np.zeros(3)
	v         = np.zeros((natom,3))
	f         = np.zeros(3)
	a         = np.zeros((natom,3))
	r         = fun.my_pos_ini_2(natom,lbox)
	v         = fun.my_v_ini(natom, mass, T_i)
	v_ini     = v
	T_t       = T_t_a[i]
	dt_half   = dt * 0.5
	dt_half_2 = dt * dt * 0.5

	print 'Temperature: ' + str(T_t)

	OUTPUT   = open("positions_"     + str(T_t) + ".txt"   , "w")
	OUTPUT1  = open("positions_VMD_" + str(T_t) + ".XYZ"   , "w")
	OUTPUT3  = open("velocities_"    + str(T_t) + ".txt"   , "w")
	OUTPUT2  = open("energy_"        + str(T_t) + ".txt"   , "w")
	OUTPUT4  = open("Temperature_"   + str(T_t) + ".txt"   , "w")

	for t in range(0,nstep):
		if t % 100 == 0:
			print t 

		OUTPUT1.write('{}  \n'.format(natom))
		OUTPUT1.write('{}  \n'.format(t))

		for i in range(0,natom):
			OUTPUT.write('{0:.8f}  ' .format(r[i][0]))
			OUTPUT.write('{0:.8f}  ' .format(r[i][1]))
			OUTPUT.write('{0:.8f}\n' .format(r[i][2]))
			OUTPUT3.write('{0:.8f}  '.format(v[i][0]))
			OUTPUT3.write('{0:.8f}  '.format(v[i][1]))
			OUTPUT3.write('{0:.8f}\n'.format(v[i][2]))

		for i in range(0,natom):
			OUTPUT1.write("C  ")
			OUTPUT1.write('{0:.8f}  '.format(r[i][0]))
			OUTPUT1.write('{0:.8f}  '.format(r[i][1]))
			OUTPUT1.write('{0:.8f}\n'.format(r[i][2]))

		K = fun.my_kinetic_energy(v , mass)

		OUTPUT2.write('{}       '.format(t))
		OUTPUT2.write('{0:.8f}  '.format(fun.my_potential_energy_total(r, lbox, rc)))
		OUTPUT2.write('{0:.8f}\n'.format(K))

		OUTPUT4.write('{}       '.format(t))
		OUTPUT4.write('{0:.8f}\n'.format(fun.my_temperature(K, natom)))

		#Force t
		for i in range(0,natom):
			f = fun.my_force_on(i, r, lbox,rc)
			a[i][0] = f[0] * mass_inv
			a[i][1] = f[1] * mass_inv
			a[i][2] = f[2] * mass_inv

		#Position and half step velocities
		for i in range(0,natom):
			r[i][0] = r[i][0] + v[i][0] * dt + dt_half_2 * a[i][0]
			r[i][1] = r[i][1] + v[i][1] * dt + dt_half_2 * a[i][1]
			r[i][2] = r[i][2] + v[i][2] * dt + dt_half_2 * a[i][2]
			v[i][0] = v[i][0] + a[i][0] * dt_half
			v[i][1] = v[i][1] + a[i][1] * dt_half
			v[i][2] = v[i][2] + a[i][2] * dt_half

		#Force t+dt
		for i in range(0,natom):
			f = fun.my_force_on(i, r, lbox,rc)
			a[i][0] = f[0] * mass_inv
			a[i][1] = f[1] * mass_inv
			a[i][2] = f[2] * mass_inv

		#Velocity
		for i in range(0,natom):
			v[i][0] = v[i][0] + a[i][0] * dt_half
			v[i][1] = v[i][1] + a[i][1] * dt_half
			v[i][2] = v[i][2] + a[i][2] * dt_half

		#PBC
		for i in range(0,natom):
			R[0] = r[i][0]
			R[1] = r[i][1]
			R[2] = r[i][2]
			R = fun.my_pos_in_box(R, lbox)
			r[i][0] = R[0] 
			r[i][1] = R[1]
			r[i][2] = R[2]
		
		#Thermostat
		v = fun.my_anderson_thermostat(v, mass, T_t)

