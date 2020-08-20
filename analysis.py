import functions as fun
import numpy     as np
import math      as m


natom     = 64
lbox      = 4.232317 
dt        = 0.032
T_t_a     = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5]

nstep     = 1000
nbins     = 100
dr        = 0.02
maxn      =  5
max_dis   = lbox * m.sqrt(3.0)
kvecs     = np.zeros(((maxn + 1)**3 , 3))
S_F       = np.zeros(len(kvecs))    
distances = np.zeros((natom * (natom - 1) * 0.5))
g         = np.zeros(nbins)
d         = np.zeros(3)
for fu in range(0,len(T_t_a)):
	
	T_t      = T_t_a[fu]
	INPUT1  = open("positions_"      + str(T_t) + ".txt" , "r")
	INPUT2  = open("velocities_"     + str(T_t) + ".txt" , "r")
	OUTPUT2 = open("V_correlation_"  + str(T_t) + ".txt" , "w")
	OUTPUT5 = open("pair_cor_"       + str(T_t) + ".txt" , "w")
	OUTPUT6 = open("stru_fac_"       + str(T_t) + ".txt" , "w")

	data1   = INPUT1.read().split()
	data2   = INPUT2.read().split()
	v_all   = np.zeros((nstep,natom,3))
	r_last  = np.zeros((natom,3))
	vacf    = np.zeros((nstep))

	print '1- done with the reading'

	for t in range(nstep - 1,nstep):
		for i in range(0,natom):
			r_last[i][0] = float(data1[t * natom * 3 + i * 3])
			r_last[i][1] = float(data1[t * natom * 3 + i * 3 + 1])
			r_last[i][2] = float(data1[t * natom * 3 + i * 3 + 2])

	counter = 0

	for i in range(0, natom):
		for j in range(i + 1, natom):
			d[0] = r_last[i][0] - r_last[j][0]
			d[1] = r_last[i][1] - r_last[j][1]
			d[2] = r_last[i][2] - r_last[j][2]

			d    = fun.my_disp_in_box(d, lbox)
			distances[counter] = fun.my_distance(d)

			if distances[counter] > max_dis:
				sys.exit("wrong distances :-(")

			counter = counter + 1
	print '2- done with the distances'

	g = fun.my_pair_correlation(distances, natom, nbins, dr, lbox)

	print '3- done with the pair_cor'

	for i in range(0, nbins):
		OUTPUT5.write('{0:.8f}  '.format(i * dr))
		OUTPUT5.write('{0:.8f}\n'.format(g[i]))

	kvecs = fun.my_legal_kvecs(maxn, lbox)
	S_F   = fun.my_calc_sk(kvecs, r_last)

	print '4- done with the S_F'

	for i in range(0, len(kvecs)):
		OUTPUT6.write('{}       '.format(i))
		OUTPUT6.write('{0:.8f}\n'.format(S_F[i]))

# for t in range(0,nstep):
# 	for i in range(0,natom):
# 		v_all[t][i][0] = float(data2[t * natom * 3 + i * 3])
# 		v_all[t][i][1] = float(data2[t * natom * 3 + i * 3 + 1])
# 		v_all[t][i][2] = float(data2[t * natom * 3 + i * 3 + 2])

# for taw in range(0,nstep):
# 	vacf[taw] = fun.my_vel_cor(v_all, taw, nstep)
# 	OUTPUT2.write('{}       '.format(taw))
# 	OUTPUT2.write('{0:.8f}\n'.format(vacf[taw]))

# print '5- done with the vacf'

# print fun.my_diffusion_constant(vacf, dt)
# print v_all
#print r_last

