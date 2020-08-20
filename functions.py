import numpy  as np
import math   as m
import random as rand
import sys

sigma   = 1.0
epsilon = 1.0

def my_histogram_distances(dists, nbins, dr):
    counts = np.zeros(nbins, dtype=int)
    n      = len(dists)
    for i in range(0,nbins):
        min_r = i * dr
        max_r = min_r + dr
        for j in range(0,n):
            if (dists[j] <= max_r and dists[j] >= min_r):
                counts[i] = counts[i] + 1    
    return counts

def my_pair_correlation(dists, natom, nbins, dr, lbox):
    gr       = np.zeros(nbins)
    V        = np.zeros(nbins)
    PI       = 4.0 * 3.14159265359 / 3.0
    counts   = np.zeros(nbins, dtype=int)
    n        = len(dists)
    V_max    = (lbox)**3
    N_Max    = natom * (natom - 1.0) / 2.0
    density_IV = V_max / N_Max

    for i in range(0,nbins):
        min_r = i * dr
        max_r = min_r + dr
        V[i]  = PI * (max_r**3 - min_r**3)
        for j in range(0,n):
            if (dists[j] < max_r and dists[j] > min_r):
                counts[i] = counts[i] + 1  
        gr[i] = counts[i] * density_IV / V[i]
    return gr

def my_legal_kvecs(maxn, lbox):
    kvecs = np.zeros(((maxn + 1)**3 , 3))
    PI    = 2.0 * 3.14159265359 / lbox
    index = 0
    for i in range(0,maxn + 1):
        for j in range(0,maxn + 1):
            for k in range(0,maxn + 1):
                kvecs[index][0] = i * PI
                kvecs[index][1] = j * PI
                kvecs[index][2] = k * PI
                index = index + 1
    return np.array(kvecs)

def my_calc_vacf0(all_vel, t):
    n = len(all_vel[0])
    S = 0.0
    for i in range(0,n):
        S = all_vel[0][i][0] * all_vel[t][i][0] + all_vel[0][i][1] * all_vel[t][i][1] + all_vel[0][i][2] * all_vel[t][i][2] + S
    return S / n

def my_diffusion_constant(vacf, dt):
    # dt = 0.032 
    n  =  len(vacf)
    S  = 0.0
    for i in range(1,n):
        S = S + (vacf[i] + vacf[i - 1]) * 0.5 * dt
    return S / 3.0

def my_pos_in_box(pos, lbox):
    lbox_half = lbox * 0.5
    lbox_inv  = 1.0 / lbox
    pos_M = [pos[0] + lbox_half, pos[1] + lbox_half, pos[2] + lbox_half] 

    if pos_M[0] < 0.0:
        n = int(pos_M[0] * lbox_inv)
        pos_M[0] = pos_M[0] - (n - 1.0) * lbox
    if pos_M[0] >= lbox:
        n = int(pos_M[0] * lbox_inv)
        pos_M[0] = pos_M[0] - n * lbox
    if pos_M[1] < 0.0:
        n = int(pos_M[1] * lbox_inv)
        pos_M[1] = pos_M[1] - (n - 1.0) * lbox
    if pos_M[1] >= lbox:
        n = int(pos_M[1] * lbox_inv)
        pos_M[1] = pos_M[1] - n * lbox
    if pos_M[2] < 0.0:
        n = int(pos_M[2] * lbox_inv)
        pos_M[2] = pos_M[2] - (n - 1.0) * lbox
    if pos_M[2] >= lbox:
        n = int(pos_M[2] * lbox_inv)
        pos_M[2] = pos_M[2] - n * lbox
    
    pos_M = [pos_M[0] - lbox_half, pos_M[1] - lbox_half, pos_M[2] - lbox_half] 
    return pos_M

def my_kinetic_energy(vel, mass):
    n = len(vel)
    S = 0.0
    for i in range(0,n):
        S = S + 0.5 * mass * (vel[i][0] * vel[i][0] + vel[i][1] * vel[i][1] + vel[i][2] * vel[i][2])
    return S

def my_potential_energy(rij, rc):
    # epsilon = 1.0
    # # sigma   = 1.0
    n       = len(rij)
    U       = 0.0
    div_c   = sigma / rc
    U_c     = 4.0 * epsilon * (div_c)**6 * (div_c**6 - 1)
    for i in range(0,n):
        for j in range(i + 1,n):
            if rij[i][j]<=rc:
                div     = sigma / rij[i][j]
                U       = 4.0 * epsilon * (div)**6 * (div**6 - 1) + U - U_c
    return U

def my_disp_in_box(drij, lbox):
    drij_M = drij
    lbox_inv  = 1.0 / lbox
    lbox_half = lbox * 0.5

    drij_M[0] = drij_M[0] + lbox_half
    drij_M[1] = drij_M[1] + lbox_half
    drij_M[2] = drij_M[2] + lbox_half

    if drij_M[0] < 0.0:
        n = int(drij_M[0] * lbox_inv)
        drij_M[0] = drij_M[0] - (n - 1.0) * lbox
    if drij_M[0] >= lbox:
        n = int(drij_M[0] * lbox_inv)
        drij_M[0] = drij_M[0] - n * lbox
    if drij_M[1] < 0.0:
        n = int(drij_M[1] * lbox_inv)
        drij_M[1] = drij_M[1] - (n - 1.0) * lbox
    if drij_M[1] >= lbox:
        n = int(drij_M[1] * lbox_inv)
        drij_M[1] = drij_M[1] - n * lbox
    if drij_M[2] < 0.0:
        n = int(drij_M[2] * lbox_inv)
        drij_M[2] = drij_M[2] - (n - 1.0) * lbox
    if drij_M[2] >= lbox:
        n = int(drij_M[2] * lbox_inv)
        drij_M[2] = drij_M[2] - n * lbox
    drij_M[0] = drij_M[0] - lbox_half
    drij_M[1] = drij_M[1] - lbox_half
    drij_M[2] = drij_M[2] - lbox_half
    return drij_M

def my_distance(drij):
    return m.sqrt(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2])

def my_distance_vec(r1,r2):
    return m.sqrt(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2])

def my_force_on(i, pos, lbox,rc):
    force = np.zeros(3)
    rij   = np.zeros(3)
    n     = len(pos)
    # epsilon = 1.0
    # sigma   = 0.10
    epsilon_24 = 24.0 * epsilon

    for j in range(0,n):
        if j != i:
            rij[0] = -pos[i][0] + pos[j][0]
            rij[1] = -pos[i][1] + pos[j][1]
            rij[2] = -pos[i][2] + pos[j][2]
            rij    = my_disp_in_box(rij,lbox)
            dij    = my_distance(rij)
            if dij == 0:
                print dij 
                sys.exit("particles overlap :-(")
            if dij < rc:
                div = (sigma / dij)
                com = -48.0 * div**14 + 24.0 * div**7
                # com = div * div**3 * (2 * div**3 - 1.0) 
                force[0] = force[0] + com * rij[0]
                force[1] = force[1] + com * rij[1]
                force[2] = force[2] + com * rij[2]
    force[0] = force[0] * epsilon
    force[1] = force[1] * epsilon
    force[2] = force[2] * epsilon
    return force

def my_pos_ini(natom,lbox):
    r = np.zeros((natom,3))
    lbox_half = lbox * 0.5
    index = 0
    for i in range(0,natom):
        r[i][0] = rand.uniform(-lbox_half, lbox_half)
        r[i][1] = rand.uniform(-lbox_half, lbox_half)
        r[i][2] = rand.uniform(-lbox_half, lbox_half)
    return r

def my_potential_energy_total(pos, lbox, rc):
    # sigma   = 1.0
    n       = len(pos)
    rij     = np.zeros(3)
    U       = 0.0
    div_c   = sigma / rc
    U_c     = 4.0 * epsilon * (div_c)**6 * (div_c**6 - 1)
    for i in range(0,n):
        for j in range(i + 1,n):
            rij[0] = -pos[i][0] + pos[j][0]
            rij[1] = -pos[i][1] + pos[j][1]
            rij[2] = -pos[i][2] + pos[j][2]
            rij    = my_disp_in_box(rij,lbox)
            dij    = my_distance(rij)
            if dij == 0:
                print dij 
                sys.exit("particles overlap :-(")
            if dij < rc:
                div     = sigma / dij
                U       = 4.0 * epsilon * (div)**6 * (div**6 - 1) + U - U_c
    return U

def my_v_ini(natom, mass, T_f):
    vij   = np.zeros(natom * 3)
    v     = np.zeros((natom,3))
    mu    = 0.0
    Kb    = 1.0
    # Sigma = m.sqrt(Kb * T / mass) 
    Sigma = 1.0
    vij = np.random.normal(mu, Sigma, natom * 3)
    for i in range(0,natom):
        v[i][0] = vij[3 * i]
        v[i][1] = vij[3 * i + 1]
        v[i][2] = vij[3 * i + 2]
    v_ave = my_ave_v(v)
    for i in range(0,natom):
        v[i][0] = v[i][0] - v_ave[0]
        v[i][1] = v[i][1] - v_ave[1]
        v[i][2] = v[i][2] - v_ave[2]
    # v_ave = my_ave_v(v)
    K = my_kinetic_energy(v, mass)
    T = my_temperature(K , natom)
    landa = m.sqrt(T_f / T)
    for i in range(0,natom):
        v[i][0] = v[i][0] * landa
        v[i][1] = v[i][1] * landa
        v[i][2] = v[i][2] * landa
    return v 

def my_temperature(K , natom):
    N  = 3 * natom - 3
    Kb = 1.0
    T = 2.0 * K / (N * Kb)
    return T

def my_pos_ini_2(natom,lbox):
    r = np.zeros((natom,3))
    n = int(natom**(1.0/3.0)) + 1

    delta      = lbox / n
    delta_half = delta * 0.5
    lbox_half  = lbox  * 0.5
    index = 0
    for i in range(0,n):
        x = delta * i
        for j in range(0,n):
            y = delta * j
            for k in range(0,n):
                z = delta * k
                r[index][0] = x - lbox_half + delta_half
                r[index][1] = y - lbox_half + delta_half
                r[index][2] = z - lbox_half + delta_half
                index = index + 1
    return r

def my_ave_v(v):

    S   = np.zeros(3)
    # print S
    n = len(v)
    for i in range(0,n):
        S[0] = S[0] + v[i][0]
        S[1] = S[1] + v[i][1]
        S[2] = S[2] + v[i][2]
    return S/n

def my_anderson_thermostat(v, mass, T):
    mu      = 0.0
    Kb      = 1.0
    nu      = 10
    dt      = 0.01
    check   = nu * dt
    Sigma_t = m.sqrt(Kb * T / mass) 

    for i in range(0,len(v)):
        if rand.random() < check:
            v[i][0] = np.random.normal(mu, Sigma_t, 1)
            v[i][1] = np.random.normal(mu, Sigma_t, 1)
            v[i][2] = np.random.normal(mu, Sigma_t, 1)
    return v

def my_vel_cor(all_vel, taw, nsteps):
    n = len(all_vel[0])
    S = 0.0
    N = 0
    for j in range(0,nsteps):
        t = j + taw
        if t < nsteps:
            for i in range(0,n):
                S = all_vel[j][i][0] * all_vel[t][i][0] + all_vel[j][i][1] * all_vel[t][i][1] + all_vel[j][i][2] * all_vel[t][i][2] + S
                N = N + 1
    return S / N

def my_calc_rhok(kvecs, pos):
    nk = len(kvecs)
    nr = len(pos)
    rhok = np.zeros(nk, dtype=complex)
    for i in range(0,nk):
        S = 0 + 0j
        for j in range(0,nr):
            a = kvecs[i][0] * pos[j][0] + kvecs[i][1] * pos[j][1] + kvecs[i][2] * pos[j][2]
            S = S + np.exp(-a * 1j)
        rhok[i] = S
    return rhok

def my_calc_sk(kvecs, pos):
    nk   = len(kvecs)
    nr   = len(pos)
    sk   = np.zeros(nk)
    rhok = np.zeros(nk, dtype=complex)
    rhok = my_calc_rhok(kvecs, pos)
    for i in range(0,nk):
        sk[i] = (rhok[i] * rhok[i].conjugate()) / nr
    return sk

