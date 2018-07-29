import numpy as np

from global_const import *
from scipy.optimize import fsolve
from angle import *

OMG = global_const.OMG
sol = global_const.SOL
mu = global_const.GM

def get_doppler(fs,wl,state0,r0,nc,ellipsoid_model,orbit_sense,skip_size):

    if ellipsoid_model == 'GRS80':
        RA = global_const.RA
        F1 = global_const.F1
<<<<<<< HEAD
        # print('Reference ellipsoid model is ',ellipsoid_model, '...', sep='')
    else:
        # print('Reference ellipsoid model is not found...')
=======
        print('Reference ellipsoid model is ',ellipsoid_model, '...', sep='')
    else:
        print('Reference ellipsoid model is not found...')
>>>>>>> 549cf963b095fe8da148f5bc11bff1f71073c5c6
        return -1

    num_skip_rc = np.floor(np.array(nc,dtype='float64')/skip_size).astype(int)

    # prepare empty arrays
    doppler_table = np.zeros((num_skip_rc,5))

    for i in range(num_skip_rc): # range-wise

        ii = skip_size*i

        r_ecr = state0[0,slice(0,3)]
        v_ecr = state0[0,slice(3,6)]
        omg = np.array([0,0,OMG])

        # state vector in pseudo-ECI
        r_eci = r_ecr
        v_eci = v_ecr+np.cross(omg,r_ecr)

        # ECR to pseudo-ECI (assuming ideal "yaw steering")
        pitch = 0
        roll = 0
        if orbit_sense == 'ASCENDING': # ascending
            yaw = -angle(v_eci,v_ecr)
        elif orbit_sense == 'DESCENDING': # descending
            yaw = angle(v_eci,v_ecr)
        else:
            print('orbit sense is unknown...')

        # satellite frame transfomation
        rsz = np.zeros(3)-r_eci
        rsz = rsz/np.linalg.norm(rsz)
        rsy = np.cross(rsz,v_eci)
        rsy = rsy/np.linalg.norm(rsy)
        rsx = np.cross(rsy,rsz)
        Tsat = np.array([rsx,rsy,rsz])
        Tsat = Tsat.T

        # euler transformation
        ca = np.cos(yaw)
        sa = np.sin(yaw)
        Teuler = np.array([[ca, -sa, 0],[sa, ca, 0],[0, 0, 1]])

        T = np.dot(Tsat,Teuler)
        
        rs = r0+ii*(sol/fs/2)

        lam0,phi0,h_gd0,phi_gc0,h_gc0 =est_geodetic(r_eci,RA,F1)
        h_gd0 = np.array(h_gd0,dtype='float64')
        x0 = np.arccos(h_gd0/rs)
        # h_gc0 = np.array(h_gc0,dtype='float64')
        # x0 = np.arccos(h_gc0/rs)
        alpha = fsolve(offnadir, x0, args=(r_eci,rs,T,RA,F1))

        beam_direction_arr = rs*np.dot(T,np.array([0, np.sin(alpha), np.cos(alpha)]))
        Pt = np.array(r_eci + beam_direction_arr)
        alpha = np.abs(alpha)

        Vt = np.cross(omg,Pt)
        At = np.cross(omg,Vt)

        r = Pt - r_eci
        v = Vt - v_eci
        a = At - -mu*r_eci/np.linalg.norm(r_eci)**3
        
        a1 = np.dot(r,r)
        a2 = np.dot(r,v)
        a3 = np.dot(v,v)
        a4 = np.dot(r,a)

        c2 = -1.082637032e-3
        cpm = np.zeros(3)
        azz = 1-5*(r_eci[0]/np.linalg.norm(r_eci))**2
        azz1 = 3-5*(r_eci[0]/np.linalg.norm(r_eci))**2
        czz = 1-3*c2/2*RA**2/np.linalg.norm(r_eci)**2*azz
        czz1 = 1-3*c2/2*RA**2/np.linalg.norm(r_eci)**2*azz1
        cpm[0:2] = -mu/np.linalg.norm(r_eci)**2*(r_eci[0:2]/np.linalg.norm(r_eci))*czz
        cpm[2] = -mu/np.linalg.norm(r_eci)**2*(r_eci[2]/np.linalg.norm(r_eci))*czz1
        a = At - cpm
        a4 = np.dot(r,a)

        vrel = np.dot((r/rs),v)
        doppler_centre = -2*vrel/wl

        arel = (a1**(-3/2)*a2**2+a1**-0.5*(a3+a4))
        doppler_rate = -2*arel/wl

        doppler_table[i,0] = ii
        doppler_table[i,1] = rs/1000
        doppler_table[i,2] = alpha*180/np.pi
        doppler_table[i,3] = doppler_centre
        doppler_table[i,4] = doppler_rate

    return doppler_table

def offnadir(alpha, *constants):

    r_eci = constants[0]
    rs    = constants[1]
    T     = constants[2]
    RA    = constants[3]
    F1    = constants[4]

    Pt  = r_eci + rs*np.dot(T,np.array([0, np.sin(alpha), np.cos(alpha)]))

    lam,phi,h,phi_gc,h_gc = est_geodetic(Pt,RA,F1)

    # Geodetic
    y = h

    # Geodetic + Geoid (prototype)
    # hGeoid = 36.81
    # y = h - hGeoid

    # Geocentric
    # y = h_gc

    return y

def est_geodetic(r_eci,RA,F1):

    RX=r_eci[0]
    RY=r_eci[1]
    RZ=r_eci[2]
    R=np.linalg.norm(r_eci)

    lamda_gc=np.arctan2(RY,RX)
    phi_gc=np.arctan(RZ/np.sqrt(RX**2+RY**2))

    lam=lamda_gc
    ecc=np.sqrt(1-(1-F1)**2)

    # initial values
    phi=np.arctan(np.tan(phi_gc)/(1-F1)**2)
    N=RA/np.sqrt(1-F1*(2-F1)*np.sin(phi)**2) # auxiliary quantity
    H=R-N
    h_gc = H # geocentric altitude

    K=0 # counter
    dphi=100

    while np.abs(dphi)>1e-15:
        N=RA/np.sqrt(1-ecc**2*np.sin(phi)**2) # improved N
        X=(N+H)*np.cos(lam)*np.cos(phi)
        Y=(N+H)*np.sin(lam)*np.cos(phi)
        Z=((1-F1)**2*N+H)*np.sin(phi)
        dR = np.array([RX, RY, RZ]) - np.array([X, Y, Z])

        # option 2: dZ
        dZ=dR[2]
        dphi=np.arcsin((Z+dZ)/np.sqrt(X**2+Y**2+(Z+dZ)**2))-np.arcsin(Z/np.sqrt(X**2+Y**2+Z**2)) # improved phi
        dH=np.cos(phi)*(dR[0]*np.cos(lam)+dR[1]*np.sin(lam))+np.sin(phi)*dR[2]

        # update
        phi=phi+dphi
        H= H+dH

        # counter
        K=K+1
        if K>=200:
            break

    # update values
    lam=lamda_gc
    h_gd=H

    return lam,phi,h_gd,phi_gc,h_gc
