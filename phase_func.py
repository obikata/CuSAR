import numpy as np
import struct
import re

from global_const import *
from read_bin_as import *
from scipy import interpolate
from get_doppler import *

def phase_func(fp0,fp1,nc,nl,orbit_sense):

    # get global constant
    sol = global_const.SOL

    # get sensor params    
    tmp = np.zeros(6).astype('single')
    ind = [500,934,710,550,742,966]
    for i in range(6):
        fp1.seek(720+ind[i],0)
        tmp[i] = read_bin_as_str(fp1,16)
    wl=tmp[0]
    prf=tmp[1]*1e-3
    prt=1/prf
    fs=tmp[2]*1e6
    k=-tmp[3]
    tau=tmp[4]*1e-6
    theta_azbw=tmp[5]
    dr=sol/(2*fs)

    # calculate r_shift value

    fp0.seek(720+8,0)
    nrec =  int.from_bytes(fp0.read(4), byteorder='big')

    fp0.seek(int(720+288+(64+32+48+96+16+48)/8),0)
    xx = int.from_bytes(fp0.read(2), byteorder='big')
    tad_tmp = np.array([xx],dtype=np.int64)

    i = 1
    while xx != int():
        fp0.seek(int(720+288+(64+32+48+96+16+48)/8+i*nrec),0)
        xx = int.from_bytes(fp0.read(2), byteorder='big')
        if xx == int():
            break
        tad_tmp = np.concatenate((tad_tmp,np.array([xx],dtype=np.int64)),axis=0)
        i = i+1

    tad_tmp = np.mod(tad_tmp,2**11)
    tad2 = tad_tmp/2**10*1023E-6

    fp0.seek(720+116,0)
    sr0_rec=np.array(struct.unpack('>I',fp0.read(4)))

    t_nr = 2*sr0_rec/sol
    pdc = np.floor(t_nr/prt)

    toff = -8.31539*1e-6
    t_nr = pdc*prt+tad2+toff
    sr = sol/2*t_nr

    sr_max = np.max(sr)

    r_shift = np.ceil((sr-sr[0])/dr)

    # get satellite state

    fp1.seek(720+4096+12,0)
    flag = read_bin_as_str(fp1,1)
    if flag == '0':
        print('Loading predicted ephemeris', end="")
    elif flag == '1':
        print('Loading GPS ephemeris', end="")
    elif flag == '2':
        print('Loading precise ephemeris', end="")

    fp1.seek(720+4096+204,0)
    orb_type = read_bin_as_str(fp1,3)
    print(' in ',orb_type, '...', sep='')

    fp1.seek(720+4096+140,0)
    num_state_pts = read_bin_as_int(fp1,4)
    year_obs = read_bin_as_int(fp1,4)
    month_obs = read_bin_as_int(fp1,4)
    day_obs = read_bin_as_int(fp1,4)
    date_elapsed = read_bin_as_int(fp1,4)
    init_time_od_elapsed = read_bin_as_double(fp1,22)
    time_interval = read_bin_as_double(fp1,22)

    state_pts = np.arange(num_state_pts)
    time_od_elapsed = state_pts * time_interval + init_time_od_elapsed # time elapsed from 0h0m0s (UTC) of the obs day

    state_arr_od = np.zeros((num_state_pts,6),dtype='double')
    fp1.seek(720+4096+386,0)
    for i in range(num_state_pts):
        state_arr_od[i,0] = read_bin_as_double(fp1,22)
        state_arr_od[i,1] = read_bin_as_double(fp1,22)
        state_arr_od[i,2] = read_bin_as_double(fp1,22)
        state_arr_od[i,3] = read_bin_as_double(fp1,22)
        state_arr_od[i,4] = read_bin_as_double(fp1,22)
        state_arr_od[i,5] = read_bin_as_double(fp1,22)

    # elapsed time of first record
    fp0.seek(720+36,0)
    data_year = np.array(struct.unpack('>I',fp0.read(4)))
    data_date = np.array(struct.unpack('>I',fp0.read(4)))
    t0 = np.array(struct.unpack('>I',fp0.read(4)))/10**3

    state_function = interpolate.PchipInterpolator(time_od_elapsed,state_arr_od,0)
    state_arr_obs = state_function(t0)
    
    # platform velocity
    pos = np.linalg.norm(state_arr_obs[0,0:3])
    vel = np.linalg.norm(state_arr_obs[0,3:6])

    # nearest range
    fp0.seek(720+116,0)
    r0 = np.array(struct.unpack('>I',fp0.read(4)))
    # reference range (= mid range)
    nref = nc/2
    rref = r0 + nref*(0.5*sol/fs)
    # range vector
    r_full = r0 + np.arange(nc)*(0.5*sol/fs)
    
    # doppler centre & dopper rate
    skip_size = 512
    fp1.seek(720+164,0)
    ellipsoid_model = read_bin_as_str(fp1,5) # Reference ellipsoid model
    doppler_table = get_doppler(fs,wl,state_arr_obs,r0,nc,ellipsoid_model,orbit_sense,skip_size)
    dfaz = prf/nl
    faz = np.reshape(np.arange(0,nl)*dfaz-0.5*prf,(1,nl))
    dc_function = interpolate.PchipInterpolator(doppler_table[:,0],doppler_table[:,3],0)
    dc_full = dc_function(np.arange(nc))
    dr_function = interpolate.PchipInterpolator(doppler_table[:,0],doppler_table[:,4],0)
    dr_full = dr_function(np.arange(nc))

    # azimuth parameters
    az_res = 10
    daz = prt*vel
    laz = int(4*np.ceil(np.max(r_full)*wl/(4*az_res*2*daz)))
    taz = prt*(np.arange(laz))-laz/2*prt
    
    # bulk scaling (referance cell = nc/2)
    rcm_dr=wl**2*(np.tile(faz,(nc,1))-np.tile(dc_full,(nl,1)).T)**2*np.tile(r_full,(nl,1)).T/(8*vel**2)
    rcm=rcm_dr/dr
    rcm1=rcm[int(nc/2),:]
    rcm_dif=rcm_dr-np.tile(rcm_dr[int(nc/2),:],(nc,1))
    rcm1=np.roll(rcm1,int(nl/2))
    phi = np.roll(np.arange(int(-nc/2),int(nc/2))*2*np.pi/nc,int(-nc/2))
    phi = np.tile(phi,(nl,1)).T*np.tile(rcm1,(nc,1))
    bulk = np.asarray(np.cos(phi) + 1j*np.sin(phi), np.complex64)

    # diff scaling (referance cell = nc/2)
    trg = (np.arange(nc)-nc/2)/fs
    sc1 = faz**2
    sc2 = 2*k/sol*wl*(1/(4*dr_full)-1/(4*dr_full[int(nc/2)]))*trg
    sc_arr = np.tile(sc1,(nc,1))*np.transpose(np.tile(sc2,(nl,1)))
    phi = 2*np.pi*sc_arr
    diff = np.asarray(np.cos(phi) + 1j*np.sin(phi), np.complex64)

    ##############################################################################
    ############################## phase function 1 ##############################
    ##############################################################################

    phi1 = np.asarray(np.zeros((nc, nl)) + 1j * np.zeros((nc, nl)), np.complex64)

    phi1 = diff

    ##############################################################################
    ############################## phase function 2 ##############################
    ##############################################################################

    phi2 = np.asarray(np.zeros((nc, nl)) + 1j * np.zeros((nc, nl)), np.complex64)

    # kaiser window
    kaiser_window = np.asarray(np.zeros((nc, 1)), np.float32)
    kaiser_window[0:nc,0] = np.kaiser(nc,3)
    kaiser_window = np.roll(kaiser_window,int(nc/2), axis=0)

    # range matched filter
    nmf = int(tau*fs)
    tmf = np.asarray(np.zeros((nc, 1)) + 1j * np.zeros((nc, 1)), np.complex64)
    dt = 1/fs
    t = np.arange(0,tau,dt)
    tmf[0:nmf,0] = np.asarray(np.cos(-2*np.pi*k/2*(t-0.5*tau)**2) + 1j*np.sin(-2*np.pi*k/2*(t-0.5*tau)**2), np.complex64)
    phi = np.tile(np.fft.fft(np.flip(tmf,axis=0),axis=0),(1,nl))

    # window x rmf x bulk
    phi = np.tile(kaiser_window,(1,nl)) * phi * bulk

    phi2 = np.asarray(phi,dtype='complex64')

    ##############################################################################
    ############################## phase function 3 ##############################
    ##############################################################################

    phi3 = np.asarray(np.zeros((nc, nl)) + 1j * np.zeros((nc, nl)), np.complex64)

    # kaiser window
    kaiser_window = np.asarray(np.zeros((nl, 1)), np.float32)
    kaiser_window = np.kaiser(nl,3)
    kaiser_window = np.roll(kaiser_window,int(nl/2), axis=0)

    # azimuth matched filter
    # phi = np.asarray(np.zeros((nc,nl)), np.complex64)
    # phi[:,slice(laz)] = -2*np.pi*(np.transpose(np.tile(dc_full,(laz,1)))*np.tile(taz,(nc,1))+0.5*np.transpose(np.tile(dr_full,(laz,1)))*np.tile(taz,(nc,1))**2)
    # phi[:,slice(laz)] = np.cos(phi[:,slice(laz)]) + 1j* np.sin(phi[:,slice(laz)])
    # phi = np.fft.fft(phi,axis=1)

    # azimuth matched filter with phase correction
    sc3=-np.roll(sc2,-int(np.ceil(fs*tau)/2))
    phi=np.tile(sc1,(nc,1))*np.transpose(np.tile(1/(2*dr_full)+sc3,(nl,1)))+np.tile((laz*prt)*faz,(nc,1))
    phi = np.exp(2*np.pi*1j*phi)

    # kaiser x azmf_phase
    phi = np.tile(kaiser_window,(nc,1))*phi

    phi3 = np.asarray(phi,dtype='complex64')

    ##########################################################################################################################################

    return phi1, phi2, phi3, laz, r_shift