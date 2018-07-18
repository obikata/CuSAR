import numpy as np

def read_bin_as_str(fp,length):
    oTmp = np.fromfile(fp,np.uint8,length)
    return oTmp.tostring().decode("ascii")

def read_bin_as_double(fp,length):
    oTmp = np.fromfile(fp,np.uint8,length)
    oTmp = oTmp.tostring().decode("ascii")
    return np.double(oTmp)

def read_bin_as_single(fp,length):
    oTmp = np.fromfile(fp,np.uint8,length)
    oTmp = oTmp.tostring().decode("ascii")
    return np.single(oTmp)

def read_bin_as_int(fp,length):
    oTmp = np.fromfile(fp,np.uint8,length)
    oTmp = oTmp.tostring().decode("ascii")
    return np.int(oTmp)
