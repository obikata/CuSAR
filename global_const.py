import numpy as np

class global_const():

    # global
    SOL = 299792458 # Speed of light (m/s)
    OMG = np.radians(360.985647346/86400) # Rotation rate of Earth (rad/s)
    RA = 6378.137*1e+3 # Equatorial radius of Earth (m) (GRS80)
    F1 = 1/298.257222101 # Inverse flattening of Earth (GRS80)
    GM = 398600.4418*1e+9 # Standard gravitational parameter (m**3/s2)