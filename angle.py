import numpy as np

def angle(v1, v2):

    if v1.shape[0] == 3:
        y = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    else:
        num_skip_rc = v1.shape[0]
        num_skip_rl = v1.shape[2]
        y = np.zeros((num_skip_rc,num_skip_rl))
        for j in range(num_skip_rl):
            for i in range(num_skip_rc):
                y[i,j] = np.arccos(np.dot(v1[i,:,j], v2[i,:,j]) / (np.linalg.norm(v1[i,:,j]) * np.linalg.norm(v2[i,:,j])))
    return y
