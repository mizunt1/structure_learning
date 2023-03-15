import numpy as np
import os

path = os.path.expanduser("~/projects/gflownet_sl-main/saved_stuff/posterior_estimate.npy")
full_post = np.load(path)
print("ge")
