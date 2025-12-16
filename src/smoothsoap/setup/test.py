import random

import numpy as np

A = np.zeros((3,4,5))
print(np.mean(A, axis=1).unsqueeze(1).shape)