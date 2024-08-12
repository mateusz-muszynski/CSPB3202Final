import numpy as np

def saveQValues(self, filename):
    np.savez_compressed(filename, self.qValues)

def load_q_values(filename):
    return np.load(filename, allow_pickle=True).item()
