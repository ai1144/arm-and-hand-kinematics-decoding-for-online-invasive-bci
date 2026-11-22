"""Causal output EMA with per-trial state reset."""
import numpy as np

def smooth_output(values, alpha=1.0):
    values=np.asarray(values,dtype=np.float32)
    out=values.copy()
    for i in range(1,len(values)):
        out[i]=np.float32(alpha)*values[i]+np.float32(1-alpha)*out[i-1]
    return out
