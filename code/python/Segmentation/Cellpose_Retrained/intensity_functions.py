import numpy as np
from .utils import make_3d

def mean_intensity(I, masks, channel_names=None):
    
    if I.ndim == 2:
        I = make_3d(I)
    
    labels = []
    vals = []
    
    for c in np.unique(masks):
        
        if c != 0:
            
            labels.append(c)
            vals.append(I[:, masks==c].mean(axis=1))          
        
    labels = np.array(labels)
    vals = np.array(vals).T
    
    results = {"Object": labels}
    
    for n, val in enumerate(vals):
        
        if isinstance(channel_names, (list, tuple)):
            results[f"{channel_names[n]}"] = val
            
        else:
            results[f"{n}"] = val
    
    return results


def mean_80_intensity(I, masks, channel_names=None):
    
    if I.ndim == 2:
        I = make_3d(I)
    
    labels = []
    vals = []
    
    for c in np.unique(masks):
        
        if c != 0:
            
            intensity_values = np.sort(I[:, masks==c])
            top80 = int(0.8*intensity_values.shape[1])
            labels.append(c)
            vals.append(intensity_values[:, top80:].mean(axis=1))
        
    labels = np.array(labels)
    vals = np.array(vals).T
    
    results = {"Object": labels}
    
    for n, val in enumerate(vals):
        
        if isinstance(channel_names, (list, tuple)):
            results[f"{channel_names[n]}"] = val
            
        else:
            results[f"{n}"] = val
    
    return results