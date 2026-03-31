import numpy as np
import logging

logger = logging.getLogger(__name__)

def final_label(row):
    # Rule 1: Absolute NDVI threshold
    if row['NDVI'] < 0.3:
        return 0
    
    # Rule 2 + 3: Significant drop + persistence
    if row['ndvi_drop'] >= 0.25:
        if row['next_ndvi'] < row['prev_ndvi'] * 0.75:
            return 0
    
    return 1

def process_group(group):
    group = group.sort_values('date').copy()

    # Lag and lead features
    group['prev_ndvi'] = group['NDVI'].shift(1)
    group['next_ndvi'] = group['NDVI'].shift(-1)

    # NDVI drop (robust to division by zero)
    group['ndvi_drop'] = (group['prev_ndvi'] - group['NDVI']) / (group['prev_ndvi'] + 1e-6)


    group['label'] = group.apply(final_label, axis=1)

    return group


def smooth_labels(group):
    labels = group['label'].values
    
    for i in range(1, len(labels)-1):
        if labels[i-1] == 1 and labels[i] == 0 and labels[i+1] == 1:
            labels[i] = 1
    
    group['label'] = labels
    return group




# Apply once (clean pipeline)
