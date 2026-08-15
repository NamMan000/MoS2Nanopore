
from skimage.feature import blob_log
from skimage.draw import disk
import numpy as np

def extract_atoms_manual(image, edge_padding):

    #Finds the blobs and returns 3-column array of x,y, and sigma, where sigma represents size of blob
    blobs = blob_log(image, min_sigma=1, max_sigma=3, num_sigma=10, threshold=0.05)    
    inside_mask = (
    (blobs[:, 1] > edge_padding) &
    (blobs[:, 0] > edge_padding) &
    (blobs[:, 1] < image.shape[1] - edge_padding) &
    (blobs[:, 0] < image.shape[0] - edge_padding)
    )
    blobs = blobs[inside_mask]

    # transform the sigma number to radius
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    r = np.median(blobs[:,2])

    #calcualtes the average intensity per blob
    avg_intensities = []
    for y, x, r in blobs:
        rr, cc = disk((y, x), r, shape=image.shape)
        avg_intensity = np.mean(image[rr, cc])
        avg_intensities.append(avg_intensity)
    avg_intensities = np.array(avg_intensities)
    
    #differentiates the tunsgten used z-score (standard deviation) test
    z_scores = (avg_intensities - np.mean(avg_intensities)) / np.std(avg_intensities)
    mask = np.abs(z_scores) > 3.0
    tungs_blobs = blobs[mask]
    moly_blobs = blobs[~mask]

    #contamination -- runs a functions to remove neighboring 'tungstens' and reclassify as molybdenum
    keep_w, reclass_mo =  reclassify_w_simple(tungs_blobs)
  
    #returns array of blobs, each with x, y, and size sata
    #return moly_blobs, tungs_blobs

# identifiblobs_w, factor = 5.0):
    factor = 5.0 
    if len(blobs_w) == 0:
        return np.empty((0,3)), np.empty((0,3))
    
    N = len(blobs_w)
    reclass_mask = np.zeros(N, dtype=bool)
    
    for i in range(N):
        y0, x0, r0 = blobs_w[i]
        threshold = factor * r0  # distance threshold

        for j in range(N):
            if i == j:
                continue
            y1, x1, _ = blobs_w[j]

            # Euclidean distance
            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            if dist <= threshold:
                reclass_mask[i] = True
                break  # found neighbor → reclassify

    keep_w = blobs_w[~reclass_mask]
    reclass_mo = blobs_w[reclass_mask]

    moly_blobs = np.vstack([moly_blobs, reclass_mo])
    blobs_w = keep_w

    
    return moly_blobs, tungs_blobs