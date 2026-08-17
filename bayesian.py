# Func # 4 - atom extraction 

import streamlit as st
from skimage import filters
from skimage.feature import blob_log
from skimage.draw import disk
import numpy as np
from skimage.draw import disk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer
from functools import partial


def extract_atoms_bayesian(image, edge_padding):

    image = image.astype(np.float32)  
    image = filters.gaussian(image, sigma=2.5)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Scale to [0, 1]

    opt_min, opt_max, opt_thresh, opt_rad = auto_tune_image_parameters(image)
    blobs = blob_log(image, min_sigma=opt_min, max_sigma=opt_max, threshold=opt_thresh)
    inside_mask = (
    (blobs[:, 1] > edge_padding) &
    (blobs[:, 0] > edge_padding) &
    (blobs[:, 1] < image.shape[1] - edge_padding) &
    (blobs[:, 0] < image.shape[0] - edge_padding)
    )
    moly_blobs = blobs[inside_mask]


    # transform the sigma number to radius
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    r = np.median(blobs[:,2]) * 2.85

    param_string = f"min_sigma: {opt_min}, max_sigma: {opt_max}, threshold: {opt_thresh}"
    #returns array of blobs, each with x, y, and size sata
    return moly_blobs, r, param_string

#Func #5 -- bayesian optimization
def evaluate_parameters(params, image):
  
    
    # Unpack the parameters the ML is testing
    min_sigma, max_sigma, threshold, radius = params
    
    # 1. Run your exact blob detection
    blobs = blob_log(
        image, 
        min_sigma=min_sigma, 
        max_sigma=max_sigma, 
        num_sigma=10, 
        threshold=threshold
    )
    
    # Penalty if the ML chooses settings that find absolutely no atoms
    if len(blobs) < 5: 
        return 9999.0 
        
    atom_coords = blobs[:, 0:2] # Extract X, Y coordinates
    tree = cKDTree(atom_coords)
    neighbors_list = tree.query_ball_tree(tree, r=radius)

    penalty = atom_fitness(blobs, neighbors_list)
    
 
    return float(penalty)



def atom_fitness(blobs, neighbors_list, image=None):

    radii = blobs[:, 2]

    # ============================
    # 1. Atom size uniformity
    # ============================

    # Ignore extreme outliers (usually white noise blobs)
    q_low, q_high = np.percentile(radii, [10, 90])

    good_radii = radii[
        (radii >= q_low) &
        (radii <= q_high)
    ]

    radius_variation = np.std(good_radii) / np.mean(good_radii)


    # ============================
    # 2. Penalize too-small/large blobs
    # ============================

    median_radius = np.median(good_radii)

    bad_atoms = np.sum(
        (radii < 0.5 * median_radius) |
        (radii > 1.5 * median_radius)
    )

    size_outlier_penalty = bad_atoms / len(radii)


    # ============================
    # 3. Lattice connectivity
    # ============================

    neighbor_counts = np.array(
        [len(n)-1 for n in neighbors_list]
    )

    # prefer roughly 6-fold coordination
    network_penalty = (
        np.mean(neighbor_counts) - 6
    )**2


    isolated_atoms = np.sum(
        neighbor_counts == 0
    )

    isolation_penalty = isolated_atoms / len(radii)


    # ============================
    # Total score
    # ============================

    total_penalty = (
        5 * radius_variation +
        3 * size_outlier_penalty +
        1 * network_penalty +
        5 * isolation_penalty
    )

    return float(total_penalty)


def auto_tune_image_parameters(image):
    """
    Runs a Bayesian Optimization ML loop to find the best parameters 
    uniquely tailored to the provided image.
    """
   
    
    # Define the search space boundaries for the ML to explore
    search_space = [
        Integer(1, 3, name='min_sigma'),
        Integer(4, 8, name='max_sigma'),
        Real(0.001, 0.2, prior='log-uniform', name='threshold'),
        Real(10.0, 60.0, name='radius')
    ]
    
    
    # gp_minimize uses Gaussian Processes (Machine Learning) to smartly guess parameters
    # n_calls=20 means it will try 20 different smart combinations before choosing the best
    result = gp_minimize(
        lambda params: evaluate_parameters(params, image),
        search_space,        
        n_calls=20, 
        random_state=42,
        verbose=False
    )
    
    # Extract the winning parameters
    best_min_sigma, best_max_sigma, best_threshold, best_radius = result.x
    
    print("✅ Optimization Complete!")
    print(f"Best min_sigma: {best_min_sigma}")
    print(f"Best max_sigma: {best_max_sigma}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best Tree Radius: {best_radius:.1f}")
    
    return best_min_sigma, best_max_sigma, best_threshold, best_radius