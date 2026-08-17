
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_atoms(moly_blobs, tungs_blobs, image, style = "dot"):  #input needs to be the outputed value from extract_atoms function
    fig, ax = plt.subplots() 
    ax.imshow(image, cmap='gray')
    
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xticklabels([]), ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_aspect('equal', adjustable='box')
    
    for y, x, r in moly_blobs:        
        if style == "circle": # a different style of output, only if user defines 'circle'
            c = Circle((x, y), r, color='red', linewidth=1.0, fill=False)
            ax.add_patch(c)
        else:
            ax.plot(x, y, 'ro', markersize=0.5)  # red dot at center of perceived Mo
    
    if tungs_blobs not None: 
        for y, x, r in tungs_blobs:
            if style == "circle":
                c = Circle((x, y), r, color='blue', linewidth=1.0, fill=False)
                ax.add_patch(c)
            else:
                ax.plot(x, y, 'bo', markersize=0.5)  # blue dot at center of perceived W     
    return fig, ax