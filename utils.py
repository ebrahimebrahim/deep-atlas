import numpy as np
import matplotlib.pyplot as plt
import monai

def preview_image(image_array, normalize_by = "volume", cmap = None, figsize = (12,12), threshold = None):
    """
    Display three orthogonal slices of the given 3D image.
    
    image_array is assumed to be of shape (L,W,H)
    
    if a number is provided for threshold, then pixels for which the value
    is below the threshold will be colored red
    """
    if normalize_by == "slice" :
        vmin = None
        vmax = None
    elif normalize_by == "volume" :
        vmin = 0
        vmax = image_array.max().item()
    else :
        raise(ValueError(f"Invalid value '{normalize_by}' given for normalize_by"))
    
    # half-way slices
    x,y,z = np.array(image_array.shape)//2
    imgs = (image_array[x,:,:], image_array[:,y,:], image_array[:,:,z])
    
    fig, axs = plt.subplots(1,3,figsize=figsize)
    for ax,im in zip(axs,imgs):
        ax.axis('off')
        ax.imshow(im, origin = 'lower', vmin = vmin, vmax = vmax, cmap=cmap)
        
        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape+(4,)) # RGBA array
            red[im<=threshold] = [1,0,0,1]
            ax.imshow(red, origin = 'lower')
        
    plt.show()


def plot_2D_vector_field(vector_field, downsampling):
    """vector_field should be a tensor of shape (2,L,W)"""
    downsample2D = monai.networks.layers.factories.Pool['AVG',2](kernel_size=downsampling)
    vf_downsampled = downsample2D(vector_field.unsqueeze(0))[0]
    plt.quiver(vf_downsampled[0,:,:], vf_downsampled[1,:,:], angles='xy', scale_units='xy', scale=1);
    

def preview_3D_vector_field(vector_field, downsampling=None):
    """
    Display three orthogonal slices of the given 3D vector field.
    
    vector_field should be a tensor of shape (3,L,W,H)
    
    Vectors are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """
    
    if downsampling is None:
        # guess a reasonable downsampling value to make a nice plot
        downsampling = max(1, int(max(vector_field.shape[1:])) >> 5 )
    
    x,y,z = np.array(vector_field.shape[1:])//2 # half-way slices
    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1); plt.axis('off')
    plot_2D_vector_field(vector_field[[1,2],x,:,:], downsampling)
    plt.subplot(1,3,2); plt.axis('off')
    plot_2D_vector_field(vector_field[[0,2],:,y,:], downsampling)
    plt.subplot(1,3,3); plt.axis('off')
    plot_2D_vector_field(vector_field[[0,1],:,:,z], downsampling)
    plt.show()


def jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.
    
    vf is assumed to be a vector field of shape (3,L,W,H),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.
    
    Returns a numpy array of shape (L,W,H).
    """

    _, L, W, H = vf.shape
    
    # Compute discrete spatial derivatives
    diff_and_trim = lambda array, axis : np.diff(array, axis=axis)[:,:(L-1),:(W-1),:(H-1)]
    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = dx[0]*(dy[1]*dz[2]-dz[1]*dy[2]) - dy[0]*(dx[1]*dz[2]-dz[1]*dx[2]) + dz[0]*(dx[1]*dy[2]-dy[1]*dx[2])

    return det