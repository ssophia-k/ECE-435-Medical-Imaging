import matplotlib.pyplot as plt

def display_slice(volume, plane='xy', index=0, display=True, pixel_spacing=0.703, slice_spacing=0.625):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Get the dimensions of the volume
    nx, ny, nz = volume.shape
    img = None

    if plane == 'xy':
        # Fixed Z: Plane is (x, y). imshow needs (y, x) -> .T
        img = volume[:, :, index].T
        extent = [0, nx * pixel_spacing, 0, ny * pixel_spacing]
        
        ax.imshow(img, cmap='gray', origin='lower', extent=extent)
        #ax.set_title(f"Axial slice at z = {index}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    elif plane == 'xz':
        # Fixed Y: Plane is (x, z). imshow needs (z, x) -> .T
        img = volume[:, index, ::-1].T
        extent = [0, nx * pixel_spacing, 0, nz * slice_spacing]
        
        ax.imshow(img, cmap='gray', origin='lower', extent=extent)
        #ax.set_title(f"Coronal slice at y = {index}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")

    elif plane == 'yz':
        # Fixed X: Plane is (y, z). imshow needs (z, y) -> .T
        img = volume[index, :, ::-1].T
        extent = [0, ny * pixel_spacing, 0, nz * slice_spacing]
        
        ax.imshow(img, cmap='gray', origin='lower', extent=extent)
        #ax.set_title(f"Sagittal slice at x = {index}")
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("z (mm)")

    # Force axes and labels to be visible
    ax.set_axis_on()
    plt.tight_layout()
    if display:
        plt.show()
    else:
        plt.close()

    return img