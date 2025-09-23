from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

def get_differential_filter():
    filter_x, filter_y = None, None
    ## Using what i assume is a basic differential filter, maybe switch to sobel later?
    filter_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]]);
    
    filter_y = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])

    return filter_x, filter_y


def filter_image(image, filter):
    image_filtered = None

    ## gettingimage dimensions
    m, n = image.shape
    k = filter.shape[0]  ## square filter here but could be changed later
    
    ## padding image here
    pad_size = k // 2
    image_padded = np.pad(image, pad_size, mode='constant', constant_values=0)

    image_filtered = np.zeros((m, n))

    ## doing convolution step here -- I assume there is a less manual way to do this
    for i in range(m):
        for j in range(n):
            patch = image_padded[i:i+k, j:j+k]
            image_filtered[i, j] = np.sum(patch * filter)

    return image_filtered


def get_gradient(image_dx, image_dy):
    grad_mag, grad_angle = None, None
    ## we start with getting magnitude which is just the pythagorean theorem
    grad_mag = np.sqrt(image_dx**2 + image_dy**2)
    
    ## getting angle so that we can bin it later
    grad_angle = np.arctan2(image_dy, image_dx)
    
    ## convert to unsigned angles [0, π)
    ## Original angle (arctan2) → Unsigned angle (% π)
    ## -π     (-180°) → 0     (0°)
    ## -π/2   (-90°)  → π/2   (90°)
    ## -π/4   (-45°)  → 3π/4  (135°)
    ## 0      (0°)    → 0     (0°)
    ## π/4    (45°)   → π/4   (45°)
    ## π/2    (90°)   → π/2   (90°)
    ## 3π/4   (135°)  → 3π/4  (135°)
    ## π      (180°)  → 0     (0°)
    grad_angle = grad_angle % np.pi

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    ori_histo = None
    
    m, n = grad_mag.shape
    M = m // cell_size  ## # of cells in y direction
    N = n // cell_size  ## # of cells in x direction
    
    # histogram tensor: M x N x 6 bins (as mentioned in the handout)
    ori_histo = np.zeros((M, N, 6))
    
    # Define bin edges for 6 bins covering [0, π)
    # Each bin covers π/6 = 30 degrees
    bin_edges = np.linspace(0, np.pi, 7)  # 7 edges for 6 bins
    
    # Process each cell
    for i in range(M):
        for j in range(N):
            # Define cell boundaries
            y_start = i * cell_size
            y_end = y_start + cell_size
            x_start = j * cell_size
            x_end = x_start + cell_size
            
            # Extract cell data
            cell_mag = grad_mag[y_start:y_end, x_start:x_end]
            cell_angle = grad_angle[y_start:y_end, x_start:x_end]
            
            # Build histogram for this cell
            for bin_idx in range(6):
                # Define angle range for this bin
                if bin_idx == 0:
                    # First bin: [165°, 180°) ∪ [0°, 15°)
                    angle_min1 = 11 * np.pi / 12  # 165°
                    angle_max1 = np.pi            # 180°
                    angle_min2 = 0                # 0°
                    angle_max2 = np.pi / 12       # 15°
                    
                    mask1 = (cell_angle >= angle_min1) & (cell_angle < angle_max1)
                    mask2 = (cell_angle >= angle_min2) & (cell_angle < angle_max2)
                    mask = mask1 | mask2
                else:
                    # Other bins: regular intervals
                    angle_min = (bin_idx * 2 - 1) * np.pi / 12  # 15°, 45°, 75°, 105°, 135°
                    angle_max = (bin_idx * 2 + 1) * np.pi / 12  # 45°, 75°, 105°, 135°, 165°
                    mask = (cell_angle >= angle_min) & (cell_angle < angle_max)
                
                # Sum magnitudes for pixels in this bin
                ori_histo[i, j, bin_idx] = np.sum(cell_mag[mask])

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    ori_histo_normalized = None
    # To do
    return ori_histo_normalized


def extract_hog(image, cell_size=8, block_size=2):
    # convert grey-scale image to double format
    image = image.astype('float') / 255.0
    hog = None
    # To do
    return hog


def face_detection(I_target, I_template):
    bounding_boxes = None
    # To do
    return  bounding_boxes


def face_detection_bonus(I_target, I_template):
    bounding_boxes = None
    # To do
    return  bounding_boxes


# ----- Visualization Functions -----
def visualize_hog(image, hog, cell_size=8, block_size=2, num_bins=6):
    image = image.astype('float') / 255.0
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = image.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def visualize_face_detection(I_target, bounding_boxes, box_size):

    I_target = I_target.convert("RGB")
    ww, hh = I_target.size

    draw = ImageDraw.Draw(I_target)

    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size[1]
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size[0]

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)
        bbox_text = f'{bounding_boxes[ii, 2]:.2f}'
        draw.text((x1 + 1, y1 + 2), bbox_text, fill=(0, 255, 0))

    plt.imshow(np.array(I_target), vmin=0, vmax=1)
    plt.axis("off")
    plt.show()
# ----- Visualization Functions -----


if __name__=='__main__':

    # ----- HOG -----
    image = Image.open('cameraman.tif')
    image_array = np.array(image)
    hog = extract_hog(image_array)
    visualize_hog(image_array, hog, 8, 2)

    # ----- Face Detection -----
    I_target = Image.open('target.png')
    I_target_array = np.array(I_target.convert('L'))

    I_template = Image.open('template.png')
    I_template_array = np.array(I_template.convert('L'))

    bounding_boxes=face_detection(I_target_array, I_template_array)
    visualize_face_detection(I_target, bounding_boxes, I_template_array.shape)