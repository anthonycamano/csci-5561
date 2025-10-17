from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

### Resources:
## https://www.youtube.com/watch?v=XmO0CSsKg88 - not super helpful and he seemed to exccited for no reason


def get_differential_filter():
    filter_x, filter_y = None, None
    ## Using what i assume is a basic differential filter, maybe switch to sobel later?
    filter_x = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
    ])

    filter_y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
    ])
    
    ## print to make sure we are actually getting correct values
    # print(filter_x)
    # print(filter_y)

    return filter_x, filter_y


def filter_image(image, filter):
    ## gettingimage amd filter dimensions
    m, n = image.shape
    k = filter.shape[0]
    
    ## padding image here with 0s
    pad_size = k // 2
    image_padded = np.pad(image, pad_size, mode='constant', constant_values=0)

    image_filtered = np.zeros((m, n))

    ## copied from lect02 and updated for more readability
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
    grad_angle = np.atan2(image_dy, image_dx) + np.pi;
    
    ## converting to unsigned angles [0, π) as per the hadnout
    grad_angle = grad_angle % np.pi

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    ori_histo = None
    
    m, n = grad_mag.shape
    M = m // cell_size  ## # of cells in y direction
    N = n // cell_size  ## # of cells in x direction
    
    ## histogram tensor: M(vertial) x N(Horixontal) x 6 bins (as mentioned in the handout)
    ori_histo = np.zeros((M, N, 6))
    
    for i in range(M):
        for j in range(N):
            ## define the cell bounds
            y_start = i * cell_size
            y_end = y_start + cell_size
            x_start = j * cell_size
            x_end = x_start + cell_size
            
            ## we need the mag and angle of each cell
            cell_mag = grad_mag[y_start:y_end, x_start:x_end]
            cell_angle = grad_angle[y_start:y_end, x_start:x_end]
            
            ## build histogram with 6 bins
            for bin_idx in range(6):
                ## define angle range for this bin
                if bin_idx == 0:
                    ## first bin: [165°, 180°) ∪ [0°, 15°) - this is the wrap around case
                    angle_min1 = 11 * np.pi / 12  # 165°
                    angle_max1 = np.pi            # 180°
                    angle_min2 = 0                # 0°
                    angle_max2 = np.pi / 12       # 15°
                    
                    mask1 = (cell_angle >= angle_min1) & (cell_angle < angle_max1)
                    mask2 = (cell_angle >= angle_min2) & (cell_angle < angle_max2)
                    mask = mask1 | mask2
                else:
                    ## other bins: [15°, 45°), [45°, 75°), [75°, 105°), [105°, 135°), [135°, 165°)
                    angle_min = (bin_idx * 2 - 1) * np.pi / 12  ## 15°, 45°, 75°, 105°, 135°
                    angle_max = (bin_idx * 2 + 1) * np.pi / 12  ## 45°, 75°, 105°, 135°, 165°
                    mask = (cell_angle >= angle_min) & (cell_angle < angle_max)
                
                ## Sum magnitudes for pixels in this bin
                ori_histo[i, j, bin_idx] = np.sum(cell_mag[mask])

    ## visual check to make sure histogram looks right
    # plt.imshow(ori_histo[:, :, 0], cmap='gray')
    # plt.colorbar()
    # plt.show()

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    ori_histo_normalized = None

    M, N, num_bins = ori_histo.shape
    
    ## calc output dimensions (blocks slide with stride 1)
    ## maybe check to see what effects changing stride has later on or ask in class
    output_M = M - block_size + 1  
    output_N = N - block_size + 1  
    
    ## each block has block_size^2 cells, each with num_bins values
    block_feature_size = block_size * block_size * num_bins
    
    ## init output tensor
    ori_histo_normalized = np.zeros((output_M, output_N, block_feature_size))
    
    ## just setting e as the constant mentioned in the handlout
    e = 0.001
    
    ## Process each of the blocks we have
    for i in range(output_M):
        for j in range(output_N):
            ## extract block: block_size x block_size x num_bins
            block = ori_histo[i:i+block_size, j:j+block_size, :]
            
            ## flatten block into 1D vector (concatenate all histograms) this is the zipping srot of thing from the video
            ## this is the numerator hsubi in the nprmalization formula they gave us
            block_vector = block.flatten()
            
            ## this is the denominator in the normalization formula they gave us
            norm = np.sqrt(np.sum(block_vector**2) + e**2)

            ## normalization: h_normalized = hsubi / sqrt(sum(h^2) + e^2)
            block_normalized = block_vector / norm 
            
            ## store in orihisto_normalized
            ori_histo_normalized[i, j, :] = block_normalized

    return ori_histo_normalized

def extract_hog(image, cell_size=8, block_size=2):
    hog = None
    
    ## following the steps in the handout here - reminder to review this when studying, a lot of good info in hadnout
    ## Convert the grayscale image to float format and normalize to range [0, 1].
    image = image.astype('float') / 255.0

    ## Get differential images using get_differential_filter and filter_image
    filter_x, filter_y = get_differential_filter()
    image_dx = filter_image(image, filter_x)
    image_dy = filter_image(image, filter_y)

    ## Compute the gradients using get_gradient
    grad_mag, grad_angle = get_gradient(image_dx, image_dy)

    ## Build the histogram of oriented gradients for all cells using build_histogram
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)

    ## Build the descriptor of all blocks with normalization using get_block_descriptor
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size)

    ## Return a long vector (hog) by concatenating all block descriptors.
    hog = ori_histo_normalized.flatten()

    return hog

def face_detection(I_target, I_template):
    ## get template(obama) and target(whole image) dimensions
    template_h, template_w = I_template.shape
    target_h, target_w = I_target.shape
    
    ## extract HOG descriptor from template (do this once)
    template_hog = extract_hog(I_template)
    template_hog_normalized = template_hog - np.mean(template_hog)
    
    candidates = []
    
    ## move by 8 pixels at a time (one cell size)
    step_size = 2

    y_range = range(0, target_h - template_h + 1, step_size)
    x_range = range(0, target_w - template_w + 1, step_size)
    
    # Slide template across target image in steps of 8
    for y in y_range:
        for x in x_range:
            ## rxtract patch from target image
            patch = I_target[y:y+template_h, x:x+template_w]
            
            ## extract HOG from patch
            patch_hog = extract_hog(patch)
            patch_hog_normalized = patch_hog - np.mean(patch_hog)
            
            ## calc NCC (normalized cross-correlation) score using formula they gave us
            numerator = np.dot(template_hog_normalized, patch_hog_normalized)
            denominator = (np.linalg.norm(template_hog_normalized) * 
                          np.linalg.norm(patch_hog_normalized))
            
            if denominator > 0:
                ncc_score = numerator / denominator
            else:
                ncc_score = 0
            

            threshold = 0.45
            if ncc_score > threshold :
                candidates.append([int(x), int(y), ncc_score])

    candidates = np.array(candidates)
    # return candidates
    
    ## doing the nms
    bounding_boxes = nms(candidates, template_w, template_h, iou_threshold=0.5)
    
    return bounding_boxes

def face_detection_bonus(I_target, I_template):
    bounding_boxes = None
    ## might skip this
    return  bounding_boxes

# ----- Helper Functions for face detection -----
## this is for non-maximum supression
def nms(candidates, box_w, box_h, iou_threshold=0.5):
    # Sort by confidence score (descending)
    indices = np.argsort(candidates[:, 2])[::-1]
    candidates = candidates[indices]
    
    keep = []
    
    while len(candidates) > 0:
        # Keep the detection with highest score
        current = candidates[0]
        keep.append(current)
        
        if len(candidates) == 1:
            break
        
        # Calculate IoU with remaining detections
        remaining = candidates[1:]
        ious = []
        
        for other in remaining:
            iou = calculate_iou(current[0], current[1], box_w, box_h,
                               other[0], other[1], box_w, box_h)
            ious.append(iou)
        
        # Remove detections with high IoU
        ious = np.array(ious)
        keep_mask = ious < iou_threshold
        candidates = remaining[keep_mask]
    
    return np.array(keep) if keep else np.array([]).reshape(0, 3)

## this is for non-maximum supression - calcs the interesection of union
def calculate_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    # Calculate intersection rectangle
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    # Check if there's no intersection
    if left >= right or top >= bottom:
        return 0.0
    
    # Calculate areas
    intersection_area = (right - left) * (bottom - top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

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
    # image = Image.open('cameraman.tif')
    # image_array = np.array(image)
    # hog = extract_hog(image_array)
    # visualize_hog(image_array, hog, 8, 2)

    # ----- Face Detection -----
    I_target = Image.open('target.png')
    I_target_array = np.array(I_target.convert('L'))

    I_template = Image.open('template.png')
    I_template_array = np.array(I_template.convert('L'))

    bounding_boxes=face_detection(I_target_array, I_template_array)
    visualize_face_detection(I_target, bounding_boxes, I_template_array.shape)