from PIL import Image
import numpy as np
from cv2 import resize
import matplotlib.pyplot as plt

from cv2 import SIFT_create, KeyPoint_convert, filter2D
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

def find_match(img1, img2):
    x1, x2 = None, None
    dis_thr = 0.7
    
    # create sift object
    sift = SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    kp1 = KeyPoint_convert(kp1)
    kp2 = KeyPoint_convert(kp2)

    # Forward matching: img1 -> img2
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des2)
    distances, indices = neighbors.kneighbors(des1)

    # Backward matching: img2 -> img1 (for cross-check)
    neighbors_back = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des1)
    distances_back, indices_back = neighbors_back.kneighbors(des2)

    good_matches = []

    for i in range(len(distances)):
        dis_closest = distances[i][0]
        dis_second = distances[i][1]
        
        if dis_second > 0 and (dis_closest / dis_second) < dis_thr:
            j = indices[i][0]  # Best match in img2
            
            # Apply Lowe's ratio test for backward match as well
            dis_back_closest = distances_back[j][0]
            dis_back_second = distances_back[j][1]
            
            # Cross-check: verify that img2[j]'s best match is img1[i]
            # AND that backward match also passes ratio test
            if (indices_back[j][0] == i and 
                dis_back_second > 0 and 
                (dis_back_closest / dis_back_second) < dis_thr):
                good_matches.append((i, j))
    
    x1 = np.zeros((len(good_matches), 2))
    x2 = np.zeros((len(good_matches), 2))

    for idx, (i, j) in enumerate(good_matches):
        x1[idx] = kp1[i]
        x2[idx] = kp2[j]

    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    A = None

    # To do

    return A


def warp_image(img, A, output_size):
    img_warped = None

    # To do

    return img_warped


def align_image(template, target, A):
    A_refined = None
    errors = None

    # To do
    
    return A_refined, errors

def track_multi_frames(template, img_list):
    A_list = None
    errors_list = None

    # To do

    return A_list, errors_list

# ----- Visualization Functions -----
def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    plt.show()


def visualize_align_image(template, target, A, A_refined, errors=None):
    import cv2
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list, errors_list=None):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()

    if errors_list is not None:
        for i, errors in enumerate(errors_list):
            plt.plot(errors * 255)
            plt.title(f'Frame {i}')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.show()
# ----- Visualization Functions -----

if __name__=='__main__':

    template = Image.open('template.jpg')
    template = np.array(template.convert('L'))
    
    target_list = []
    for i in range(4):
        target = Image.open(f'target{i+1}.jpg')
        target = np.array(target.convert('L'))
        target_list.append(target)
    
    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    # To do
    ransac_thr = None
    ransac_iter = None
    # ----------
    
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[1], A)
    visualize_align_image(template, target_list[1], A, A_refined, errors)

    A_list, errors_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list, errors_list)