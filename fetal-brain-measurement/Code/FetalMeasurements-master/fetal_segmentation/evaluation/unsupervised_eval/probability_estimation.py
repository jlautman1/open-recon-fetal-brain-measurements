# some setup
import numpy as np
import nibabel as nib
import scipy.ndimage.filters
from scipy import ndimage
import os
import matplotlib.pyplot as plt
from math import *
from PIL import Image
from skimage.filters import sobel_h, sobel_v
EPSILON = 0.0001


def overlay_image_mask(img, mask, mask2 = None):
    img *= 255.0/img.max()
    img = Image.fromarray(img.astype(np.uint8)).convert("RGBA")
    gt_np = np.zeros([mask.shape[0], mask.shape[1],3], dtype=np.uint8)
    gt_np[:,:,0] = (mask.astype(np.uint8))*255
    if mask2 is not None:
        gt_np[:,:,1] = (mask2.astype(np.uint8))*255
    gt = Image.fromarray(gt_np).convert("RGBA")
    return Image.blend(img, gt, 0.4)


def get_image_gradient_magnitudes(image):
    """
    :param image: A grayscale image
    :return: An image of gradient magnitudes
    """
    dx = sobel_v(image)
    dy = sobel_h(image)
    magnitudes = np.sqrt((dx ** 2) + (dy ** 2))
    return magnitudes


def get_contour_im_from_seg_slice(seg):
    """
    :param seg: A binary image that represents a volumetric segmentation of a slice
    :return: A binary image that represents a contour segmentation of a slice
    """
    gradient_im = get_image_gradient_magnitudes(seg)
    contour_im = (gradient_im > 0).astype(np.int)
    return contour_im

def get_contour(img):
    sx = ndimage.sobel(img,axis=0,mode='constant')
    sy = ndimage.sobel(img,axis=1,mode='constant')
    sobel=np.hypot(sx,sy)
    sobel[sobel > 0 ]=1
    return sobel


def intensity_prior(img, contur): # per slice
    k = 1/16
 #   w = 1.5 #weighting parameter
    w = 2 #weighting parameter
    # sobel_x = k*np.array([[1,1,1,0,1,1,1],[1,2,2,0,-2,-2,-1],[1,2,3,0,-3,-2,-1],[1,2,3,0,-3,-2,-1],
    #                     [1,2,3,0,-3,-2,-1],[1,2,2,0,-2,-2,-1],[1,1,1,0,1,1,1]])
    # sobel_y = k*np.array([[-1,-1,-1,-2,-1,-1,-1],[-1,-1,-1,-2,-1,-1,-1],[-1,-1,-1,-2,-1,-1,-1],[0,0,0,0,0,0,0],
    #                     [1,1,1,2,1,1,1],[1,1,1,2,1,1,1],[1,1,1,2,1,1,1]])
    sx = ndimage.sobel(img,axis=0,mode='constant')
    sy = ndimage.sobel(img,axis=1,mode='constant')
    sp_intensity = np.zeros(img.shape)
    sp_intensity[contur==1]=1

    sigma = np.std(img[contur==1])
    c_points = np.argwhere(contur == 1)
    # get gradients:
    for i in range(len(c_points)):
        y, x = c_points[i]
     #   sub = img[y-3:y+4,x-3:x+4]
        grad_x = np.average(sx[y-3:y+4,x-3:x+4])
        grad_y = np.average(sy[y-3:y+4,x-3:x+4])
        grad_tot = (grad_x**2 + grad_y**2)**0.5
        if abs(grad_tot)<= sigma*w:
            sp_intensity[y,x] = abs(grad_tot)/(sigma*w)
        else:
            sp_intensity[y,x] = 1
    return sp_intensity


def get_intervals(img, contour, T_quality, T_direction):
    R = []
    R_arr = np.zeros(img.shape)
    pts = np.argwhere(contour==1)
    intensity_pr = intensity_prior(img, contour)
    direction = []
    visited = [False]*(len(pts))
    for i in range(len(pts)):
        curr = []
        deg = 0
        y,x = pts[i]
        if not visited[i] and intensity_pr[y,x] < T_quality:
            visited[i] == True
            for j in range(1,T_length):
                if i+j < len(visited):
                    visited[i+j] = True
                    y_2, x_2 = pts[i+j]
                    y_1, x_1 = pts[i+j-1]
                    if((x_2-x_1)!=0):
                        dir_p = (y_2-y_1)/(x_2-x_1)
                    else:
                        dir_p = (y_2-y_1)/EPSILON
                    deg += atan(dir_p)
                    if intensity_pr[y_2,x_2] < T_quality and deg/(j+1) < T_direction:
                        curr.append([y_2,x_2])
                        R_arr[y_2, x_2] = 1
                        continue
                    else:
                        if j != 1:
                            curr.insert(0, [y, x])
                            R.append(np.asarray(curr))
                            R_arr[y, x] = 1
                            direction.append(deg/(j))

                        break

    return R, R_arr, direction


def get_segmentation_variability(img, T_length, intervals, direction, intensity_pr):
    """
    This function performs variability estimation using intensity prior
    :param img:
    :param contour:
    :param T_length:
    :return:
    """
  #  intervals, mask, direction = get_intervals(img, contour)

    new_seg = {}

  #  intensity_pr = intensity_prior(img, cont)
    TSPQ = np.sum(intensity_pr)

    variability_est = np.zeros(img.shape)

    new_seg['mask'] = contour.copy()
    new_seg['segments'] = np.zeros(img.shape)
    direction_lst = [0, np.pi]

    for d in direction_lst:
        for i in range(len(intervals)):

            #print('segments' + str(count))
            curr_mask = contour.copy()
            segments = []
            y_tot = intervals[i][: , 0].copy()
            x_tot = intervals[i][: , 1].copy()

            curr_xy = (y_tot, x_tot)
            curr_mask[y_tot, x_tot] = 0 #This, for the computation of the new TSPQ


            for t in range(1,T_length):
                y_new = np.around(y_tot + t*ceil(sin(direction[i]+ d ))).astype(int)
                x_new = np.around(x_tot + t*ceil(cos(direction[i]+ d ))).astype(int)
                curr_mask[y_new, x_new] = 1
                TSPQ_new = np.sum(intensity_prior(slice, curr_mask))

                if abs(TSPQ-TSPQ_new)< eps:
                    curr_xy = (y_new, x_new)
                    variability_est[y_new, x_new] = 1
                    continue
                else:
                    y_final, x_final = curr_xy
                    segments.append(curr_xy)
                    variability_est[y_final, x_final] = 1
                    new_seg['mask'][y_tot, x_tot] = 0
                    new_seg['mask'][y_final, x_final] = 1
                    new_seg['segments'][y_final, x_final] = 1
                    break

    return new_seg, variability_est
    segment


def get_surrounding_contour_pts(current_point, low_quality_pts, img_shape):
    """
    Checking adjecent low quality pixels 1 pixel away (3*3 neigborhood)
    :param current_point: point od reference
    :param low_quality_pts: points that are quality pixels
    :return:
    """
    [x, y] = current_point
    surrounding_pts = []
    if(x-1>=0):
        if([x-1,y] in low_quality_pts):
            surrounding_pts.append([x-1,y])
        if((y-1>=0) and [x-1,y-1] in low_quality_pts):
            surrounding_pts.append([x-1,y-1])
        if(y+1<=img_shape[1] and [x-1, y+1] in low_quality_pts):
            surrounding_pts.append([x-1, y+1])
    if(y-1>=0 and [x, y-1] in low_quality_pts):
        surrounding_pts.append([x,y-1])
    if(y+1<=img_shape[1] and [x, y+1] in low_quality_pts):
        surrounding_pts.append([x,y+1])

    if(x+1<=img_shape[0]):
        if([x+1,y] in low_quality_pts):
            surrounding_pts.append([x+1,y])
        if(y-1>=0 and [x+1, y-1] in low_quality_pts):
            surrounding_pts.append([x+1,y-1])
        if(y+1<=img_shape[1] and [x+1, y+1] in low_quality_pts):
            surrounding_pts.append([x+1, y+1])

    return surrounding_pts


def pts_direction(curr_point, reference_point):
    """
    Calculate points derivative direction
    :param curr_point: current point
    :param reference_point: reference point
    :return: direction of points
    """
    [x1, y1] = curr_point
    [x2, y2] = reference_point
    if(x2-x1!=0):
        derivative = (y2-y1)/(x2-x1)
    else:
        derivative = (y2-y1)/EPSILON

    return atan(derivative)


def low_quality_segments(intensity_prior_img, T_quality, T_direction):
    """
    Find low quality segments to expand later on
    :param intensity_prior_img: image that is a result of intensity prior calculation
    :param T_quality: Quality threshold
    :param T_direction: Direction threshold
    :return: low quality segments represented by lists of points
    """
    segments = []
    low_quality_natrix = np.matrix(np.where((intensity_prior_img < T_quality) & (intensity_prior_img > 0))).T
    low_quality_pts = low_quality_natrix.tolist()

    while(len(low_quality_pts)>0):
        current_point = low_quality_pts.pop()
        surrounding_pts = get_surrounding_contour_pts(current_point, low_quality_pts, intensity_prior_img.shape)
        if(len(surrounding_pts)==0): #we need at least 2 points to form a segment, single low quality points are not relevant
            continue

        #segment initialization
        segment_direction = pts_direction(current_point, surrounding_pts[0])
        segment_pts = [current_point, surrounding_pts[0]]
        segment_stack = [surrounding_pts[0]]
        segment_directions = [segment_direction]
        low_quality_pts.remove(surrounding_pts[0])

        for i in range(1, len(surrounding_pts)):
            direction = pts_direction(current_point, surrounding_pts[i])
            if(abs(segment_direction - direction)<T_direction):
                segment_pts.append(surrounding_pts[i])
                segment_stack.append(surrounding_pts[i])
                segment_directions.append(direction)
                segment_direction = np.average(segment_directions)
                low_quality_pts.remove(surrounding_pts[i])

        while(len(segment_stack)>0):
            current_point = segment_stack.pop()
            surrounding_pts = get_surrounding_contour_pts(current_point, low_quality_pts, intensity_prior_img.shape)
            if(len(surrounding_pts)==0):
                continue

            for i in range(0, len(surrounding_pts)):
                direction = pts_direction(current_point, surrounding_pts[i])
                if(abs(segment_direction - direction)<T_direction):
                    segment_pts.append(surrounding_pts[i])
                    segment_stack.append(surrounding_pts[i])
                    segment_directions.append(direction)
                    segment_direction = np.average(segment_directions)
                    low_quality_pts.remove(surrounding_pts[i])

        segments.append(segment_pts)

    return segments


if __name__ == '__main__':

    data_path = '/home/bella/Phd/data/brain/TRUFI_axial_siemens/19' #19
    T_direction = np.pi/6 #a direction change threshold
   # T_direction = np.pi/3 #a direction change threshold
    T_quality = 0.7 #a quality threshold above which segmentation point considered of high quality
    T_length = 5
    eps = 0.05

    example_filename = os.path.join(data_path, 'volume.nii')
    img = nib.load(example_filename)
    img_data = img.get_fdata()
    idx = 11
    slice = img_data[:, :, idx]

    truth_path = os.path.join(data_path, 'truth.nii')
    truth_img = nib.load(truth_path)
    truth_data = truth_img.get_fdata()
    slice_truth = truth_data[ :, :,idx]

    # plt.figure(figsize=(15,15))
    # plt.subplot(1,2,1)
    # plt.title('Scan')
    # plt.imshow(slice, 'gray', interpolation='none')
    # plt.subplot(1,2,2)
    # plt.title('Scan with segmentation')
    # pil_im = overlay_image_mask(slice, slice_truth)
    # plt.imshow(np.asarray(pil_im))
    # plt.show()
    contour = get_contour(slice_truth)
    #cont = get_contour_im_from_seg_slice(slice_truth)
    intensity_prior_img = intensity_prior(slice, contour)

    low_gradient = np.zeros(slice.shape)
    low_gradient[intensity_prior_img < T_quality] = 1
 #   low_gradient[intensity_prior_img < 1] = 1
    low_gradient[intensity_prior_img == 0] = 0

    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.title('Scan')
    plt.imshow(slice, 'gray', interpolation='none')
    plt.subplot(1,2,2)
    plt.title('Intensity Prior')
    #pil_im = overlay_image_mask(slice, low_gradient)
    pil_im = overlay_image_mask(slice, contour, low_gradient)
    plt.imshow(np.asarray(pil_im))
    plt.show()

    #intervals, mask, direction = get_intervals(slice, contour, T_quality, T_direction)
    segments = low_quality_segments(intensity_prior_img, T_quality, T_direction)
    mask = low_gradient
    mask[mask>T_quality]=0
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.title('Original scan with contour')
    pil_im = overlay_image_mask(slice, contour)
    plt.imshow(np.asarray(pil_im))
    plt.subplot(1,2,2)
    plt.title('Low quality intervals')
    pil_im = overlay_image_mask(slice, mask)
    plt.imshow(np.asarray(pil_im))

    plt.show()
#
#     res_dict, variability_est = get_segmentation_variability(slice, T_length, intervals, direction, intensity_prior_img)
#
#
#     plt.figure(figsize=(15,15))
#
#     plt.title('New uncertinty intervals at SD + $\pi,0$')
# #    pil_im = overlay_image_mask(slice_normal,cont ,res_dict['segments'])
#  #   pil_im = overlay_image_mask(slice,cont ,variability_est)
#     pil_im = overlay_image_mask(slice,variability_est)
#     plt.imshow(np.asarray(pil_im))
#
#     plt.show()