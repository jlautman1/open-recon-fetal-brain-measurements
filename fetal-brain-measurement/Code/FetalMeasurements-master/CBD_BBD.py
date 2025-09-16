import numpy as np
import pandas as pd
import os
import skimage.draw as draw
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.signal as spsignal
import scipy.spatial.distance as spdist
from skimage.morphology import convex_hull_image
from skimage import img_as_float
import scipy.ndimage as spimage
import statsmodels.graphics.agreement as blandaltman
import skimage
from scipy.spatial import ConvexHull , distance


def slop_of_line(point1, point2, ):
    m_x = point1[0]- point2[0]
    m_y = point1[1] - point2[1]
    m_mag = np.sqrt(m_x * m_x + m_y * m_y)
    m_y = m_y / m_mag
    m_x = m_x / m_mag

    return m_x, m_y


def find_symmetry_line(step_size, locUp, locDown, rrUp, ccUp,rrDown,ccDown, img, orig_img):



    Score = np.zeros((2 * step_size, 2 * step_size))
    point_a = np.zeros(2,dtype=int)
    point_b = np.zeros(2,dtype=int)

    for i in range(-step_size, step_size):
        point_a[0] = rrUp[locUp + i]
        point_a[1] = ccUp[locUp + i]
        for l in range(-step_size, step_size):
            c_img = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
            point_b[0] = rrDown[locDown + l]
            point_b[1] = ccDown[locDown + l]
            rr, cc = draw.line(point_b[1], point_b[0], point_a[1], point_a[0])
            c_img[cc, rr] = 1
            s_img = img * c_img[0:img.shape[0], 0:img.shape[1]]


            Score[i + step_size, l + step_size] = np.sum(s_img)

    return Score


def line_rect_intersection(line1, rect):
    x_diff = (line1[0][0] - line1[1][0], rect[0][0] - rect[1][0])
    y_diff = (line1[0][1] - line1[1][1], rect[0][1] - rect[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        return None

    d = (det(*line1), det(*rect))
    x = int(round(det(d, x_diff) / div))
    y = int(round(det(d, y_diff) / div))

    if rect[0][0] == rect[1][0]:
        if y < max(rect[0][1], rect[1][1]) and y > min(rect[0][1], rect[1][1]):
            return x, y
        else:
            return None

    if rect[0][1] == rect[1][1]:
        if x < max(rect[0][0], rect[1][0]) and x > min(rect[0][0], rect[1][0]):
            return x, y
        else:
            return None
    return


def mid_line_orientation(mid_up, mid_down, orig_img):

    r1 = ((0, 0), (orig_img.shape[0], 0))
    r2 = ((0, orig_img.shape[1]), (orig_img.shape[0], orig_img.shape[1]))
    r3 = ((0, 0), (0, orig_img.shape[1]))
    r4 = ((orig_img.shape[0], 0), (orig_img.shape[0], orig_img.shape[1]))

    mid_pts = ((mid_up), (mid_down))

    inter1 = line_rect_intersection(mid_pts, r1)
    inter2 = line_rect_intersection(mid_pts, r2)
    inter3 = line_rect_intersection(mid_pts, r3)
    inter4 = line_rect_intersection(mid_pts, r4)

    if inter1 != None and inter2 != None:
        if spdist.euclidean(inter1, mid_up) > spdist.euclidean(inter1, mid_down):
            mid_down = inter1
            mid_up = inter2
            rr_down, cc_down = draw.line(0, 0, orig_img.shape[0], 0)
            rr_up, cc_up= draw.line(0, orig_img.shape[1], orig_img.shape[0], orig_img.shape[1])
        else:
            mid_up = inter1
            mid_down = inter2
            rr_up, cc_up = draw.line(0, 0, orig_img.shape[0], 0)
            rr_down, cc_down = draw.line(0, orig_img.shape[1], orig_img.shape[0], orig_img.shape[1])
    elif inter3 != None and inter4 != None:
        if spdist.euclidean(inter3, mid_up) > spdist.euclidean(inter3, mid_down):
            mid_down = inter3
            mid_up = inter4
            rr_down, cc_down = draw.line(0, 0, 0, orig_img.shape[1])
            rr_up, cc_up = draw.line(orig_img.shape[0], 0, orig_img.shape[0], orig_img.shape[1])
        else:
            mid_up = inter3
            mid_down = inter4
            rr_up, cc_up = draw.line(0, 0, 0, orig_img.shape[1])
            rr_down, cc_down = draw.line(orig_img.shape[0], 0, orig_img.shape[0], orig_img.shape[1])
    else:
        return 0,0,0,0, mid_up, mid_down
    return rr_up, cc_up, rr_down, cc_down, mid_up, mid_down


def optimize_MSL(MSL_up, MSL_down, orig_img, seg_img, step_size=10):
    rr_up, cc_up, rr_down, cc_down, MSL_up_new, MSL_down_new = mid_line_orientation(MSL_up, MSL_down, orig_img)
    a = (rr_up == MSL_up_new[0])
    b = (cc_up == MSL_up_new[1])
    c = (rr_down == MSL_down_new[0])
    d = (cc_down == MSL_down_new[1])

    loc_up = np.where(np.logical_and(a, b))[0][0]
    loc_down = np.where(np.logical_and(c, d))[0][0]

    Score = find_symmetry_line(step_size, loc_up, loc_down, rr_up, cc_up, rr_down, cc_down, seg_img, orig_img)

    loc_min = np.stack(np.where(Score == Score.min())).T
    'choosing one of the min locations. not sure if this is the right option ??? '
    locMin = loc_min[0, :]

    new_MSL_up = np.zeros_like(MSL_up_new)
    new_MSL_down = np.zeros_like(MSL_up_new)

    new_MSL_down[0] = rr_down[loc_down + locMin[1] - step_size]
    new_MSL_down[1] = cc_down[loc_down + locMin[1] - step_size]

    new_MSL_up[0] = rr_up[loc_up + locMin[0] - step_size]
    new_MSL_up[1] = cc_up[loc_up + locMin[0] - step_size]

    MSL = np.stack((draw.line(new_MSL_up[0], new_MSL_up[1], new_MSL_down[0], new_MSL_down[1]))).T

    return new_MSL_up, new_MSL_down, MSL

def mid_line_Brain(seg_img, MSL_up, MSL_down, plot=False):
    if MSL_up[0] == MSL_down[0] and MSL_up[1] == MSL_down[1]:
        return (0, 0), (0, 0), 0, 0

    mid_line_brain = []

    m_MSL_x, m_MSL_y = slop_of_line(MSL_up, MSL_down)
    mid_line = np.stack((draw.line(MSL_up[0], MSL_up[1], MSL_down[0], MSL_down[1]))).T

    chull = convex_hull_image(seg_img)
    img_bool = np.zeros_like(seg_img, dtype=bool)
    img_bool[np.where(seg_img == 1)] = 'True'
    chull_diff = img_as_float(chull.copy())
    chull_diff[img_bool] = 2

    for i in range(1, mid_line.shape[0] - 1):
        if chull[mid_line[i, 0], mid_line[i, 1]] == True:
            mid_line_brain.append(mid_line[i])

    mid_line_brain = np.array(mid_line_brain)

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(chull_diff, cmap='gray')
        ax.set_title('Difference')

    return mid_line_brain, m_MSL_x, m_MSL_y


def perpendicular_line(mid_line_brain, seg_img, MSL_down, m_MSL_y, m_MSL_x, resX, resY):
    roi_width = int(seg_img.shape[0] / 3)
    temp_CBD_left = np.zeros((2), dtype=int)
    temp_CBD_right = np.zeros((2), dtype=int)
    m_perp_x = -m_MSL_y
    m_perp_y = m_MSL_x

    temp_CBD_image = np.zeros_like(seg_img)
    temp_CBD_left[0] = int(mid_line_brain[0] + m_perp_x * (2*roi_width))
    temp_CBD_left[1] = int(mid_line_brain[1] + m_perp_y * (2*roi_width))
    temp_CBD_right[0] = int(mid_line_brain[0] + m_perp_x * -(2*roi_width))
    temp_CBD_right[1] = int(mid_line_brain[1] + m_perp_y * -(2*roi_width))

    if temp_CBD_left[0] >= seg_img.shape[0]:
        temp_CBD_left[0] = seg_img.shape[0]-1
    if temp_CBD_left[0] <= 0:
        temp_CBD_left[0] = 1
    if temp_CBD_right[0] >= seg_img.shape[0]:
        temp_CBD_right[0] = seg_img.shape[0]-1
    if temp_CBD_right[0] <= 0:
        temp_CBD_right[0] =1
    if temp_CBD_left[1] >= seg_img.shape[1]:
        temp_CBD_left[1] = seg_img.shape[1]-1
    if temp_CBD_left[1] <= 0:
        temp_CBD_left[1] = 1
    if temp_CBD_right[1] >= seg_img.shape[1]:
        temp_CBD_right[1] = seg_img.shape[1]-1
    if temp_CBD_right[1] <= 0:
        temp_CBD_right[1] = 1

    rr, cc = draw.line(temp_CBD_left[0], temp_CBD_left[1], temp_CBD_right[0], temp_CBD_right[1])
    temp_CBD_image[rr, cc] = 1
    cross_img = seg_img * temp_CBD_image

    '''calculating the real length of CBD with no impact of the ventricles'''

    CBD_coor_temp = np.transpose(np.nonzero(cross_img))
    if CBD_coor_temp.shape[0] < 2:
        return 0,0,0

    cross_product = np.zeros(CBD_coor_temp.shape[0])

    for point in range(CBD_coor_temp.shape[0]):
        cross_product[point] = np.cross((MSL_down-mid_line_brain), (CBD_coor_temp[point]-mid_line_brain))

    CBD_left_temp = np.squeeze(CBD_coor_temp[np.where(cross_product == cross_product.max())])
    CBD_right_temp = np.squeeze(CBD_coor_temp[np.where(cross_product == cross_product.min())])
    del_x = CBD_left_temp[0] - CBD_right_temp[0]
    del_y = CBD_left_temp[1] - CBD_right_temp[1]
    CBD_length = np.sqrt((resX * del_x) ** 2 + (resY * del_y) ** 2)

    return CBD_length, CBD_left_temp, CBD_right_temp


def CBD_points(seg_img, mid_up, mid_down, resX, resY, CBD_min_th):
    '''
    :param seg_img: segmentation of brain (slice of nii)
    :param mid_up: MSL upper point (int)
    :param mid_down: MSL down point (int)
    :param resX:
    :param resY:
    :return: CBD left amd right points, slope in x and y
    '''

    mid_line_brain,m_MSL_x, m_MSL_y = mid_line_Brain(seg_img, mid_up, mid_down)
    edge1 = (mid_line_brain[0][0], mid_line_brain[0][1])
    CBD_length = np.zeros((mid_line_brain.shape[0]))

    for i in range(mid_line_brain.shape[0]-1):
        CBD_length[i], __, __= perpendicular_line(mid_line_brain[i], seg_img, mid_down, m_MSL_y, m_MSL_x, resX, resY)


    # Sylvian fissure
    local_minima = []
    minima_loc = spsignal.find_peaks(CBD_length * (-1), plateau_size=(1, 3))
    minima_loc_filter = []
    for i in range(len(minima_loc[0])):
        if CBD_length[minima_loc[0][i]] > CBD_min_th:
            minima_loc_filter.append(minima_loc[0][i])
            local_minima.append(CBD_length[minima_loc[0][i]])

    center_mass = spimage.measurements.center_of_mass(seg_img)
    dist = []
    if len(local_minima)==0:
        raise Exception('Can not solve')

    for i in range(len(minima_loc_filter)):
        dist.append(spdist.euclidean(center_mass, mid_line_brain[minima_loc_filter[i]]))

    silvian_loc = minima_loc_filter[dist.index(min(dist))]

    if spdist.euclidean(edge1, mid_up) < spdist.euclidean(edge1, mid_down):
            CBD_length_silv = CBD_length[0:silvian_loc]
            loc_CBD = np.stack(np.where(CBD_length_silv == CBD_length_silv.max())).T.max()

    else:
            CBD_length_silv = CBD_length[silvian_loc:]
            loc_CBD = np.stack(np.where(CBD_length_silv == CBD_length_silv.max())).T.min()

    CBD , CBD_left, CBD_right = perpendicular_line(mid_line_brain[loc_CBD], seg_img, mid_down, m_MSL_y, m_MSL_x, resX, resY)

    RGB_img3 = cv2.cvtColor(np.uint8(seg_img * 255.), cv2.COLOR_GRAY2BGR)

    cv2.line(RGB_img3, (CBD_left[1], CBD_left[0]), (CBD_right[1], CBD_right[0]), (255, 0, 0), 1)
    cv2.line(RGB_img3, (mid_up[1], mid_up[0]), (mid_down[1], mid_down[0]), (255, 0, 0), 1)
    #showing it didn't work, saving it instead:
    #cv2.imshow("CBD", RGB_img3)
    cv2.imwrite("/workspace/output/debug_cbd.png", RGB_img3)


    return CBD, CBD_left, CBD_right

def profile_BBD(orig_img,rr_left, cc_left, rr_right, cc_right, BBD_left_line, BBD_right_line, resX, resY, BBD_th, plot = False):

    profile_extrema_left = []
    profile_extrema_right = []
    BBD_left = np.zeros((2), dtype=int)
    BBD_right = np.zeros((2), dtype=int)


    if plot:
        plt.imshow(orig_img, cmap='gray')
        plt.plot([BBD_left[1], BBD_right[1]], [BBD_left[0], BBD_right[0]], linewidth=1, color='blue',linestyle='dashed')

    left_profile = orig_img[tuple(tuple(a) for a in BBD_left_line.T)][:15]
    right_profile = orig_img[tuple(tuple(a) for a in BBD_right_line.T)][:15]

    extrema_deriv_left,_ = spsignal.find_peaks(np.gradient((left_profile*(-1)),1),plateau_size = (1,3))
    for i in range(extrema_deriv_left.shape[0]-1):
        if np.gradient(left_profile, 1)[extrema_deriv_left[i]] > BBD_th:
            if np.gradient(left_profile, 1)[extrema_deriv_left[0]] > np.gradient(left_profile, 3)[extrema_deriv_left[i+1]]:
                extrema_deriv_left[0] = extrema_deriv_left[i+1]
    extrema_deriv_right,_ = spsignal.find_peaks(np.gradient((right_profile*(-1)),1),plateau_size = (1,3))
    for i in range(extrema_deriv_right.shape[0]-1):
        if np.gradient(right_profile, 1)[extrema_deriv_right[i]] > BBD_th:
            if np.gradient(right_profile, 1)[extrema_deriv_right[0]] > np.gradient(right_profile, 3)[extrema_deriv_right[i+1]]:
                extrema_deriv_right[0] = extrema_deriv_right[i+1]
    if plot:
        plt.figure()
        plt.imshow(np.stack([left_profile, right_profile]))
        plt.show()

    # extrema_deriv_left[0]= extrema_deriv_left[0]+1
    # extrema_deriv_right[0] = extrema_deriv_right[0]+1

    BBD_left[0] = rr_left[extrema_deriv_left[0]]
    BBD_left[1] = cc_left[extrema_deriv_left[0]]
    BBD_right[0] = rr_right[extrema_deriv_right[0]]
    BBD_right[1] = cc_right[extrema_deriv_right[0]]

    profile_extrema_left.append(extrema_deriv_left[0])
    profile_extrema_right.append(extrema_deriv_right[0])

    del_x = BBD_left[0] - BBD_right[0]
    del_y = BBD_left[1] - BBD_right[1]

    BBD = np.sqrt((resX * del_x) ** 2 + (resY * del_y) ** 2)

    return BBD, BBD_left, BBD_right

def BBD_points(orig_img, CBD_left, CBD_right, mid_up, mid_down, resX, resY, plot=False, BBD_th=-150, step_size=20):
    '''
    :param orig_img: MRI scan. (slice of nii)
    :param CBD_left: CBD left point
    :param CBD_right: CBD right point
    :param plot: plotting in the function. default False
    :param BBD_th: derivative threshold for local minima. default is -100
    :param step_size: expansion size to each size, from CBD points
    :return: left and right expansion to BBD, left and right BBD points, derivative of the expansion
    '''

    BBD_val = True

    max_img = np.max(orig_img)
    crop_img_clahe = skimage.exposure.equalize_adapthist(orig_img / max_img, kernel_size=(20, 20), clip_limit=0.01, nbins=256) * max_img

    BBD_left = np.zeros((2), dtype=int)
    BBD_right = np.zeros((2), dtype=int)

    m_MSL_x, m_MSL_y = slop_of_line(mid_up, mid_down)
    m_BBD_x = -m_MSL_y
    m_BBD_y = m_MSL_x

    BBD_left[0] = int(CBD_left[0] - m_BBD_x * step_size)
    BBD_left[1] = int(CBD_left[1] - m_BBD_y * step_size)
    BBD_right[0] = int(CBD_right[0] + m_BBD_x * step_size)
    BBD_right[1] = int(CBD_right[1] + m_BBD_y * step_size)

    rr_left, cc_left = draw.line(CBD_left[0], CBD_left[1], BBD_left[0], BBD_left[1])
    rr_right, cc_right = draw.line(CBD_right[0], CBD_right[1], BBD_right[0], BBD_right[1])

    BBD_left_line = np.stack((rr_left, cc_left)).T
    BBD_right_line = np.stack((rr_right, cc_right)).T

    BBD, BBD_left, BBD_right = profile_BBD(orig_img,rr_left, cc_left, rr_right, cc_right, BBD_left_line, BBD_right_line, resX, resY, BBD_th)
    BBD_clahe, BBD_left_clahe, BBD_right_clahe = profile_BBD(crop_img_clahe,rr_left, cc_left, rr_right, cc_right, BBD_left_line, BBD_right_line, resX, resY, BBD_th)

    if np.abs(BBD_clahe-BBD) > 3:
        BBD_val = False

    if plot:
        plt.figure()
        plt.imshow(orig_img, cmap='gray')
        plt.plot([BBD_left[1], BBD_right[1]], [BBD_left[0], BBD_right[0]], linewidth=1, color='red', linestyle='dashed')
        plt.show()

        plt.figure()
        plt.imshow(crop_img_clahe, cmap='gray')
        plt.plot([BBD_left_clahe[1], BBD_right_clahe[1]], [BBD_left_clahe[0], BBD_right_clahe[0]], linewidth=1, color='red', linestyle='dashed')
        plt.show()


    return BBD, BBD_left, BBD_right, BBD_val

def checkangle(x, y, eps=13):
    diff = abs(x - y) % 180
    diff = min(diff, 180 - diff)
    return diff < eps


def find_max_in_hull(hull, points):
    max_dist = 0
    max_pts = []
    for x in hull.vertices:
        x_p = points[x]
        for y in hull.vertices:
            y_p = points[y]
            dist = distance.euclidean(x_p, y_p)
            if dist > max_dist:
                max_pts = np.array([x_p, y_p])
                max_dist = dist
    return max_pts


def measure_and_show_cerebellum(cerebellum, res_px, plot=False, overlay=None, img=None):
    # cerebellum = find_cerebellum(overlay)
    xxx = np.where(cerebellum != 0)
    points = np.array([xxx[1], xxx[0]]).T
    #print(f"[DEBUG] Non-zero cerebellum points found: {points.shape[0]}")

    if points.shape[0] < 5:
        print("Skip")
        #print("[DEBUG] Too few points for convex hull — skipping.")
        return
    hull = ConvexHull(points)
    # try:
    #     hull = ConvexHull(points)
    # except Exception as e:
    #     print(f"[ERROR] Failed to compute convex hull: {e}")
    #     return
    if plot:
        plt.imshow(img, cmap='gray')
        plt.imshow(overlay, alpha=.2, cmap="PuBu")
        plt.imshow(cerebellum, alpha=.4, cmap="Reds")
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    pt_hull = find_max_in_hull(hull, points)
    pt_hull = (pt_hull.T[0], pt_hull.T[1])
    #print(f"[DEBUG] Hull endpoints for TCD: {pt_hull}")

    box = cv2.minAreaRect(np.int0(points))
    box = cv2.boxPoints(box)
    diff = (box[2] - box[1]) / 2.
    movbox = box + diff
    pt_box1 = (movbox.T[0, 0:2], movbox.T[1, 0:2])

    diff2 = (box[1] - box[0]) / 2.
    movbox2 = box - diff2
    pt_box2 = (movbox2.T[0, 1:3], movbox2.T[1, 1:3])

    if plot:
        plt.plot(pt_hull[0], pt_hull[1], 'b-')
        plt.plot(pt_box1[0], pt_box1[1], 'g-')
        plt.plot(pt_box2[0], pt_box2[1], 'r-')
        # print(f"[DEBUG] Hull points: {pt_hull}")
        # print(f"[DEBUG] Box1 points: {pt_box1}")
        # print(f"[DEBUG] Box2 points: {pt_box2}")
        # print(f"[DEBUG] pt_angle = {pt_angle}")

        # for i, (p1, p2) in enumerate(([0, 1], [0, 2])):
        #     print(f"[DEBUG] Pair {i}: angle diff = {abs(pt_angle[p1] - pt_angle[p2])}, checkangle = {checkangle(pt_angle[p1], pt_angle[p2])}")

    pt_all = pt_hull, pt_box1, pt_box2
    pt_angle = []
    for pt in pt_all:
        pt_angle.append(np.rad2deg(np.arctan2((pt[1][1] - pt[1][0]), (pt[0][1] - pt[0][0]))))
    # print(f"[DEBUG] Angles (hull, box1, box2): {pt_angle}")

    # print(f"[DEBUG] Hull points: {pt_hull}")
    # print(f"[DEBUG] Box1 points: {pt_box1}")
    # print(f"[DEBUG] Box2 points: {pt_box2}")
    # print(f"[DEBUG] pt_angle = {pt_angle}")

    # for i, (p1, p2) in enumerate(([0, 1], [0, 2])):
    #     print(f"[DEBUG] Pair {i}: angle diff = {abs(pt_angle[p1] - pt_angle[p2])}, checkangle = {checkangle(pt_angle[p1], pt_angle[p2])}")

    for pair in ([0, 1], [0, 2]):
        pt_hull, pt_box = pt_all[pair[0]], pt_all[pair[1]]
        pt_hull_p = np.array(pt_hull).T[::-1, ::-1]
        tcd_meas = distance.euclidean(pt_hull_p[0], pt_hull_p[1]) * res_px
        angle_ok = checkangle(pt_angle[pair[0]], pt_angle[pair[1]])
        #print(f"[DEBUG] Pair {pair}: TCD={tcd_meas:.2f} mm, angle match={angle_ok}")
        if angle_ok:
            return pt_hull_p, tcd_meas, True
    #print("[DEBUG] No valid angle alignment found — TCD marked as invalid.")
    return pt_hull_p, tcd_meas, False


def TCD_points(subseg_img, mid_up, mid_down, resX, resY, plot=False):
    mid_line_brain, m_MSL_x, m_MSL_y = mid_line_Brain(subseg_img, mid_up, mid_down)
    #print(f"[DEBUG] mid_line_brain has {mid_line_brain.shape[0]} points")

    TCD_length = np.zeros((mid_line_brain.shape[0]))
    for i in range(mid_line_brain.shape[0] - 1):
        TCD_length[i], _, _ = perpendicular_line(mid_line_brain[i], subseg_img, mid_down, m_MSL_y, m_MSL_x, resX, resY)

    if mid_line_brain.shape[0]==0:
        #print("[DEBUG] No mid-line points found — returning invalid TCD")
        return None, None, None, False

    TCD_pts, TCD, TCD_valid = measure_and_show_cerebellum(subseg_img, resX, plot, subseg_img, subseg_img)
    #print(f"[DEBUG] TCD = {TCD}, TCD_pts = {TCD_pts}, Valid = {TCD_valid}")
    return TCD, TCD_pts[0], TCD_pts[1], TCD_valid

def load_nii(Data, index, DIR):

    elem_fname = Data[Data.columns[0]][index]
    file_name = os.path.join(DIR,elem_fname)

    segmented = nib.load(os.path.join(file_name, "subseg.nii.gz")).get_fdata()
    seg_img = segmented[:,:, Data['LiatBBDSelection'][index]-1]

    original = nib.load(os.path.join(file_name, "cropped.nii.gz")).get_fdata()
    orig_img = original[:,:, Data['LiatBBDSelection'][index]-1]

    return elem_fname, seg_img, orig_img


if __name__ == "__main__":

    BASE_DIR = r'S:\Bossmat\results\03_03_bbdimprove'
    DIR = r'S:\Netanell\DemoCode_01_03'
    Data = pd.read_excel(os.path.join(DIR,'Analysis1.xlsx'), sheet_name= 'Sheet1')

    th = 0.5
    CBD_model = []
    dif_CBD = []
    BBD_model = []
    dif_BBD = []
    TCD_model = []
    dif_TCD = []
    m_bbd_model = []
    BBD_valid = []

    # index = 44

    file_name = Data.columns[0]

    for index in Data.index:

        elem_fname, seg_img, orig_img = load_nii(Data, index, DIR)
        img2 = np.uint8(seg_img * 255.)
        RGB_img = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        MSL_up_CBD = np.array([eval(Data.msl_points[index])[str(Data.LiatBBDSelection[index]-1)][0][0],
                               eval(Data.msl_points[index])[str(Data.LiatBBDSelection[index]-1)][0][1]], int)

        MSL_down_CBD = np.array([eval(Data.msl_points[index])[str(Data.LiatBBDSelection[index]-1)][1][0],
                                 eval(Data.msl_points[index])[str(Data.LiatBBDSelection[index]-1)][1][1]], int)
        resX = eval(Data.Resolution[index])[0]
        resY = eval(Data.Resolution[index])[1]

        # new_MSL_up, new_MSL_down, MSL = optimize_MSL(MSL_up_CBD, MSL_down_CBD, orig_img, seg_img_clean)
        #
        # MSL_up_CBD = new_MSL_up
        # MSL_down_CBD = new_MSL_down

         # CBD

        CBD_min_th = (seg_img.shape[0] /4)

        CBD, CBD_left, CBD_right = CBD_points((seg_img > 0).astype(int), MSL_up_CBD,  MSL_down_CBD, resX, resY, CBD_min_th)
        if CBD_left[0] == 0 and CBD_left[1] == 0:
            raise Exception('Can not solve')
        CBD_model.append(CBD)

        plt.imshow(orig_img, cmap='gray')
        plt.plot([CBD_left[1], CBD_right[1]], [CBD_left[0], CBD_right[0]], linewidth=0.5, color='blue')
        plt.plot([MSL_up_CBD[1], MSL_down_CBD[1]], [MSL_up_CBD[0], MSL_down_CBD[0]], linewidth=0.25, color='yellow')
        plt.title(Data[file_name][index])
        plt.ion()
        plt.show()
        fig_name = Data[file_name][index].replace(".nii", ".png")
        plt.savefig(os.path.join(BASE_DIR,'CBD', fig_name), dpi=400)
        plt.close()

        print(str(str('index num. = ') + str(index)))

        print (str(str('CBD = ') + str(CBD)))

        # BBD

        BBD, BBD_left, BBD_right, BBD_val = BBD_points(orig_img, CBD_left, CBD_right, MSL_up_CBD, MSL_down_CBD,  resX, resY)
        print (str(str('BBD = ') + str(BBD)))
        BBD_model.append(BBD)
        BBD_valid.append(BBD_val)

        plt.imshow(orig_img, cmap='gray')
        plt.plot([BBD_left[1], BBD_right[1]], [BBD_left[0], BBD_right[0]], linewidth=0.25, color='blue')
        plt.plot([MSL_up_CBD[1], MSL_down_CBD[1]], [MSL_up_CBD[0], MSL_down_CBD[0]], linewidth=0.25, color='yellow')
        plt.title(Data[file_name][index])
        plt.ion()
        plt.show()
        fig_name = Data[file_name][index].replace(".nii", ".png")
        plt.savefig(os.path.join(BASE_DIR, 'BBD', fig_name), dpi=400)
        plt.close()


    Data['CBD_new'] = np.array(CBD_model)
    Data['BBD_new'] = np.array(BBD_model)
    Data['bbd_valid'] = np.array(BBD_valid)

    for i in range(Data.shape[0]):
        dif_CBD.append(abs(Data['LiatCBD'][i]-Data['CBD_new'][i]))
        dif_BBD.append(abs(Data['LiatBBD'][i]-Data['BBD_new'][i]))

    Data['CBDDiff'] = np.array(dif_CBD)
    Data['BBDDiff'] = np.array(dif_BBD)

    # for measurtype in ["cbd", "bbd"]:
    #     for title in ["_measure_with_opt_with_SS_subseg", "_measure_no_opt_with_SS_subseg"]:
    #         plt_title = measurtype + title
    #         blandaltman.mean_diff_plot(Data[plt_title], Data["Liat" + measurtype.upper()])
    #         plt.title(plt_title)

    Data.to_excel(os.path.join(BASE_DIR, "new_analysis_stopPoint.xlsx"))








