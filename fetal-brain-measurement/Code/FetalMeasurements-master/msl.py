
import numpy as np
import time

from matplotlib import pyplot as plt

from sklearn import svm

RIGHT_HEMI = 1
LEFT_HEMI = 5

def find_planes(sub_img, sub_seg, visualize=False):
    #print("printing the subseg: ", sub_seg)
    #print("printing the sub image: ", sub_img)
    print("started MSL first")
    
    start_time = time.time()
    max_total_time = 240  # 4 minutes total timeout
    last_print_time = start_time
    print_interval = 5  # Print every 5 seconds
    
    OUTPUT_PLANES = {}
    for i in range(sub_seg.shape[2]):
        # Simple timeout check
        elapsed_total = time.time() - start_time
        if elapsed_total > max_total_time:
            print(f"MSL TIMEOUT! Stopping at slice {i} after {elapsed_total:.1f}s")
            break
            
        # Progress print every 5 seconds
        if elapsed_total - (last_print_time - start_time) >= print_interval:
            print(f"MSL Progress: {elapsed_total:.1f}s elapsed, processing slice {i}/{sub_seg.shape[2]-1}")
            last_print_time = time.time()
        
        # Take current slice
        cur_plane_seg = sub_seg[:, :, i]
        cur_plane_img = sub_img[:, :, i]
        #print("printing the cur_plane_seg: ", cur_plane_seg)
        #print("printing the cur_plane_img: ", cur_plane_img)
        # Seperate to two hemispheres
        # right_pts = np.array(np.where(cur_plane_seg == RIGHT_HEMI)).T
        # left_pts = np.array(np.where(cur_plane_seg == LEFT_HEMI)).T
        right_pts = np.array(np.where(cur_plane_seg == 1.)).T
        left_pts = np.array(np.where(cur_plane_seg == 3.)).T
        #print("printing the right_pts: ", right_pts)
        #print("printing the left_pts: ", left_pts)


        #print(f"[SLICE {i}] found R={len(right_pts)} L={len(left_pts)}") #DEBUG

        # If no points of hemisphere, skip
        if len(left_pts) <= 0 or len(right_pts) <= 0:
            #print("#####################one of the points are <0 - left points: ", len(left_pts),len(right_pts))
            continue

        # Create SVM between two hemispheres
        X = np.concatenate([right_pts, left_pts])
        Y = np.array(np.concatenate([[0, ] * len(right_pts), [1, ] * len(left_pts)]))
        clf = svm.SVC(kernel='linear', C=10, tol=1e-1, max_iter=int(1e8))
        clf.fit(X, Y)

        # Calculate line parameters
        a = -clf.coef_[0][0] / clf.coef_[0][1]
        b = -clf.intercept_ / clf.coef_[0][1]
        OUTPUT_PLANES[i] = (float(a), float(b), clf.coef_[0], clf.intercept_)
        #print("moving to visualize if needed: ", visualize)
        #print (np.abs(np.linalg.norm(clf.coef_[0])))
        # Visualize if needed
        if visualize:
            plt.figure()
            plt.imshow(cur_plane_img.T)
            plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired, alpha=.005)

            # plot the decision function
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # create grid to evaluate model
            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = clf.decision_function(xy).reshape(XX.shape)

            # plot decision boundary and margins
            ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                       linestyles=['--', '-', '--'])
            plt.show()
        print("finished for i: ", i)
    print("finished MSL first")
    return OUTPUT_PLANES


def determine_sides(subseg_plane, w, b_svm,  visualize=False):
    # All code assumes segmentation in centralized
    #print("w[1]: ", w[1])    
    if w[1]!=0:
        #print("w[1] is zero")
        a = float(-w[0] / w[1])
        b = float(-b_svm / w[1])
    else: 
        #print("w[1] is not zero")
        a = float(-w[0])
        b = float(-b_svm)

    # Split to cases
    W, H = subseg_plane.shape
    if np.isinf(a) or np.abs(a) > 100:  # Maybe wee should use some threshold a > 1000
        # The MSL is horizontal
        pt_0 = np.array((float(-b_svm / w[0]), 0))
        pt_1 = np.array((float(-b_svm / w[0]), H))

        pt_c = np.array((float(-b_svm / w[0]), H / 2))
        pt_q = np.array((0, H / 2))

    elif a == 0:  # Same, maybe we should use epsilon
        pt_0 = np.array((0, b))
        pt_1 = np.array((W, b))

        pt_c = np.array((W / 2, H / 2))
        pt_q = np.array((W / 2, 0))
    elif np.abs(a) <= 1:
        pt_0 = np.array((0, b))
        pt_1 = np.array((W, a * W + b))

        pt_c = np.array((W / 2, a * (W / 2) + b))
        pt_q = np.array((a * W / 2 * (a + 1 / a) - a * b, 0))
    else:  # np.abs(a) > 1
        pt_0 = np.array((-b / a, 0))
        pt_1 = np.array(((H - b) / a, H))

        pt_c = np.array((W / 2, a * (W / 2) + b))
        pt_q = np.array((0, W / 2 * (a + 1 / a) + b))

    # Sample points on cerebellum
    pt_on_cerebellum = np.array(np.where(subseg_plane == 2.)).T
    if pt_on_cerebellum.shape[0] == 0:
        return pt_0, pt_1, False
    pt_sample_on_cerebellum = pt_on_cerebellum[np.random.choice(pt_on_cerebellum.shape[0], 5)]

    # Determine sign and quality of segmentation
    pt_c_sign = np.sign(np.cross(pt_q - pt_c, pt_0 - pt_c)) > 0
    pt_sign_all = np.sign(np.cross(pt_c - pt_q, pt_sample_on_cerebellum - pt_q)) > 0
    if pt_c_sign:
        pt_sign_all = ~pt_sign_all


    if all(pt_sign_all):
        pt_0, pt_1 = pt_1, pt_0
    elif all(~pt_sign_all):
        pass
    else:
        print ("Problem")
        return pt_0, pt_1, False

    if visualize:
        # Visualize
        plt.figure()
        plt.imshow(subseg_plane)
        plt.scatter(pt_0[1], pt_0[0], c='r')
        plt.scatter(pt_1[1], pt_1[0], c='k')
        plt.scatter(pt_q[1], pt_q[0], c='c')
        print(pt_sign_all)

        # Second way to debug
        for pt_idx in range(pt_sample_on_cerebellum.shape[0]):
            pt = pt_sample_on_cerebellum[pt_idx, :]
            pt_sign = np.cross(pt_c - pt_q, pt - pt_q)
            if pt_sign > 0:
                plt.scatter(pt[1], pt[0], c='b')
            else:
                plt.scatter(pt[1], pt[0], c='g')
        all_pt = np.stack([pt_0, pt_1]).T
        print(all_pt)
        plt.plot(all_pt[1], all_pt[0], 'b--')

    return pt_0, pt_1, True


def findMSLforAllPlanes(img, subseg, planes_dict, visualize=False):
    print("starting findMSLforAllPlanes")
    PLANES_MSL = {}
    valid_up_pts = None
    valid_down_pts = None

    # Collect valid planes
    for i in planes_dict.keys():
        a, b, w, b_svm = planes_dict[i]
        PLANES_MSL[i] = determine_sides(subseg[:, :, i], w, b_svm, visualize)
        pt_0, pt_1, isValid = PLANES_MSL[i]
        if not isValid:
            continue
        if valid_up_pts is None:
            valid_up_pts = np.expand_dims(pt_0, axis=0)
            valid_down_pts = np.expand_dims(pt_1, axis=0)
        else:
            valid_up_pts = np.concatenate([valid_up_pts, np.expand_dims(pt_0, axis=0)], axis=0)
            valid_down_pts = np.concatenate([valid_down_pts, np.expand_dims(pt_1, axis=0)], axis=0)

    # Fix with Nearest neighbors
    for i in planes_dict.keys():
        pt_0, pt_1, isValid = PLANES_MSL[i]
        if isValid:
            continue
        dist_from_up = np.max(np.linalg.norm(valid_up_pts - pt_0, axis=1))
        dist_from_down = np.max(np.linalg.norm(valid_down_pts - pt_0, axis=1))
        if dist_from_down < dist_from_up:
            # Reverse
            PLANES_MSL[i] = pt_1, pt_0, True
        else:
            PLANES_MSL[i] = pt_0, pt_1, True
    print("finished findMSLforAllPlanes")
    return PLANES_MSL
