import numpy as np
import matplotlib.pyplot as plt
from data_generation.preprocess import window_1_99


def prepare_for_plotting(volume, truth, pred):
    #swap x and y axes
    volume = np.swapaxes(volume,0,1)
    truth = np.swapaxes(truth,0,1)
    pred = np.swapaxes(pred,0,1)

    #transpose x axes
    volume = np.flip(volume, 1)
    truth = np.flip(truth, 1)
    pred = np.flip(pred, 1)

    #transpose y axes
    volume = np.flip(volume, 0)
    truth = np.flip(truth, 0)
    pred = np.flip(pred, 0)

    volume = window_1_99(volume, 0, 99)

    return volume, truth, pred


def plot_vol_gt(vol, labels, thresh = 0):
    num_slices = vol.shape[2]
    for i in range(0, num_slices):
        slice_data = vol[:,:,i]
        slice_labels = labels[:,:,i]
        indices = np.nonzero(slice_labels > thresh)
        if (len(indices[0]) == 0):
            continue
        plot_data(slice_data, slice_labels, "slice: " + str(i+1))


def compare_data(image1, image2, title = "compare"):
    fig, ax = plt.subplots(2, 1, figsize=[8, 8])
    plt.suptitle(title)
    plt.subplot(2, 1, 1)
    plt.imshow(image1, cmap='gray', origin='lower')
    plt.subplot(2, 1, 2)
    plt.imshow(image2, cmap='gray', origin='lower')


def plot_data(imageData,labelData, plotTitle):

    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    plt.suptitle(plotTitle)
    plt.subplot(2, 2, 1)
    plt.imshow(imageData, cmap='gray', origin='lower')
    plt.subplot(2, 2, 2)
    plt.imshow(labelData, cmap='gray', origin='lower')
    plt.subplot(2, 2, 3)
    plt.imshow(imageData * labelData, cmap='gray', origin='lower')
    plt.subplot(2, 2, 4)
    plt.imshow(imageData, cmap='gray')
    plt.imshow(labelData, cmap='jet', alpha=0.5, origin='lower')

    plt.show()


def plot_data(imageData,labelData, plotTitle):

    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    plt.suptitle(plotTitle)
    plt.subplot(2, 2, 1)
    plt.imshow(imageData, cmap='gray', origin='lower')
    plt.subplot(2, 2, 2)
    plt.imshow(labelData, cmap='gray', origin='lower')
    plt.subplot(2, 2, 3)
    plt.imshow(imageData * labelData, cmap='gray', origin='lower')

    plt.subplot(2, 2, 4)
    plt.imshow(imageData, cmap='gray')
    plt.imshow(labelData, cmap='jet', alpha=0.5, origin='lower')

    plt.show()


def plot_sample(image, gt, result, title_plot='comparison', title_image='image', title_overlay1='overlay1', title_overlay2 = 'overlay2'):
    plt = get_plot_sample(image, gt, result, title_plot, title_image, title_overlay1, title_overlay2)
    plt.show()


def visualize_res_gt(vol, labels, res):
    for i in range(0, len(vol)):
        indices_labels = np.nonzero(labels[:,:,i] > 0)
        indices_res = np.nonzero(res[:,:,i] > 0)
        if ((len(indices_labels[0]) == 0) & (len(indices_res[0]) == 0)):
            continue
        plot_sample(vol[:,:,i], labels[:,:,i],res[:,:,i], "gt visualization, slice: " + str(i+1))


def get_plot_sample(image, gt, result, title_plot='comparison', title_image='image', title_overlay1='overlay1', title_overlay2 = 'overlay2'):
    fig, ax = plt.subplots(1,3, figsize=[15, 15])
    plt.suptitle(title_plot)
    plt.subplot(1, 3, 1, title = title_image)
    plt.imshow(image, cmap='gray', origin='lower')
    plt.subplot(1, 3, 2, title = title_overlay1)
    plt.imshow(image, cmap='gray')
    plt.imshow(gt, cmap='jet', alpha=0.5, origin='lower')
    plt.subplot(1, 3, 3, title = title_overlay2)
    plt.imshow(image, cmap='gray')
    plt.imshow(result, cmap='jet', alpha=0.5, origin='lower')

    return plt

def get_plot_gt_res_overlays(image, gt, result, title_plot='comparison', title_overlay1='truth', title_overlay2 ='prediction'):
    fig, ax = plt.subplots(1,2, figsize=[15, 15])
    plt.suptitle(title_plot)
    plt.subplot(1, 2, 1, title = title_overlay1)
    min = np.amin(image)
    max = np.amax(image)
    plt.imshow(image, cmap='gist_gray', interpolation='none', vmin=np.amin(image), vmax=np.amax(image))
    plt.imshow(gt, cmap='Reds', alpha=0.4, interpolation='none', origin='upper')
    plt.axis('off')
    plt.subplot(1, 2, 2, title = title_overlay2)
    plt.imshow(image, cmap='gist_gray', interpolation='none', vmin=np.amin(image), vmax=np.amax(image))
    plt.imshow(result, cmap='Reds', alpha=0.4,interpolation='none', origin='upper')
    fig.tight_layout()
    fig.subplots_adjust(top=0.98)
    plt.axis('off')
    figure = plt.gcf() # get current figure
    figure.set_size_inches(8, 5)
    return plt