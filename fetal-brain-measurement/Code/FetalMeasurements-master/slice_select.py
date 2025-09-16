import sys

sys.path.append('slice_select_repo/slice_select/')

import torch
from slice_select_repo.slice_select.loader import NiftiElement
import slice_select_repo.slice_select.transforms as tfs
from torchvision import transforms as pytorch_tfs
from torchvision import models
import math
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
import numpy as np

from matplotlib import pyplot as plt


class SliceSelect(object):
    def __init__(self, model_file, cuda_id=0, basemodel='ResNet34'):
        print("model file: ", model_file)
        model = torch.load(model_file)
        if basemodel == 'ResNet34':
            model_ft = models.resnet34(pretrained=True)
        elif basemodel == 'ResNet50':
            model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, 2)
        model_ft.load_state_dict(model)
        torch.cuda.set_device(cuda_id)
        device = torch.device("cuda:" + str(cuda_id))
        self.model = model_ft.to(device)

        self.transform = pytorch_tfs.Compose([
            tfs.PadZ(0),
            # tfs.RandomRotate(),
            tfs.cropByBBox(min_upcrop=1.1, max_upcrop=1.1),
            tfs.Rescale((224, 224)),
            tfs.ToTensor(),
            tfs.Normalize(mean=0.456,
                          std=0.224),
            tfs.SampleFrom3D(None, sample_idx="Selection", context=0),

            # tfs.RandomRotate(),
            # tfs.toXY("image", "TCD_Selection"),
        ])
        self.device = device

    def _visualize_model(self, data_elem, device, visualize=False, out_dir='./Pdfs/'):
        model = self.model
        print("device: ", device)
        class_names = ['no', 'yes']
        was_training = model.training
        model.eval()
        images_so_far = 0

        df = pd.DataFrame(columns=['filename', 'Selection', 'prediction', 'Z', 'val_choose_acc', 'prob_vec', 'isValid'])
        i = 0
        os.makedirs(out_dir, exist_ok=True)
        with torch.no_grad():

            x = data_elem["image"]
            y = data_elem["Selection"]

            x = x.reshape((x.shape[0], *(x.shape[1:])))

            inputs = x.to(device)
            #print("inputs inside visulaize: ", inputs)
            labels = y.to(device)
            orig_fname = os.path.basename(data_elem['filename'])
            # ✅ Save preprocessed slices being fed into the model
            slice_dir = os.path.join(out_dir, "slice_debug", orig_fname)
            os.makedirs(slice_dir, exist_ok=True)
            for j in range(inputs.size(0)):
                img = inputs[j].cpu().numpy().transpose((1, 2, 0))[:, :, 1]  # Shape: H x W
                img = img - np.min(img)
                img = img / (np.max(img) - np.min(img) + 1e-5)  # Normalize for visibility
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.title(f"Slice {j}")
                plt.savefig(os.path.join(slice_dir, f"slice_{j:02d}.png"))
                plt.close()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            outputs_sfmax = torch.nn.functional.softmax(outputs, dim=1)

            values, indices = outputs_sfmax[:, 1].max(0)
            if visualize:

                dest_pdf = os.path.join(out_dir, orig_fname + '.pdf')
                pp = PdfPages(dest_pdf)

                fig = plt.figure()
                fig.set_figheight(15)
                fig.set_figwidth(15)
                for j in range(inputs.size()[0]):

                    ax = plt.subplot(math.ceil(inputs.size()[0] / 3.), 3, j + 1)
                    ax.axis('off')
                    probs_sfmax = outputs_sfmax[j].cpu().numpy().tolist()
                    probs = outputs[j].cpu().numpy().tolist()
                    # ax.set_title('predicted: {} probs : {:.2f},{:.2f} [{:.2f},{:.2f}]'.format(class_names[preds[j]],probs[0],probs[1],probs_sfmax[0],probs_sfmax[1]))
                    # if (j == indices):
                    #    ax.set_title('**** {}, {:.2f},{:.2f} [{:.2f},{:.2f}]'.format(class_names[preds[j]],probs[0],probs[1],probs_sfmax[0],probs_sfmax[1]))
                    ax.set_title(' {} '.format(j, ))
                    img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))[:, :, 1]
                    img = img + np.min(img)
                    img = img / (np.max(img) - np.min(img))
                    if (j == indices):
                        ax.imshow(img)
                    else:
                        ax.imshow(img, cmap='Greys_r')

                pp.savefig(fig)

                fig2 = plt.figure()
                ax = plt.subplot(1, 1, 1)

            max_val, max_idx = torch.max(outputs_sfmax[:, 1], dim=0)
            max_val_lbl, max_val_idx = torch.max(labels.data, dim=0)
            val_choose_acc = (1. - (abs(float(max_idx) - float(max_val_idx)) / float(inputs.size(0))))

            labels = labels.cpu().numpy()
            labels = labels * .5 + .25
            outputs_ok = ((outputs_sfmax.cpu().numpy() > .5) * .5).T[1, :] + .25

            if visualize:
                ax.plot(np.arange(len(labels)) + .5, outputs_sfmax.cpu().numpy()[:, 1], 'seagreen', linewidth=3,
                        label='probability', marker='o', markerfacecolor='black', markersize=3.)

                ax.barh(outputs_ok, left=range(len(inputs)), width=1, color='g', alpha=.4, align='center', height=.2,
                        label='alg: candidate')
                chosen_id = np.argmax(outputs_sfmax.cpu().numpy().T[1, :])
                chosen = [.25, ] * len(outputs_ok)
                chosen[chosen_id] = .75
                ax.barh(chosen, left=range(len(inputs)), width=1, color='None', alpha=.4, align='center', height=.2,
                        edgecolor="darkgreen", linewidth=4, label='alg: chosen')

                if np.sum(outputs_sfmax.cpu().numpy().T[1, :]) > 0:
                    ax.barh(labels, left=range(len(inputs)), width=1, height=.5, color='None', alpha=.3, linewidth=4,
                            align='center', edgecolor="black", label='GT')

                ax.legend()
                plt.xlabel('Slice #')
                plt.ylabel('Probablity of slice to be chosen for TCD')
                plt.title('Choose ACC= %f' % (val_choose_acc,))
                pp.savefig(fig2)
                pp.close()
            df.loc[i] = [orig_fname, y.cpu().numpy(), indices.cpu().numpy(), inputs.size(0), val_choose_acc,
                         outputs.cpu().numpy(), bool(max_val > .5)]

            model.train(mode=was_training)
        return df

    def execute(self, img_file, seg_file, visualize=False):
        elem = NiftiElement(nii_elem=img_file,
                            seg_elem=seg_file,
                            selection=-1,
                            transform=self.transform)
        return self._visualize_model(elem(), device=self.device, visualize=visualize)

    def get_cropped_elem(self, img_file, seg_file):
        elem = NiftiElement(nii_elem=img_file,
                            seg_elem=seg_file,
                            selection=-1,
                            transform=tfs.cropByBBox(min_upcrop=1.5, max_upcrop=1.5))
        orig_seg = elem()["seg_image"]
        cropper = tfs.cropByBBox(min_upcrop=1.5, max_upcrop=1.5)
        cropped_seg = cropper({"image":orig_seg.copy(), "seg_image":orig_seg})
        return elem()["image"], cropped_seg["image"]