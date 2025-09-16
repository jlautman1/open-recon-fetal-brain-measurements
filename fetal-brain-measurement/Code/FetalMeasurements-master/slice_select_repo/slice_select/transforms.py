from scipy import ndimage
import copy
import torch
import random

from torch.utils.data import dataloader
default_collate_func = dataloader.default_collate

from typing import Tuple
import numpy as np
import itertools


def all_except(tag):
    if isinstance(tag, str):
        tag_l = [tag, ]
    else:
        tag_l = tag
    def all_except__(func):
        def func_wrapper(self,sample, *args, **kwargs):

            out = {}
            ret_val = func(self, sample, *args, **kwargs)
            for k,v in sample.items():
                if k in tag_l:
                    if len(tag_l) == 1:
                        out[k] = ret_val
                    else:
                        out[k] = ret_val[k]
                else:
                    out[k] = v
                
            return out
        return func_wrapper
    return all_except__

def bbox_ND(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.append(np.where(nonzero)[0][[0, -1]])

    return tuple(out)
class Normalize(object):
    def __init__(self, mean, std):
        #self.norm_tf = pytorch_transforms.Normalize(mean=mean, std=std)
        self.mean = mean
        self.std = std
    
    @all_except(tag="image")
    def __call__(self, sample):
        sample_min = torch.min(sample["image"])
        sample_max = torch.max(sample["image"])
        sample_norm = sample["image"] / (sample_max - sample_min)
        sample_norm = (sample_norm - self.mean)/ self.std
        return sample_norm
class ToTensor(object):
    @all_except(tag="image")
    def __call__(self, sample):
        return torch.from_numpy(sample["image"]).float()

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    @all_except(tag="image")
    def __call__(self, sample:np.ndarray)-> np.ndarray:
        image = sample["image"]
        h, w = image.shape[1:3]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = ndimage.zoom(image, (1, new_h/h, new_w/w)) #TODO:support ND
        return img

class PadZ(object):
    def __init__(self, pad):
        self.pad = int(pad)

    @all_except(tag="image")
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        img_out = np.pad(sample["image"], ((self.pad, self.pad),(0,0),(0,0)), mode='edge')
        #print('shape',img_out.shape)
        return img_out

class SampleFrom3D(object):
    def __init__(self, negative_slides, sample_idx:str, context:int):
        assert (isinstance(negative_slides, (int,)) or (negative_slides is None))
        assert (context >= 0)

        self.negative_slides = negative_slides
        self.sample_idx = sample_idx
        self.context = context

    
    def __call__(self, sample):
        @all_except(["image", self.sample_idx])
        def helper(self, sample):
            negative_slides = self.negative_slides
            num_slides = sample['image'].shape[0]
            if (negative_slides is None):
                selection = torch.LongTensor(num_slides-2*self.context)
                selection.zero_()
                if self.sample_idx in sample.keys() and (not np.isnan(sample[self.sample_idx])):
                    selection[int(sample[self.sample_idx]) ] = 1
                #img = sample['image']
                contextrange = np.arange(-self.context, self.context+1)
                if self.context == 0:
                    sampling3 = [(a,a,a) for a in
                                 range(self.context, sample['image'].shape[0] - self.context)]
                else:
                    sampling3 = [tuple(contextrange + a) for a in range(self.context, sample['image'].shape[0] - self.context)]
                all_images = [sample['image'][elem,:,:] for elem in sampling3]
                img = torch.stack(all_images)
            else:
                sampling_population = set(range(self.context, sample['image'].shape[0] - self.context))
                sampling_population.remove(sample[self.sample_idx])
                sampling = random.sample(sampling_population, negative_slides)
                sampling.append(sample[self.sample_idx])

                #Sample + context
                contextrange = np.arange(-self.context, self.context+1)
                sampling3 = []
                for a in sampling:
                    if self.context == 0:
                        sampling3.append((a,a,a))
                    else:
                        if random.random() > .5:
                            sampling3.append(tuple(contextrange + a))
                        else:
                            sampling3.append(tuple(reversed(contextrange + a)))
                all_images = [sample['image'][elem,:,:] for elem in sampling3]
                img = torch.stack(all_images)
                #img = sample['image'][sampling,:,:]
                selection = torch.LongTensor([0,]*negative_slides + [1,])

            return {'image': img, self.sample_idx: selection}
        return helper(self, sample)


class RandomFlip(object):
    @all_except(tag="image")
    def __call__(self, sample):
        img = sample["image"]
        slices = img.size()[0]
        
        subs_flip = random.sample(range(slices),k=int(slices/2))
        subs_rotate = random.sample(range(slices),k=int(slices/2))
        img_out = img.clone()
        img_out[subs_flip,:,:] = torch.flip(img[subs_flip,:,:],[1,2])
        img_out[subs_rotate,:,:] = torch.rot90(img_out[subs_rotate,:,:],1,[1,2])

        return img_out


class RandomRotate(object):
    @all_except(tag=["image", "seg_image"])
    def __call__(self, sample):
        rot_angle = np.random.randint(360)
        # image_g = cp.asarray(sample["image"].astype(np.float32))
        # elem_nii_rot = cupyx.scipy.ndimage.rotate(image_g, angle=rot_angle,axes=(1,2))
        # image = cp.asnumpy(elem_nii_rot)

        # seg_image_g = cp.asarray(sample["seg_image"].astype(np.float32))
        # elem_nii_rot_seg = cupyx.scipy.ndimage.rotate(seg_image_g, angle=rot_angle,axes=(1,2))
        # seg_image = cp.asnumpy(elem_nii_rot_seg)
        image = ndimage.rotate(sample["image"], angle= rot_angle,axes=(1,2))
        seg_image = ndimage.rotate(sample["seg_image"], angle= rot_angle,axes=(1,2))
      
        return {'image':image, 'seg_image':seg_image}


class toXY(object):
    def __init__(self, x_type:str, y_type:str):
        self.x_type = x_type
        self.y_type = y_type

    def __call__(self, sample)-> Tuple[np.ndarray,np.ndarray]:
        return sample[self.x_type], sample[self.y_type]


class cropByBBox(object):

    def __init__(self, min_upcrop:float, max_upcrop:float):
        self.min_upcrop = min_upcrop
        self.max_upcrop = max_upcrop

    @all_except(tag="image")
    def __call__(self, sample):
        img = sample["image"]
        seg_image = sample["seg_image"]
        bbox_idxs = bbox_ND(seg_image)

        bbox_idxs = (np.array([0,seg_image.shape[0]]), *bbox_idxs[1:])

        # Random resize to bbox
        bbox_z, bbox_x, bbox_y = bbox_idxs
        if self.max_upcrop:
            bbox_factor = np.random.uniform(low=self.min_upcrop, high=self.max_upcrop)
        else:
            bbox_factor = 1.1

        bbox_w = bbox_x[1] - bbox_x[0]
        bbox_h = bbox_y[1] - bbox_y[0]
        bbox_cx = bbox_x[1] - bbox_w/2
        bbox_cy = bbox_y[1] - bbox_h/2

        bbox_dim = max(bbox_w, bbox_h)

        bbox_x = ( int(bbox_cx - bbox_dim*bbox_factor/2), int(bbox_cx + bbox_dim*bbox_factor/2))
        bbox_y = ( int(bbox_cy - bbox_dim*bbox_factor/2), int(bbox_cy + bbox_dim*bbox_factor/2))

        #Slice
        bbox_slice = ( slice(*bbox_z), slice(*bbox_x), slice(*bbox_y))
        img_out = img[bbox_slice]

        return img_out

def custom_collate_fn(batch):
    x , y = default_collate_func(batch)
    #print ('x', x.shape, 'y', y.shape)
    x = x.reshape((x.shape[0]*x.shape[1], *(x.shape[2:])))
    #x = torch.cat([x,x,x], dim=1)
    y = y.reshape((y.shape[0]*y.shape[1], *(y.shape[2:])))
    #print ('after x', x.shape, 'y', y.shape)
    return (x, y)


