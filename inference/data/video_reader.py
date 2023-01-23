import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch 

from dataset.range_transform import im_normalization

import sys
sys.path.append('/l/users/ariana.venegas/Documents/Documents/RESEARCH/FlowFormer-Official')
print (sys.path)
from visualize_flow_xmem import compute_flow_dir, build_model, generate_pairs

class VideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self, vid_name, image_dir, mask_dir, size=-1, to_save=None, use_all_mask=False, size_dir=None, of_dir="optical_flow_results"):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        of_dir - the folder name of the generated optical flow results (Flowformer)
        """
        
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        self.frames = sorted([f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))])  
        self.palette = Image.open(path.join(mask_dir, sorted(os.listdir(mask_dir))[0])).getpalette()
        self.first_gt_path = path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[0])

        if size < 0:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
            ])
        self.size = size

        # ----- test optical flow adaption decoder (erase) -----
        im_path = path.join(self.image_dir, self.frames[0]) #erase 
        img = Image.open(im_path).convert('RGB')

        
        self.shape = np.array(img).shape[:2]
        print("size is: ", self.size)
        # ----- end (erase) -----
        # ----- optical flow addition -----
        self.of_dir = of_dir
        model = build_model()
        self.img_pairs = generate_pairs(self.image_dir, self.frames, 0, len(self.frames)-1)
        path_results = compute_flow_dir(".", self.of_dir, model, self.img_pairs, keep_size=True)
        self.path_results = sorted(path_results)
        #  in none to liberate the memory 
        model = None
        # ----- optical flow end -----


    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        gt_path = path.join(self.mask_dir, frame[:-4]+'.png')
        img = self.im_transform(img)

        load_mask = self.use_all_mask or (gt_path == self.first_gt_path)
        if load_mask and path.exists(gt_path):
            mask = Image.open(gt_path).convert('P')
            mask = np.array(mask, dtype=np.uint8)
            data['mask'] = mask

         # ----- optical flow addition ----
        if (idx != 0 and idx<len(self.path_results)): 
            # load optical flow
            opt_flow = np.load(self.path_results[idx])
        else: 
            opt_flow = torch.zeros(np.load(self.path_results[1]).shape)

        data['optical_flow'] = opt_flow
        # ----- optical flow end -----
        info['shape'] = shape
        info['need_resize'] = not (self.size < 0)
        data['rgb'] = img
        data['info'] = info

        return data

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need tostep post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)
