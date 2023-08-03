from dataclasses import dataclass, replace
import os
from os import path
from tempfile import TemporaryDirectory
from typing import Optional
import cv2
import progressbar

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization


@dataclass
class Sample:
    rgb: torch.Tensor
    raw_image_pil: Image.Image
    frame: str
    save: bool
    shape: tuple
    need_resize: bool
    mask: Optional[torch.Tensor] = None


class VideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self, vid_name, video_path, mask_dir, size=-1, to_save=None, use_all_masks=False, size_dir=None):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        """
        self.vid_name = vid_name
        self.video_path = video_path
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_masks = use_all_masks

        self.reference_mask = Image.open(path.join(mask_dir, sorted(os.listdir(mask_dir))[0])).convert('P')
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

        if os.path.isfile(self.video_path):
            self.tmp_dir = TemporaryDirectory()
            self.image_dir = self.tmp_dir.name
            self._extract_frames()
        else:
            self.image_dir = video_path

        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir
        
        self.frames = sorted(os.listdir(self.image_dir))

    def __getitem__(self, idx) -> Sample:
        data = {}
        frame_name = self.frames[idx]
        im_path = path.join(self.image_dir, frame_name)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame_name)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        gt_path = path.join(self.mask_dir, frame_name[:-4]+'.png')
        if not os.path.exists(gt_path):
            gt_path = path.join(self.mask_dir, frame_name[:-4]+'.PNG')
        
        data['raw_image_pil'] = img
        img = self.im_transform(img)

        load_mask = self.use_all_masks or (gt_path == self.first_gt_path)
        if load_mask and path.exists(gt_path):
            mask = Image.open(gt_path).convert('P')
            mask = np.array(mask, dtype=np.uint8)
            data['mask'] = mask

        info = {}
        info['save'] = (self.to_save is None) or (frame_name[:-4] in self.to_save)
        info['frame'] = frame_name
        info['shape'] = shape
        info['need_resize'] = not (self.size < 0)

        data['rgb'] = img

        data = Sample(**data, **info)

        return data
    
    def __len__(self):
        return len(self.frames)
    
    def __del__(self):
        if hasattr(self, 'tmp_dir'):
            self.tmp_dir.cleanup()

    def _extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_index = 0
        print(f'Extracting frames from {self.video_path} into a temporary dir...')
        bar = progressbar.ProgressBar(max_value=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        while(cap.isOpened()):
            _, frame = cap.read()
            if frame is None:
                break
            if self.size > 0:
                h, w = frame.shape[:2]
                new_w = (w*self.size//min(w, h))
                new_h = (h*self.size//min(w, h))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
            cv2.imwrite(path.join(self.image_dir, f'frame_{frame_index:06d}.jpg'), frame)
            frame_index += 1
            bar.update(frame_index)
        bar.finish()
        print('Done!')
    

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    def map_the_colors_back(self, pred_mask: Image.Image):
        # https://stackoverflow.com/questions/29433243/convert-image-to-specific-palette-using-pil-without-dithering
        # dither=Dither.NONE just in case
        return pred_mask.quantize(palette=self.reference_mask, dither=Image.Dither.NONE).convert('RGB')

    @staticmethod
    def collate_fn_identity(x):
        if x.mask is not None:
            return replace(x, mask=torch.tensor(x.mask))
        else:
            return x