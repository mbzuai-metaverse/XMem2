import json
import os
from os import path
from pathlib import Path
import shutil
import collections

import cv2
from PIL import Image
import torch
from torchvision.transforms import Resize, InterpolationMode

from util.image_loader import PaletteConverter

if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
import numpy as np

from util.palette import davis_palette
import progressbar
 

# https://bugs.python.org/issue28178
# ah python ah why
class LRU:
    def __init__(self, func, maxsize=128):
        self.cache = collections.OrderedDict()
        self.func = func
        self.maxsize = maxsize
 
    def __call__(self, *args):
        cache = self.cache
        if args in cache:
            cache.move_to_end(args)
            return cache[args]
        result = self.func(*args)
        cache[args] = result
        if len(cache) > self.maxsize:
            cache.popitem(last=False)
        return result

    def invalidate(self, key):
        self.cache.pop(key, None)


class ResourceManager:
    def __init__(self, config):
        # determine inputs
        images = config['images']
        video = config['video']
        self.workspace = config['workspace']
        self.size = config['size']
        self.palette = davis_palette
        self.palette_converter = PaletteConverter(self.palette)

        # create temporary workspace if not specified
        if self.workspace is None:
            if images is not None:
                p_images = Path(images)
                if p_images.name == 'JPEGImages' or (Path.cwd() / 'workspace') in p_images.parents:
                    # take the name instead of actual images dir (second case checks for videos already in ./workspace )
                    basename = p_images.parent.name
                else:
                    basename = p_images.name
            elif video is not None:
                basename = path.basename(video)[:-4]
            else:
                raise NotImplementedError(
                    'Either images, video, or workspace has to be specified')

            self.workspace = path.join('./workspace', basename)

        print(f'Workspace is in: {self.workspace}')
        self.workspace_info_file = path.join(self.workspace, 'info.json')
        self.references = set()
        self._num_objects = None
        self._try_load_info()

        if config['num_objects'] is not None:  # forced overwrite from user
            self._num_objects = config['num_objects']
        elif self._num_objects is None:  # both are None, single object first run use case
            self._num_objects = config['num_objects_default_value']
        self._save_info()

        # determine the location of input images
        need_decoding = False
        need_resizing = False
        if path.exists(path.join(self.workspace, 'images')):
            pass
        elif images is not None:
            need_resizing = True
        elif video is not None:
            # will decode video into frames later
            need_decoding = True

        # create workspace subdirectories
        self.image_dir = path.join(self.workspace, 'images')
        self.mask_dir = path.join(self.workspace, 'masks')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)

        # convert read functions to be buffered
        self.get_image = LRU(self._get_image_unbuffered, maxsize=config['buffer_size'])
        self.get_mask = LRU(self._get_mask_unbuffered, maxsize=config['buffer_size'])

        # extract frames from video
        if need_decoding:
            self._extract_frames(video)

        # copy/resize existing images to the workspace
        if need_resizing:
            self._copy_resize_frames(images)

        # read all frame names
        self.names = sorted(os.listdir(self.image_dir))
        self.names = [f[:-4] for f in self.names] # remove extensions
        self.length = len(self.names)

        assert self.length > 0, f'No images found! Check {self.workspace}/images. Remove folder if necessary.'

        print(f'{self.length} images found.')

        self.height, self.width = self.get_image(0).shape[:2]
        self.visualization_init = False

        self._resize = None
        self._masks = None
        self._keys = None
        self._keys_processed = np.zeros(self.length, dtype=bool)
        self.key_h = None
        self.key_w = None

    def _extract_frames(self, video):
        cap = cv2.VideoCapture(video)
        frame_index = 0
        print(f'Extracting frames from {video} into {self.image_dir}...')
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
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

    def _copy_resize_frames(self, images):
        image_list = os.listdir(images)
        print(f'Copying/resizing frames into {self.image_dir}...')
        for image_name in progressbar.progressbar(image_list):
            if self.size < 0:
                # just copy
                shutil.copy2(path.join(images, image_name), self.image_dir)
            else:
                frame = cv2.imread(path.join(images, image_name))
                h, w = frame.shape[:2]
                new_w = (w*self.size//min(w, h))
                new_h = (h*self.size//min(w, h))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, image_name), frame)
        print('Done!')

    def add_key_and_stuff_with_mask(self, ti, key, shrinkage, selection, mask):
        if self._keys is None:
            c, h, w = key.squeeze().shape
            if self.key_h is None:
                self.key_h = h
            if self.key_w is None:
                self.key_w = w
            c_mask, h_mask, w_mask = mask.shape
            self._keys = torch.empty((self.length, c, h, w), dtype=key.dtype, device=key.device)
            self._shrinkages = torch.empty((self.length, 1, h, w), dtype=key.dtype, device=key.device)
            self._selections = torch.empty((self.length, c, h, w), dtype=key.dtype, device=key.device)
            self._masks = torch.empty((self.length, c_mask, h_mask, w_mask), dtype=mask.dtype, device=key.device)
            # self._resize = Resize((h, w), interpolation=InterpolationMode.NEAREST)
        
        if not self._keys_processed[ti]:
            # keys don't change for the video, so we only save them once
            self._keys[ti] = key
            self._shrinkages[ti] = shrinkage
            self._selections[ti] = selection
            self._keys_processed[ti] = True
                
        self._masks[ti] = mask# self._resize(mask)

    def all_masks_present(self):
        return self._keys_processed.sum() == self.length
    
    def add_reference(self, frame_id: int):
        self.references.add(frame_id)
        self._save_info()

    def remove_reference(self, frame_id: int):
        print(self.references)
        self.references.remove(frame_id)
        self._save_info()

    def _save_info(self):
        p_workspace_subdir = Path(self.workspace_info_file).parent
        p_workspace_subdir.mkdir(parents=True, exist_ok=True)
        with open(self.workspace_info_file, 'wt') as f:
            data = {'references': sorted(self.references), 'num_objects': self._num_objects}

            json.dump(data, f, indent=4)

    def _try_load_info(self):
        try:
            with open(self.workspace_info_file) as f:
                data = json.load(f)
                self._num_objects = data['num_objects']

                # We might have num_objects, but not references if imported the project
                self.references = set(data['references'])
        except Exception:
            pass


    def save_mask(self, ti, mask):
        # mask should be uint8 H*W without channels
        assert 0 <= ti < self.length
        assert isinstance(mask, np.ndarray)

        mask = Image.fromarray(mask)
        mask.putpalette(self.palette)
        mask.save(path.join(self.mask_dir, self.names[ti]+'.png'))
        self.invalidate(ti)

    def save_visualization(self, ti, image):
        # image should be uint8 3*H*W
        assert 0 <= ti < self.length
        assert isinstance(image, np.ndarray)
        if not self.visualization_init:
            self.visualization_dir = path.join(self.workspace, 'visualization')
            os.makedirs(self.visualization_dir, exist_ok=True)
            self.visualization_init = True

        image = Image.fromarray(image)
        image.save(path.join(self.visualization_dir, self.names[ti]+'.jpg'))

    def _get_image_unbuffered(self, ti):
        # returns H*W*3 uint8 array
        assert 0 <= ti < self.length

        image = Image.open(path.join(self.image_dir, self.names[ti]+'.jpg'))
        image = np.array(image)
        return image

    def _get_mask_unbuffered(self, ti):
        # returns H*W uint8 array
        assert 0 <= ti < self.length

        mask_path = path.join(self.mask_dir, self.names[ti]+'.png')
        if path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = np.array(mask)
            return mask
        else:
            return None

    def read_external_image(self, file_name, size=None, force_mask=False):
        image = Image.open(file_name)
        is_mask = image.mode in ['L', 'P']

        if size is not None:
            # PIL uses (width, height)
            image = image.resize((size[1], size[0]), 
                    resample=Image.Resampling.NEAREST if is_mask or force_mask else Image.Resampling.BICUBIC)
        
        if force_mask and image.mode != 'P':
            image = self.palette_converter.image_to_index_mask(image)
        #     if image.mode in ['RGB', 'L'] and len(image.getcolors()) <= 2:
        #         image = np.array(image.convert('L'))
        #         # hardcoded for b&w images
        #         image = np.where(image, 1, 0)  # 255 (or whatever) -> binarize

        #         return image.astype('uint8')
        #     elif image.mode == 'RGB':
        #         image = image.convert('P', palette=self.palette)
        #         tmp_image = np.array(image)
        #         out_image = np.zeros_like(tmp_image)
        #         for i, c in enumerate(np.unique(tmp_image)):
        #             if i == 0:
        #                 continue
        #             out_image[tmp_image == c] = i  # palette indices into 0, 1, 2, ...
        #         self.palette = image.getpalette()
        #         return out_image
                
        #     image = image.convert('P', palette=self.palette)  # saved without DAVIS palette, just number objects 0, 1, ...
            
        image = np.array(image)
        return image

    def invalidate(self, ti):
        # the image buffer is never invalidated
        self.get_mask.invalidate((ti,))

    def __len__(self):
        return self.length

    @property
    def h(self):
        return self.height

    @property
    def w(self):
        return self.width
    
    @property
    def small_masks(self):
        return self._masks

    @property
    def keys(self):
        return self._keys
        

    @property
    def shrinkages(self):
        return self._shrinkages
    
    @property
    def selections(self):
        return self._selections
    
    @property
    def num_objects(self):
        return self._num_objects
