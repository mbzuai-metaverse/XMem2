from multiprocessing import Process, Queue, Value
import os
from pathlib import Path
import queue
from time import perf_counter
import time
import cv2
import numpy as np
from PIL import Image

import torch
from dataset.range_transform import inv_im_trans
from collections import defaultdict

from inference.interact.interactive_utils import overlay_davis

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def transpose_np(x):
    return np.transpose(x, [1,2,0])

def tensor_to_gray_im(x):
    x = detach_to_cpu(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def tensor_to_im(x):
    x = detach_to_cpu(x)
    x = inv_im_trans(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

# Predefined key <-> caption dict
key_captions = {
    'im': 'Image', 
    'gt': 'GT', 
}

"""
Return an image array with captions
keys in dictionary will be used as caption if not provided
values should contain lists of cv2 images
"""
def get_image_array(images, grid_shape, captions={}):
    h, w = grid_shape
    cate_counts = len(images)
    rows_counts = len(next(iter(images.values())))

    font = cv2.FONT_HERSHEY_SIMPLEX

    output_image = np.zeros([w*cate_counts, h*(rows_counts+1), 3], dtype=np.uint8)
    col_cnt = 0
    for k, v in images.items():

        # Default as key value itself
        caption = captions.get(k, k)

        # Handles new line character
        dy = 40
        for i, line in enumerate(caption.split('\n')):
            cv2.putText(output_image, line, (10, col_cnt*w+100+i*dy),
                     font, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Put images
        for row_cnt, img in enumerate(v):
            im_shape = img.shape
            if len(im_shape) == 2:
                img = img[..., np.newaxis]

            img = (img * 255).astype('uint8')

            output_image[(col_cnt+0)*w:(col_cnt+1)*w,
                         (row_cnt+1)*h:(row_cnt+2)*h, :] = img
            
        col_cnt += 1

    return output_image

def base_transform(im, size):
        im = tensor_to_np_float(im)
        if len(im.shape) == 3:
            im = im.transpose((1, 2, 0))
        else:
            im = im[:, :, None]

        # Resize
        if im.shape[1] != size:
            im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)

        return im.clip(0, 1)

def im_transform(im, size):
        return base_transform(inv_im_trans(detach_to_cpu(im)), size=size)

def mask_transform(mask, size):
    return base_transform(detach_to_cpu(mask), size=size)

def out_transform(mask, size):
    return base_transform(detach_to_cpu(torch.sigmoid(mask)), size=size)

def pool_pairs(images, size, num_objects):
    req_images = defaultdict(list)

    b, t = images['rgb'].shape[:2]

    # limit the number of images saved
    b = min(2, b)

    # find max num objects
    max_num_objects = max(num_objects[:b])

    GT_suffix = ''
    for bi in range(b):
        GT_suffix += ' \n%s' % images['info']['name'][bi][-25:-4]

    for bi in range(b):
        for ti in range(t):
            req_images['RGB'].append(im_transform(images['rgb'][bi,ti], size))
            for oi in range(max_num_objects):
                if ti == 0 or oi >= num_objects[bi]:
                    req_images['Mask_%d'%oi].append(mask_transform(images['first_frame_gt'][bi][0,oi], size))
                    # req_images['Mask_X8_%d'%oi].append(mask_transform(images['first_frame_gt'][bi][0,oi], size))
                    # req_images['Mask_X16_%d'%oi].append(mask_transform(images['first_frame_gt'][bi][0,oi], size))
                else:
                    req_images['Mask_%d'%oi].append(mask_transform(images['masks_%d'%ti][bi][oi], size))
                    # req_images['Mask_%d'%oi].append(mask_transform(images['masks_%d'%ti][bi][oi][2], size))
                    # req_images['Mask_X8_%d'%oi].append(mask_transform(images['masks_%d'%ti][bi][oi][1], size))
                    # req_images['Mask_X16_%d'%oi].append(mask_transform(images['masks_%d'%ti][bi][oi][0], size))
                req_images['GT_%d_%s'%(oi, GT_suffix)].append(mask_transform(images['cls_gt'][bi,ti,0]==(oi+1), size))
                # print((images['cls_gt'][bi,ti,0]==(oi+1)).shape)
                # print(mask_transform(images['cls_gt'][bi,ti,0]==(oi+1), size).shape)


    return get_image_array(req_images, size, key_captions)

def _check_if_black_and_white(img: Image.Image):
    unique_colors = img.getcolors()
    if len(unique_colors) > 2:
        return False

    if len(unique_colors) == 1:
        return True  # just a black image

    for _, color_rgb in unique_colors:
        if color_rgb == (255, 255, 255):
            return True

    return False

def create_overlay(img: Image.Image, mask: Image.Image, mask_alpha=0.5, color_if_black_and_white=(255, 255, 255)):  # all RGB; Use (128, 0, 0) to mimic DAVIS color palette if you want
    mask = mask.convert('RGB')
    is_b_and_w  = _check_if_black_and_white(mask) 

    if img.size != mask.size:
        mask = mask.resize(img.size, resample=Image.NEAREST)

    mask_arr = np.array(mask)

    if is_b_and_w:
        mask_arr = np.where(mask_arr, np.array(color_if_black_and_white), mask_arr).astype(np.uint8)
        mask = Image.fromarray(mask_arr, mode='RGB')

    alpha_mask = np.full(mask_arr.shape[0:2], 255)
    alpha_mask[cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY) > 0] = int(mask_alpha * 255)  # 255 for black (to keep original image in full), `mask_alpha` for predicted pixels

    overlay = Image.composite(img, mask, Image.fromarray(alpha_mask.astype(np.uint8), mode='L'))

    return overlay

def save_image(img: Image.Image, frame_name, video_name, general_dir_path, sub_dir_name='masks', extension='.png'):
    this_out_path = os.path.join(general_dir_path, video_name, sub_dir_name)
    os.makedirs(this_out_path, exist_ok=True)

    img_save_path = os.path.join(this_out_path, frame_name[:-4] + extension)
    img.save(img_save_path)
    # cv2.imwrite(img_save_path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

class ParallelImageSaver:
    """
    A class for parallel saving of masks and / or overlay images using multiple processes.
    Composing overlays and saving images on the drive is pretty slow, this class does it in the background.

    Parameters
    ----------
    general_output_path : str
        The general path where images and masks will be saved.
    vid_name : str
        The name of the video or identifier for the output files.
    overlay_color_if_b_and_w : tuple, optional
        The RGB color to use for masks when there is only one object. Default is (255, 255, 255) (white).
    max_queue_size : int, optional
        The maximum size of the mask and overlay queues. Default is 200.

    Methods
    -------
    save_mask(mask, frame_name)
        Start saving a mask in the background.
    save_overlay(orig_img, mask, frame_name)
        Create an overlay given an image and a mask, and start saving it in the background.
    qsize() -> Tuple(int, int)
        Get the current size of the mask and overlay queues (how many frames are still left to process).
    __enter__()
        Enter the context manager and return the instance itself.
    __exit__(exc_type, exc_value, exc_tb)
        Exit the context manager and handle cleanup.
    wait_for_jobs_to_finish(verbose=False)
        Wait for all saving jobs to finish. Optional, will be called automatically in __exit__. Only recommened to use if you want to print verbose progress.

    Examples
    --------
    # Example usage of ParallelImageSaver class
    with ParallelImageSaver("/output/directory", "video_1", overlay_color_if_b_and_w=(100, 100, 100)) as image_saver:
        image = Image.open("img.jpg")
        mask = Image.open("mask.png")

        # These will be saved in parallel in background processes
        image_saver.save_mask(mask_image, "frame_000001")
        image_saver.save_overlay(image, mask, "frame_000001")

        image_saver.wait_for_jobs_to_finish(verbose=True)  # Optional

    # The images will be saved in separate processes in the background.
    """

    def __init__(self, general_output_path: str, vid_name: str, overlay_color_if_b_and_w=(255, 255, 255), max_queue_size=200) -> None:
        self._mask_queue = Queue(max_queue_size)
        self._overlay_queue = Queue(max_queue_size)

        self._mask_saver_worker = None
        self._overlay_saver_worker = None

        self._p_out = Path(general_output_path)
        self._vid_name = vid_name
        self._object_color = overlay_color_if_b_and_w
        self._finished = Value('b', False)

    def save_mask(self, mask: Image.Image, frame_name: str):
        self._mask_queue.put((mask, frame_name, 'masks', '.png'))

        if self._mask_saver_worker is None:
            self._mask_saver_worker = Process(target=self._save_mask_fn)
            self._mask_saver_worker.start()
    
    def save_overlay(self, orig_img: Image.Image, mask: Image.Image, frame_name: str):
        self._overlay_queue.put((orig_img, mask, frame_name, 'overlay', '.jpg'))

        if self._overlay_saver_worker is None:
            self._overlay_saver_worker = Process(target=self._save_overlay_fn)
            self._overlay_saver_worker.start()

    def _save_mask_fn(self):
        while True:
            try:
                mask, frame_name, subdir, extension = self._mask_queue.get_nowait()
            except queue.Empty:
                if self._finished.value:
                    return
                else:
                    time.sleep(1)
                    continue
            save_image(mask, frame_name, self._vid_name, self._p_out, subdir, extension)

    def _save_overlay_fn(self):
        while True:
            try:
                orig_image, mask, frame_name, subdir, extension = self._overlay_queue.get_nowait()
            except queue.Empty:
                if self._finished.value:
                    return
                else:
                    time.sleep(1)
                    continue
            overlaid_img = create_overlay(orig_image, mask, color_if_black_and_white=self._object_color)
            save_image(overlaid_img, frame_name, self._vid_name, self._p_out, subdir, extension)

    def qsize(self):
        return self._mask_queue.qsize(), self._overlay_queue.qsize()
    
    def __enter__(self):
        # No need to initialize anything here
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            # Just kill everything for cleaner exit
            # Yeah, the child processed should be immediately killed if the main one exits, but just in case
            if self._mask_saver_worker is not None:
                self._mask_saver_worker.kill()
            if self._mask_saver_worker is not None:
                self._mask_saver_worker.kill()

            raise exc_value
        else:   
            self.wait_for_jobs_to_finish(verbose=False)
            if self._mask_saver_worker is not None:
                self._mask_saver_worker.close()
            
            if self._overlay_saver_worker is not None:
                self._overlay_saver_worker.close()
    
    def wait_for_jobs_to_finish(self, verbose=False):
        # Optional, no need to call unless you want the verbose output
        # Will be called automatically by the __exit__ method
        self._finished.value = True  # No need for a lock, as it's a single write with multiple reads
        
        if not verbose:
            if self._mask_saver_worker is not None:
                self._mask_saver_worker.join()
            
            if self._overlay_saver_worker is not None:
                self._overlay_saver_worker.join()
                
        else:
            while True:
                masks_left, overlays_left = self.qsize()
                if max(masks_left, overlays_left) > 0:
                    print(f"Finishing saving the results, {masks_left:>4d} masks and {overlays_left:>4d} overlays left.")
                    time.sleep(1)
                else:
                    break
            
            self.wait_for_jobs_to_finish(verbose=False)  # just to `.join()` them both
            print("All saving jobs finished")