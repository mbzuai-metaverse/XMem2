from functools import partial

import torch
import torchvision.transforms.functional as FT
from torchvision.transforms import ColorJitter, Grayscale, RandomPosterize, RandomAdjustSharpness, ToTensor, RandomAffine
from tqdm import tqdm

from inference.data.video_reader import Sample


def extract_keys(dataloder, processor, print_progress=False, flatten=True, **kwargs):
    frame_keys = []
    shrinkages = []
    selections = []
    device = None
    with torch.no_grad():  # just in case
        key_sum = None

        for ti, data in enumerate(tqdm(dataloder, disable=not print_progress, desc='Calculating key features')):
            data: Sample = data
            rgb = data.rgb.cuda()
            key, shrinkage, selection = processor.encode_frame_key(rgb)

            if key_sum is None:
                device = key.device
                # to avoid possible overflow
                key_sum = torch.zeros_like(
                    key, device=device, dtype=torch.float64)

            key_sum += key.type(torch.float64)

            if flatten:
                key = key.flatten(start_dim=2)
                shrinkage = shrinkage.flatten(start_dim=2)
                selection = selection.flatten(start_dim=2)

            frame_keys.append(key.cpu())
            shrinkages.append(shrinkage.cpu())
            selections.append(selection.cpu())

        num_frames = ti + 1  # 0 after 1 iteration, 1 after 2, etc.

        return frame_keys, shrinkages, selections, device, num_frames, key_sum


WhichAugToPick = -1


def get_determenistic_augmentations(img_size=None, mask=None, subset: str = None):
    assert subset in {'best_3', 'best_3_with_symmetrical',
                      'best_all', 'original_only', 'all'}

    bright = ColorJitter(brightness=(1.5, 1.5))
    dark = ColorJitter(brightness=(0.5, 0.5))
    gray = Grayscale(num_output_channels=3)
    reduce_bits = RandomPosterize(bits=3, p=1)
    sharp = RandomAdjustSharpness(sharpness_factor=16, p=1)
    rotate_right = RandomAffine(degrees=(30, 30))
    blur = partial(FT.gaussian_blur, kernel_size=7)

    if img_size is not None:
        h, w = img_size[-2:]
        translate_distance = w // 5
    else:
        translate_distance = 200

    translate_right = partial(FT.affine, angle=0, translate=(
        translate_distance, 0), scale=1, shear=0)

    zoom_out = partial(FT.affine, angle=0,
                       translate=(0, 0), scale=0.5, shear=0)
    zoom_in = partial(FT.affine, angle=0, translate=(0, 0), scale=1.5, shear=0)
    shear_right = partial(FT.affine, angle=0,
                          translate=(0, 0), scale=1, shear=20)

    identity = torch.nn.Identity()
    identity.name = 'identity'

    # if mask is not None:
    #     if mask.any():
    #         min_y, min_x, max_y, max_x = get_bbox_from_mask(mask)
    #         h, w = mask.shape[-2:]
    #         crop_mask = partial(FT.resized_crop, top=min_y - 10, left=min_x - 10,
    #                             height=max_y - min_y + 10, width=max_x - min_x + 10, size=(w, h))
    #         crop_mask.name = 'crop_mask'
    #     else:
    #         crop_mask = identity  # if the mask is empty
    # else:
    crop_mask = None

    bright.name = 'bright'
    dark.name = 'dark'
    gray.name = 'gray'
    reduce_bits.name = 'reduce_bits'
    sharp.name = 'sharp'
    rotate_right.name = 'rotate_right'
    translate_right.name = 'translate_right'
    zoom_out.name = 'zoom_out'
    zoom_in.name = 'zoom_in'
    shear_right.name = 'shear_right'
    blur.name = 'blur'

    rotate_left = RandomAffine(degrees=(-30, -30))
    rotate_left.name = 'rotate_left'

    shear_left = partial(FT.affine, angle=0,
                         translate=(0, 0), scale=1, shear=-20)
    shear_left.name = 'shear_left'

    if WhichAugToPick != -1:
        return [img_mask_augs_pairs[WhichAugToPick]]

    if subset == 'best_3':
        img_mask_augs_pairs = [
            # augs only applied to the image
            # (bright, identity),
            # (dark, identity),
            # (gray, identity),
            # (reduce_bits, identity),
            # (sharp, identity),
            (blur, identity),

            # augs requiring modifying the mask as well:
            # (rotate_right, rotate_right),
            # (rotate_left, rotate_left),
            # (translate_right, translate_right),
            # (zoom_out, zoom_out),
            (zoom_in, zoom_in),
            (shear_right, shear_right),
            # (shear_left, shear_left),
        ]

        return img_mask_augs_pairs
    elif subset == 'best_3_with_symmetrical':
        img_mask_augs_pairs = [
            # augs only applied to the image
            # (bright, identity),
            # (dark, identity),
            # (gray, identity),
            # (reduce_bits, identity),
            # (sharp, identity),
            (blur, identity),

            # augs requiring modifying the mask as well:
            # (rotate_right, rotate_right),
            # (rotate_left, rotate_left),
            # (translate_right, translate_right),
            # (zoom_out, zoom_out),
            (zoom_in, zoom_in),
            (shear_right, shear_right),
            (shear_left, shear_left),
        ]

        return img_mask_augs_pairs
    elif subset == 'best_all':
        img_mask_augs_pairs = [
            # augs only applied to the image
            (bright, identity),
            (dark, identity),
            # (gray, identity),
            (reduce_bits, identity),
            (sharp, identity),
            (blur, identity),

            # augs requiring modifying the mask as well:
            (rotate_right, rotate_right),
            (rotate_left, rotate_left),
            # (translate_right, translate_right),
            (zoom_out, zoom_out),
            (zoom_in, zoom_in),
            (shear_right, shear_right),
            (shear_left, shear_left),
        ]

        return img_mask_augs_pairs

    elif subset == 'original_only':
        img_mask_augs_pairs = [
            # augs only applied to the image
            (bright, identity),
            (dark, identity),
            (gray, identity),
            (reduce_bits, identity),
            (sharp, identity),
            (blur, identity),

            # augs requiring modifying the mask as well:
            # (rotate_right, rotate_right),
            # (translate_right, translate_right),
            # (zoom_out, zoom_out),
            # (zoom_in, zoom_in),
            # (shear_right, shear_right),
        ]
    else:
        img_mask_augs_pairs = [
            # augs only applied to the image
            (bright, identity),
            (dark, identity),
            (gray, identity),
            (reduce_bits, identity),
            (sharp, identity),
            (blur, identity),

            # augs requiring modifying the mask as well:
            (rotate_right, rotate_right),
            (rotate_left, rotate_left),
            (translate_right, translate_right),
            (zoom_out, zoom_out),
            (zoom_in, zoom_in),
            (shear_right, shear_right),
            (shear_left, shear_left),
        ]

        if crop_mask is not None:
            img_mask_augs_pairs.append((crop_mask, crop_mask))

        return img_mask_augs_pairs