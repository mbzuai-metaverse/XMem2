import json
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import progressbar
from tqdm import tqdm

from util.image_loader import PaletteConverter


def resize_preserve(img, size, interpolation):
    h, w = img.height, img.width
    # Resize preserving aspect ratio
    new_w = (w*size//min(w, h))
    new_h = (h*size//min(w, h))

    resized_img = img.resize((new_w, new_h), resample=interpolation)
    
    return resized_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the project to use (name of the corresponding folder in the workspace). Will be created if doesn\'t exist ', required=True)
    parser.add_argument('--size', type=str, help='The name of the project to use (name of the corresponding folder in the workspace). Will be created if doesn\'t exist ', default=480)
    parser.add_argument('--images', type=str, help='Path to the folder with video frames', required=False)
    parser.add_argument('--masks', type=str, help='Path to the folder with existing masks', required=False)

    args = parser.parse_args()
    p_project = Path('workspace') / str(args.name)
    if p_project.exists():
        print(f"Found the project {args.name} in the workspace.")
    else:
        print(f"Creating new project {args.name} in the workspace.")

    if args.images is not None:
        p_imgs = Path(args.images)
        p_imgs_out = p_project / 'images'
        p_imgs_out.mkdir(parents=True, exist_ok=True)

        if any(p_imgs_out.iterdir()):
            print(f"The project {args.name} already has images in the workspace. Delete them first.")
            exit(0)
        
        img_files = sorted(p_imgs.iterdir())
        
        for i in progressbar.progressbar(range(len(img_files)), prefix="Copying/resizing images..."):
            p_img = img_files[i]
            img = Image.open(p_img)
            resized_img = resize_preserve(img, args.size, Image.Resampling.BILINEAR)
            resized_img.save(p_imgs_out / f'frame_{i:06d}{p_img.suffix}')  # keep the same image format

    if args.masks is not None:
        p_masks = Path(args.masks)
        p_masks_out = p_project / 'masks'
        p_masks_out.mkdir(parents=True, exist_ok=True)

        if any(p_masks_out.iterdir()):
            print(f"The project {args.name} already has masks in the workspace. Delete them first.")
            exit(0)
        
        from util.palette import davis_palette
        palette_converter = PaletteConverter(davis_palette)

        mask_files = sorted(p_masks.iterdir())
        
        for i in progressbar.progressbar(range(len(mask_files)), prefix="Copying/resizing masks; converting to DAVIS color palette..."):
            p_mask = mask_files[i]
            mask = Image.open(p_mask)
            resized_mask = resize_preserve(mask, args.size, Image.Resampling.NEAREST).convert('P')

            index_mask = palette_converter.image_to_index_mask(resized_mask)

            index_mask.save(p_masks_out / f'frame_{i:06d}{p_mask.suffix}')  # keep the same image form

        try:
            with open(p_project / 'info.json') as f:
                data = json.load(f)
        except Exception:
            data = {}
        
        data['num_objects'] = palette_converter._num_objects

        with open(p_project / 'info.json', 'wt') as f_out:
            json.dump(data, f_out, indent=4)

