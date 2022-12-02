import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from inference.data.video_reader import VideoReader
from model.network import XMem
from inference.inference_core import InferenceCore


def inference_on_video(how_many_extra_frames):
    torch.autograd.set_grad_enabled(False)

    config = {
        'buffer_size': 100,
        'deep_update_every': -1,
        'enable_long_term': True,
        'enable_long_term_count_usage': True,
        'fbrs_model': 'saves/fbrs.pth',
        'hidden_dim': 64,
        'images': None,
        'key_dim': 64,
        'max_long_term_elements': 10000,
        'max_mid_term_frames': 10,
        'mem_every': 10,
        'min_mid_term_frames': 5,
        'model': './saves/XMem.pth',
        'no_amp': False,
        'num_objects': 1,
        'num_prototypes': 128,
        's2m_model': 'saves/s2m.pth',
        'size': 480,
        'top_k': 30,
        'value_dim': 512,
        'video': '../VIDEOS/Maksym_frontal_simple.mp4',
        'masks_out_path': f'../VIDEOS/RESULTS/XMem_u2net/lips/MaximumPossibleIoU/{how_many_extra_frames}_extra_frames',
        'workspace': None,
        'save_masks': True,
        'restore_path':  './saves/u2net/u2net.pth'
    }

    model_path = config['model']
    network = XMem(config, model_path).cuda().eval()
    if model_path is not None:
        print("run_on_videol53: disabled loading model - it was for resnet not u2net")
        # model_weights = torch.load(model_path)
        # network.load_weights(model_weights, init_as_zero_if_needed=True)
    else:
        print('No model loaded.')

    total_process_time = 0
    total_frames = 0

    # Start eval
    vid_reader = VideoReader(
        "Maksym_frontal_simple", 
        '/home/maksym/RESEARCH/VIDEOS/IVOS_Maksym_frontal_simple_14_Nov_2022_DATA/JPEGImages',
        '/home/maksym/RESEARCH/VIDEOS/IVOS_Maksym_frontal_simple_14_Nov_2022_DATA/SegmentationMaskBinary',
        size=config['size'],
        use_all_mask=True
    )

    loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=8)
    vid_name = vid_reader.vid_name
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    config['enable_long_term_count_usage'] = (
        config['enable_long_term'] and
        (vid_length
            / (config['max_mid_term_frames']-config['min_mid_term_frames'])
            * config['num_prototypes'])
        >= config['max_long_term_elements']
    )

    mapper = MaskMapper()
    processor = InferenceCore(network, config=config)
    first_mask_loaded = False

    # e.g. [0, 10, 20, ..., 180] without 180
    frames_with_masks = set(np.linspace(0, 180, how_many_extra_frames+2)[0:-1].astype(int))

    for ti, data in enumerate(loader):
        with torch.cuda.amp.autocast(enabled=True):
            rgb = data['rgb'].cuda()[0]
            
            # TODO: - only use % of the frames
            if ti in frames_with_masks:
                msk = data['mask']
            else:
                msk = None
            
            info = data['info']
            frame = info['frame'][0]
            shape = info['shape']
            need_resize = info['need_resize'][0]

            """
            For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
            Seems to be very similar in testing as my previous timing method 
            with two cuda sync + time.time() in STCN though 
            """
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    continue

            if False:
                rgb = torch.flip(rgb, dims=[-1])
                msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            # Map possibly non-continuous labels to continuous ones
            # TODO: What are labels? Debug
            if msk is not None:
                # https://github.com/hkchengrex/XMem/issues/21 just make exhaustive = True
                msk, labels = mapper.convert_mask(msk[0].numpy(), exhaustive=True)
                msk = torch.Tensor(msk).cuda()
                if need_resize:
                    msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))
    
            else:
                labels = None

            # Run the model on this frame
            # TODO: still running inference even on frames with masks?
            prob = processor.step(rgb, msk, labels, end=(ti==vid_length-1), manually_curated_masks=False)

            # Upsample to original size if needed
            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

            end.record()
            torch.cuda.synchronize()
            total_process_time += (start.elapsed_time(end)/1000)
            total_frames += 1

            if False:
                prob = torch.flip(prob, dims=[-1])

            # Probability mask -> index mask
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            if False:
                prob = (prob.detach().cpu().numpy()*255).astype(np.uint8)

            # Save the mask
            if config['save_masks']:
                this_out_path = path.join(config['masks_out_path'], vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

            if False: #args.save_scores:
                np_path = path.join(args.output, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti==len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                if args.save_all or info['save'][0]:
                    hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')


    print(f'Total processing time: {total_process_time}')
    print(f'Total processed frames: {total_frames}')
    print(f'FPS: {total_frames / total_process_time}')
    print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')


if __name__ == '__main__':
    # for i in tqdm(range(0)):
    inference_on_video(0)
