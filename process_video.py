import argparse
import re
from pathlib import Path

from inference.run_on_video import run_on_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video frames given a few (1+) existing annotation masks')
    parser.add_argument('--video', type=str, help='Path to the video file or directory with .jpg video frames to process', required=True)
    parser.add_argument('--masks', type=str, help='Path to the directory with individual .png masks  for corresponding video frames, named `frame_000000.png`, `frame_000123.png`, ... or similarly (the script searches for the first integer value in the filename). '
                        'Will use all masks int the directory.', required=True)
    parser.add_argument('--output', type=str, help='Path to the output directory where to save the resulting segmentation masks and overlays. '
                        'Will be automatically created if does not exist', required=True)

    args = parser.parse_args()

    frames_with_masks = []
    for file_path in (p for p in Path(args.masks).iterdir() if p.is_file()):
        frame_number_match = re.search(r'\d+', file_path.stem)
        if frame_number_match is None:
            print(f"ERROR: file {file_path} does not contain a frame number. Cannot load it as a mask.")
            exit(1)
        frames_with_masks.append(int(frame_number_match.group()))
    
    print("Using masks for frames: ", frames_with_masks)

    p_out = Path(args.output)
    p_out.mkdir(parents=True, exist_ok=True)
    run_on_video(args.video, args.masks, args.output)
