#!/bin/bash

key=''

# Parsing keyword arguments
while [ $# -gt 0 ]; do
  if [ -z "${key}" ]; then
    case "$1" in
      -v|--video)
        key="vid_abs_host"
        ;;
      -m|--masks)
        key="masks_abs_host"
        ;;
      -o|--output)
        key="output_abs_host"
        ;;
      *)
        printf "***************************\n"
        printf "* Error: Invalid argument ${1}.*\n"
        printf "* Specify -v|--video <video file/frame folder path>.*\n"
        printf "* Specify -m|--masks <existing masks folder path (all will be used)>.*\n"
        printf "* Specify -o|--output <output path>.*\n"
        printf "***************************\n"
        exit 1
    esac
  else
    if [ "${key}" = 'output_abs_host' ]; then 
      value=$(realpath -m "${1}") # output path doesn't have to exist
    else
      value=$(realpath -e "${1}") || exit 1 # video and masks paths must exist
    fi
    export "${key}"="${value}"
    key=''
  fi
  shift
done


echo -e "${vid_abs_host}\n${masks_abs_host}\n${output_abs_host}"


sudo docker run --runtime=nvidia -it --rm \
  -v ${vid_abs_host}:${vid_abs_host} \
  -v ${vid_abs_host}:${vid_abs_host} \
  -v ${masks_abs_host}:${masks_abs_host} \
  -v ${output_abs_host}:${output_abs_host} \
  max810/xmem2:base-inference \
  python3 process_video.py --video ${vid_abs_host} --masks ${masks_abs_host} --output ${output_abs_host}