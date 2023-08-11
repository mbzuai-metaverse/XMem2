#!/bin/bash

key=''
key_name=''
value=''

# Parsing keyword arguments
while [ $# -gt 0 ]; do
  if [ -z "${key}" ]; then
    case "$1" in
      --images|--video|--workspace)
        key="other"
        key_name="${1}"
        ;;
      --num_objects)
        key="--num_objects"
        ;;
      *)
        printf "***************************\n"
        printf "* Error: Invalid argument ${1}\n"
        printf "* Specify one of --images --video or --workspace with <video file/frame folder path>.*\n"
        printf "***************************\n"
        exit 1
    esac
  else
    if [ ${key} = '--num_objects' ]; then
        export "num_objects"="${1}"
    else
        value=$(realpath -e "${1}") # input path must exist
        export "${key}"="${value}"
    fi
    key=''
  fi
  shift
done


cmd="python3 interactive_demo.py ${key_name} ${value}"  # $key_name would be one of 'images' 'video' or 'workspace'

if [ ! -z ${num_objects} ]; then
    cmd="${cmd} --num_objects ${num_objects}"
fi

echo "Running ${cmd}"

LOCAL_WORKSPACE_DIR="$(pwd)/workspace"  # Feel free to change if necessary
DISPLAY_TO_USE=${DISPLAY}  # Feel free to change if necessary

sudo docker run --runtime=nvidia -it --rm \
  -e DISPLAY=${DISPLAY_TO_USE} \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${value}:${value} \
  -v ${LOCAL_WORKSPACE_DIR}:/app/workspace \
  max810/xmem2:gui \
  ${cmd}