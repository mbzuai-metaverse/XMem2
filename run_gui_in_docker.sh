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
    if [ "${key}" = '--num_objects' ]; then
        export "num_objects=${1}"
    else
        value=$(realpath -e "${1}") # input path must exist
        export "${key}=${value}"
    fi
    key=''
  fi
  shift
done

if [ -z "${key_name}" ]; then
  echo "Missing argument: --images|--video|--workspace - path to the input video/frames."
  >&2 echo "Error: one or more arguments missing, exiting..."
  exit 1
fi

# An array of arguments instead of just a string
# To avoid bash quoting (e.g. instead of 'python3 file.py --arg val' -> python3 file.py --arg val)
# This way Docker can parse a command and it's arguments separately
# Otherwise it will try to run a command named 'python3 file.py --arg val', which obviously is incorrect
args=("${key_name}" "${value}")  # $key_name would be one of 'images' 'video' or 'workspace'

if [ -n "${num_objects}" ]; then
    args+=("--num_objects" "${num_objects}")
fi


LOCAL_WORKSPACE_DIR="$(pwd)/workspace"  # Feel free to change if necessary
DISPLAY_TO_USE="${DISPLAY}"  # Feel free to change if necessary

set -x  # To print the exact command that will run

docker run --runtime=nvidia -it --rm \
  -e DISPLAY="${DISPLAY_TO_USE}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${value}":"${value}" \
  -v "${LOCAL_WORKSPACE_DIR}":/app/workspace \
  max810/xmem2:gui \
  python3 interactive_demo.py "${args[@]}"