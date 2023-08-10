#!/bin/bash

docker run xmem2-base-inference --it --rm \
  -v ${1}:${1} \
  -v ${2}:${2} \
  -v ${3}:${3} \
  ${1} ${2} ${3}