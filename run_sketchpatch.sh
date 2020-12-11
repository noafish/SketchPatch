#!/bin/bash


dataset=''
name=''
recon=''
blur=''
disc=''
rgb='false'
size=64
bsz=64

while getopts 'd:n:r:b:d:cs:z:' flag; do
  case "${flag}" in
    d) dataset="${OPTARG}" ;;
    n) name="${OPTARG}" ;;
    r) recon="${OPTARG}" ;;
    b) blur="${OPTARG}" ;;
    d) disc="${OPTARG}" ;;
    c) rgb='true' ;;
    s) size="${OPTARG}" ;;
    z) bsz="${OPTARG}" ;;
  esac
done


args=''
nc=1

if [ "$recon" != '' ]
then
    args="$args --w_recon $recon"
fi

if [ "$blur" != '' ]
then
    args="$args --w_blur $blur"
fi

if [ "$disc" != '' ]
then
    args="$args --w_disc $disc"
fi

if [ $rgb == 'true' ]
then
    nc=3
fi



python3 train.py --checkpoints_dir checkpoints_sp --dataroot $dataset/styled --model sketchpatch --dataset_mode sketchpatch --name $name $args --load_size $size --crop_size $size --input_nc $nc --output_nc $nc --batch_size $bsz --no_flip --display_id -1
