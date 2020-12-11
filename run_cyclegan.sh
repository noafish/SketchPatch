#!/bin/bash



dataset=''
name=''
iter=50 #100


while getopts 'd:n:i:' flag; do
  case "${flag}" in
	d) dataset="${OPTARG}" ;;
	n) name="${OPTARG}" ;;
	i) iter="${OPTARG}" ;;
  esac
done


python3 train.py --dataroot $dataset --name $name --model cycle_gan --niter $iter --niter_decay $iter --load_size 64 --crop_size 64 --input_nc 1 --output_nc 1 --batch_size 32 --no_flip --display_id -1
python3 test.py --dataroot $dataset/trainA --name $name --model test --model_suffix _A --no_dropout --save_dir checkpoints/$name --load_size 64 --crop_size 64 --input_nc 1 --output_nc 1 --batch_size 64 --display_id -1 --no_flip

sp_ddir=datasets/$name

if [ ! -d $sp_ddir ] 
then
	mkdir -p $sp_ddir
fi

faked=`realpath results/$name/fake`
reald=`realpath results/$name/real`

ln -s $faked $sp_ddir/plain
ln -s $reald $sp_ddir/styled
