!/bin/bash -x
datasets="SMD PSM WADI SWaT"
lengths=( 64 128 256 512 1024 )
for d in $datasets
do
  for s in "${lengths[@]}"
  do
    python main.py --setting 0001 --gpu 0 --dataset "$d" --model "swin_unet" --seq_len "$s"
  done
done

datasets="MSL SMAP"
lengths=( 64 128 256 )
for d in $datasets
do
  for s in "${lengths[@]}"
  do
    python main.py --setting 0002 --gpu 0 --dataset "$d" --model "swin_unet" --seq_len "$s"
  done
done
