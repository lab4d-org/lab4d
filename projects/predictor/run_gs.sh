# bash projects/predictor/run_gs.sh 2024-05-07--19-25-33 logdir/2024-05-07--19-25-33-diffgs-fg-b32-bob-r120-ft/opts.log 0,1
# python projects/predictor/inference_3dgs.py --flagfile=logdir/predictor-fromft-2x-mouse-1/opts.log --load_suffix latest --image_dir database/processed/JPEGImages/Full-Resolution/mouse-1-0005/

seqname=predictor-fromft-2x-augd-xyz-unmask-dpt-vis
logname=$1
diffgs_path=$2
dev=$3

rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/predictor/train.py $dev --seqname $seqname --logname $logname --num_rounds 40 \
  --imgs_per_gpu 64 --iters_per_round 50 \
  --diffgs_path $diffgs_path --xyz_wt 0.01

