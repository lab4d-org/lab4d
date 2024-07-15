seqname=$1
lab4d_path=$2
field_type=$3 # fg
data_prefix=$4 # crop
dev=$5
batchsize=2
fg_motion=bob

# dynamic singlecam
# dev=0
#seqname=cat-pikachu-0
# seqname=home-2023-curated3
# logname=gsplat-ref-lab4d-shadowrgb-opt-sync
logname=diffgs-$field_type-b$batchsize-fb-2x
# lab4d_path=logdir/home-2023-curated3-compose-ft/opts.log
# lab4d_path=logdir/cat-pikachu-0-fg-skel/opts.log
# lab4d_path=logdir/cat-pikachu-0-comp/opts.log
# lab4d_path=logdir/2024-05-07--19-25-33-fg-urdf-sync-fix-sm-ft/opts.log
# lab4d_path=logdir/Oct5at10-49AM-poly-bg/opts.log
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu $batchsize --field_type $field_type --data_prefix $data_prefix --eval_res 256 \
  --num_rounds 20 --iters_per_round 200 --learning_rate 5e-3 \
  --feature_type cse --intrinsics_type const --extrinsics_type mlp --fg_motion $fg_motion \
  --use_init_cam --lab4d_path $lab4d_path --use_timesync \
  --reg_arap_wt 0.0 --num_pts 20000 --depth_wt 0.0 \
  --feature_wt 0 --xyz_wt 0
  # --depth_wt 0.01 --flow_wt 0.1
  # --flow_wt 0 --depth_wt 0.1
  # --load_path logdir/$seqname-gsplat-ref-lab4d-comp3/ckpt_latest.pth
  # --flow_wt 0.0 \
  # --bg_vid 0 # --guidance_zero123_wt 2e-4
  # --flow_wt 0.1 --reg_arap_wt 1.0 \
  # --extrinsics_type image --fg_motion image --reg_lab4d_wt 1.0
# python projects/diffgs/render.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
# python projects/diffgs/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
