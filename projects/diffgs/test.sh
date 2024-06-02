# dynamic multiview
dev=0
seqname=eagle-d
logname=diffgs-bob
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
  --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 --eval_res 256 \
  --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse --use_timesync \
  --intrinsics_type const --extrinsics_type const --fg_motion bob \
  --reg_least_action_wt 0.0 --reg_arap_wt 0.1