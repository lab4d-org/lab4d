# Description: Run experiments for PPR

# ******ama-samba
seqname=ama-samba-4v

# download pre-processed data
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/yikd46stoxe8p3m5tvpe1/ama-samba-4v.zip?rlkey=mc78xpctmis3cw6j0gzk84r2f&dl=0"

# alternatively, run this if you want to process raw video
python scripts/run_preprocess.py $seqname human "0,1";

# scene reconstruction
# rm -rf logdir/$seqname-bg
bash scripts/train.sh lab4d/train.py 0 --seqname $seqname --logname bg \
  --field_type bg --data_prefix full --num_rounds 20 --alter_flow --mask_wt 0.01 --normal_wt 1e-2 --reg_eikonal_wt 0.01 --scene_type sep-x --freeze_intrinsics 

# foreground reconstruction
# rm -rf logdir/$seqname-fg-urdf
bash scripts/train.sh lab4d/train.py 0 --seqname $seqname --logname fg-urdf --fg_motion urdf-human --num_rounds 20 --feature_type cse --freeze_intrinsics

# physical reconstruction
# rm -rf logdir/$seqname-ppr
bash scripts/train.sh projects/ppr/train.py 0 --seqname $seqname --logname ppr --field_type comp --fg_motion urdf-human --feature_type cse --scene_type sep-x \
    --num_rounds 20  --iters_per_round 100 --ratio_phys_cycle 0.5 --phys_vis_interval 20 --frame_interval 0.0333 --secs_per_wdw 2.0 --warmup_iters 100 \
    --pixels_per_image 12 --noreset_steps --learning_rate 1e-4 --noabsorb_base \
    --load_path logdir/$seqname-fg-urdf/ckpt_latest.pth \
    --load_path_bg logdir/$seqname-bg/ckpt_latest.pth
    
# export meshes and visualize results, run
python projects/ppr/render_intermediate.py --testdir logdir/$seqname-ppr/ --data_class sim
python projects/ppr/export.py --flagfile=logdir/$seqname-ppr/opts.log --load_suffix latest --inst_id 0 --vis_thresh 0 --extend_aabb
python lab4d/render_mesh.py --testdir logdir/$seqname-ppr/export_0000/ --view bev --ghosting

# *****ama-bouncing
seqname=ama-bouncing-4v

# download pre-processed data
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/hld0yyofjl5gb3hbdnra2/ama-bouncing-4v.zip?rlkey=uzoluxprkm33sryt49726wlee&dl=0"

# alternatively, run this if you want to process raw video
python scripts/run_preprocess.py $seqname human "0,1";

# rm -rf logdir/$seqname-bg
bash scripts/train.sh lab4d/train.py 0 --seqname $seqname --logname bg \
  --field_type bg --data_prefix full --num_rounds 20 --alter_flow --mask_wt 0.01 --normal_wt 1e-2 --reg_eikonal_wt 0.01 --scene_type sep-x --freeze_intrinsics

# rm -rf logdir/$seqname-fg-urdf
bash scripts/train.sh lab4d/train.py 0 --seqname $seqname --logname fg-urdf --fg_motion urdf-human --num_rounds 20 --feature_type cse --freeze_intrinsics

# rm -rf logdir/$seqname-ppr
bash scripts/train.sh projects/ppr/train.py 0 --seqname $seqname --logname ppr --field_type comp --fg_motion urdf-human --feature_type cse --scene_type sep-x \
    --num_rounds 20  --iters_per_round 100 --frame_interval 0.0333 --secs_per_wdw 1.0 --warmup_iters 100 \
    --pixels_per_image 12 --noreset_steps --learning_rate 1e-4 --noabsorb_base \
    --load_path logdir/$seqname-fg-urdf/ckpt_latest.pth \
    --load_path_bg logdir/$seqname-bg/ckpt_latest.pth
  
# export meshes and visualize results, run
python projects/ppr/render_intermediate.py --testdir logdir/$seqname-ppr/ --data_class sim
python projects/ppr/export.py --flagfile=logdir/$seqname-ppr/opts.log --load_suffix latest --inst_id 0 --vis_thresh 0 --extend_aabb
python lab4d/render_mesh.py --testdir logdir/$seqname-ppr/export_0000/ --view bev --ghosting