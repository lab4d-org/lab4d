## download data
#bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/j2bztcv49mqc4a6ngj8jp/cat-pikachu-0.zip?rlkey=g2sw8te19bsirr5srdl17kzt1&dl=0"

# reconstruct the scene
# Args: gpu-id, sequence name, hyper-parameters defined in lab4d/config.py
bash scripts/train.sh lab4d/train.py 0 --seqname cat-pikachu-0 --logname bg --field_type bg --data_prefix full --num_rounds 60 --alter_flow --mask_wt 0.01 --normal_wt 1e-2 --reg_eikonal_wt 0.01

# reconstruct the object:
# Args: gpu-id, sequence name, hyper-parameters defined in lab4d/config.py
bash scripts/train.sh lab4d/train.py 0 --seqname cat-pikachu-0 --logname fg-urdf --fg_motion urdf-quad --num_rounds 20 --feature_type cse

# physics-informed optimization
# Args: gpu-id sequence name, hyper-parameters in both lab4d/config.py and and projects/ppr/config.py
bash scripts/train.sh projects/ppr/train.py 0 --seqname cat-pikachu-0 --logname ppr --field_type comp --fg_motion urdf-quad --feature_type cse --num_rounds 20 --learning_rate 1e-4 --pixels_per_image 12 --iters_per_round 100 --secs_per_wdw 2.4 --noreset_steps  --noabsorb_base --load_path logdir/cat-pikachu-0-fg-urdf/ckpt_latest.pth --load_path_bg logdir/cat-pikachu-0-bg/ckpt_latest.pth

# visualization
python projects/ppr/render_intermediate.py --testdir logdir/cat-pikachu-0-ppr/ --data_class sim
python projects/ppr/export.py --flagfile=logdir/cat-pikachu-0-ppr/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --extend_aabb
python lab4d/render_mesh.py --testdir logdir/cat-pikachu-0-ppr/export_0000/ --view bev --ghosting
