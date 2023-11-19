#bash scripts/train.sh lab4d/train.py 1,2 --seqname cat-pikachu --logname fg-bob-b120 --fg_motion bob --num_rounds 120 --reg_gauss_skin_wt 0.01
#
#bash scripts/train.sh lab4d/train.py 1,2 --seqname penguin --logname fg-skel-b120 --fg_motion skel-human --num_rounds 120
#


# mv -f logdir/dog-robolounge-fg-comp-b120 logdir/old
# mv -f logdir/squirrel-fg-comp-b120 logdir/old
# mv -f logdir/cat-pikachu-0-fg-skel-b120 logdir/old
# mv -f logdir/cat-pikachu-0-comp-comp-s2 logdir/old
# # mv -f logdir/cat-pikachu-0-bg-b120 logdir/old
# mv -f logdir/car-turnaround-2-fg-rigid-b120 logdir/old
# # mv -f logdir/butterfly-fg-bob-b120 logdir/old
# # mv -f logdir/mochi-toro-fg-comp-b120 logdir/old

# rm -rf logdir/dog-robolounge-fg-comp-b120
# rm -rf logdir/squirrel-fg-comp-b120
# rm -rf logdir/cat-pikachu-0-fg-skel-b120
# rm -rf logdir/cat-pikachu-0-comp-comp-s2
# rm -rf logdir/cat-pikachu-0-bg-b120
rm -rf logdir/car-turnaround-2-fg-rigid-b120-f8
#rm -rf logdir/butterfly-fg-bob-b120
#rm -rf logdir/mochi-toro-fg-comp-b120


# bash scripts/train.sh lab4d/train.py 1 --seqname squirrel --logname fg-comp-b120 --fg_motion comp_skel-quad_dense --num_rounds 120
# bash scripts/train.sh lab4d/train.py 1 --seqname cat-pikachu-0 --logname fg-skel-b120 --fg_motion skel-quad --num_rounds 120
# bash scripts/train.sh lab4d/train.py 1 --seqname cat-pikachu-0 --logname comp-comp-s2 --field_type comp --fg_motion comp_skel-quad_dense --data_prefix full --num_rounds 120 --load_path logdir/cat-pikachu-0-fg-skel-b120/ckpt_latest.pth
# # bash scripts/train.sh lab4d/train.py 1 --seqname cat-pikachu-0 --logname bg-b120 --field_type bg --data_prefix full --num_rounds 120
bash scripts/train.sh lab4d/train.py 1 --seqname car-turnaround-2 --logname fg-rigid-b120-f8 --fg_motion rigid --num_rounds 120
# bash scripts/train.sh lab4d/train.py 1,2 --seqname dog-robolounge --logname fg-comp-b120 --fg_motion comp_skel-quad_dense --num_rounds 120
# # bash scripts/train.sh lab4d/train.py 1,2 --seqname butterfly --logname fg-bob-b120 --fg_motion bob --num_rounds 120 --reg_gauss_skin_wt 0.01
# # bash scripts/train.sh lab4d/train.py 1,2 --seqname mochi-toro --logname fg-comp-b120 --fg_motion comp_skel-quad_dense --num_rounds 120
# # bash scripts/train.sh lab4d/train.py 2 --seqname finch --logname fg-bob-b120 --fg_motion bob --num_rounds 120 --reg_gauss_skin_wt 0.01


#python scripts/run_rendering_parallel.py logdir/squirrel-fg-comp-b120/opts.log 0-0 2
#python scripts/run_rendering_parallel.py logdir/cat-pikachu-0-fg-skel-b120/opts.log 0-0 2
#python scripts/run_rendering_parallel.py logdir/cat-pikachu-0-comp-comp-s2/opts.log 0-0 2
#python scripts/run_rendering_parallel.py logdir/car-turnaround-2-fg-rigid-b120/opts.log 0-0 2
#python scripts/run_rendering_parallel.py logdir/dog-robolounge-fg-comp-b120/opts.log 0-0 2


#bash scripts/train.sh lab4d/train.py 0 --seqname cat-pikachu --logname bg-b120-mmlp-test --field_type bg --data_prefix full --nosingle_inst --num_rounds 120 --reg_visibility_wt 1e-6 --depth_wt 0.01
#bash scripts/train.sh lab4d/train.py 1,2 --seqname cat-pikachu --logname bg-b120-cmlp-large-depth --field_type bg --data_prefix full --nosingle_inst --num_rounds 120 --reg_visibility_wt 1e-6 --depth_wt 0.01
#bash scripts/train.sh lab4d/train.py 1 --seqname cat-pikachu-6 --logname bg-b20-mmlp-depth --field_type bg --data_prefix full --depth_wt 0.01

#python lab4d/render.py --flagfile=logdir/dog-robolounge-fg-comp-b120/opts.log --load_suffix latest --inst_id 0 --render_res 256
#python lab4d/render.py --flagfile=logdir/squirrel-fg-comp-b120/opts.log --load_suffix latest --inst_id 0 --render_res 256
#python lab4d/render.py --flagfile=logdir/cat-pikachu-0-fg-skel-b120/opts.log --load_suffix latest --inst_id 0 --render_res 256
# python lab4d/render.py --flagfile=logdir/cat-pikachu-0-fg-skel-b120/opts.log --load_suffix latest --viewpoint rot-0-360 --render_res 256 --freeze_id 50
# python scripts/render_intermediate.py --testdir logdir/cat-pikachu-0-fg-skel-b120/
# python lab4d/export.py --flagfile=logdir/cat-pikachu-0-fg-skel-b120/opts.log --load_suffix latest
# python lab4d/render.py --flagfile=logdir/cat-pikachu-0-comp-comp-s2/opts.log --load_suffix latest --inst_id 0 --render_res 256 --viewpoint bev-20
# python lab4d/render.py --flagfile=logdir/car-turnaround-2-fg-rigid-b120/opts.log --load_suffix latest --inst_id 0 --render_res 256
# python lab4d/render.py --flagfile=logdir/car-turnaround-2-fg-rigid-b120/opts.log --load_suffix latest --inst_id 0 --render_res 256 --viewpoint bev-90
# python lab4d/render.py --flagfile=logdir/car-turnaround-2-fg-rigid-b120/opts.log --load_suffix latest --inst_id 0 --render_res 256 --viewpoint rot-0-360
# python lab4d/export.py --flagfile=logdir/car-turnaround-2-fg-rigid-b120/opts.log --load_suffix latest
# python scripts/render_intermediate.py --testdir logdir/car-turnaround-2-fg-rigid-b120/

# python lab4d/render.py --flagfile=logdir/mochi-toro-fg-skel-b120/opts.log --load_suffix latest --inst_id 0 --render_res 256 --viewpoint bev-20

# python lab4d/render.py --flagfile=logdir/butterfly-fg-bob/opts.log --load_suffix latest  --inst_id 0 --viewpoint ref --render_res 256
# python lab4d/render.py --flagfile=logdir/butterfly-fg-bob/opts.log --load_suffix latest  --inst_id 1 --viewpoint ref --render_res 256
# python lab4d/render.py --flagfile=logdir/butterfly-fg-bob/opts.log --load_suffix latest  --inst_id 2 --viewpoint ref --render_res 256
# python lab4d/render.py --flagfile=logdir/butterfly-fg-bob/opts.log --load_suffix latest  --inst_id 3 --viewpoint ref --render_res 256
# python lab4d/render.py --flagfile=logdir/butterfly-fg-bob/opts.log --load_suffix latest  --inst_id 4 --viewpoint ref --render_res 256
# python lab4d/render.py --flagfile=logdir/butterfly-fg-bob/opts.log --load_suffix latest  --inst_id 5 --viewpoint ref --render_res 256
# python lab4d/render.py --flagfile=logdir/butterfly-fg-bob/opts.log --load_suffix latest  --inst_id 6 --viewpoint ref --render_res 256
# python lab4d/render.py --flagfile=logdir/butterfly-fg-bob/opts.log --load_suffix latest  --inst_id 7 --viewpoint ref --render_res 256


# python scripts/zip_logdir.py logdir/dog-robolounge-fg-comp-b120
# python scripts/zip_logdir.py logdir/squirrel-fg-comp-b120
# python scripts/zip_logdir.py logdir/cat-pikachu-0-fg-skel-b120
# python scripts/zip_logdir.py logdir/cat-pikachu-0-comp-comp-s2
# python scripts/zip_logdir.py logdir/car-turnaround-2-fg-rigid-b120