# envname=Oct31at1-13AM-poly
envname=Oct5at10-49AM-poly
# envname=home-two
# vidname=2023-11-03--20-46-57
# vidname=cat-pikachu-6
# dev=2
vidname=$1
dev=$2

# # envname, dev
# bash projects/csim/recon_scene.sh $envname $dev

# python preprocess/scripts/canonical_registration.py $vidname-0000 256 quad

# # run camera inference on the scene
# python projects/predictor/inference.py --flagfile=logdir/predictor-comb-dino-rot-aug6-highres-b256-max-uniform-fixcrop2-img-ft4/opts.log \
#   --load_suffix latest --image_dir database/processed/JPEGImages/Full-Resolution/$vidname-0000/
# python projects/csim/transform_bg_cams.py $vidname-0000

# # envname, actorname, dev
# bash projects/csim/align_scene.sh $envname $vidname $dev

# envname, dev
bash projects/csim/recon_compose.sh $vidname $dev