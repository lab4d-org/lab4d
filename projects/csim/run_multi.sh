# envname=Oct5at10-49AM-poly
# envname=Feb26at10-02 PM-poly
# envname=Feb14at5-55PM-poly
# envname=Feb19at9-47PM-poly
envname=$1
seqname=$2
dev=$3

python projects/csim/rerun_dinov2.py $seqname $dev
bash projects/csim/recon_scene.sh $envname $dev
bash projects/csim/align_scene_multi.sh $envname $seqname $dev
bash projects/csim/recon_compose_multi_fs.sh $seqname urdf-quad $dev # dog, bunny
python projects/csim/visualize/render_videos.py $seqname compose-ft
python projects/csim/visualize/render_videos.py $seqname bg-adapt4