# Agent-to-Sim

## Evaluatio scripts
Registration
```
```

4D reconstrcution
```
python projects/csim/scripts/eval_4drecon.py --flagfile=logdir-neurips-aba/cat-pikachu-2024-08-v2-compose-ft2/opts.log
```
Results should be

## 4D Reconstruction
Test
```
python lab4d/export.py --flagfile=logdir/cat-pikachu-2024-08-v2-compose-ft/opts.log --load_suffix latest --inst_id 23 --vis_thresh -20 --grid_size 128 --data_prefix full 0
```

## Motion Generator
Train
```
bash projects/gdmdm/train.sh cat-pikachu-2024-08-v2-compose-ft b128 128 1
```

Test
```
python projects/gdmdm/long_video_two_agents.py --load_logname cat-pikachu-2024-07-compose-ft --logname_gd b128-past-old --sample_idx 0 --eval_batch_size 1  --load_suffix latest
```