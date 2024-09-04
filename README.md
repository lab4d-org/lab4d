# Installation
Follow Lab4d guidance.

# Lab4d-GS
## Run on synthetic data
Download dataset
```
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/4u6saejl01okrhtkq3xdh/eagle-d.zip?rlkey=qjx292weid7uj53ok6aomm6px&st=8whvphtl&dl=0"
```

Train
```
bash projects/diffgs/run_fs_sync.sh eagle-d fg crop 1
```

Visualize
```
python projects/diffgs/visergui.py --flagfile=logdir/eagle-d-diffgs-fs-fg-b4-bob-r120-mlp/opts.log --load_suffix latest --data_prefix crop --render_res 512 --lab4d_path ""
```

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