# Installation
```
git checkout lab4dgs
git submodule update --init --recursive
```
Follow [this](https://lab4d-org.github.io/lab4d/get_started/) to install lab4d in a new conda environment.
Run the following to resolve dependency issue.
```
pip install networkx==2.5
```

# Lab4d-GS
## Run on synthetic data
Download dataset
```
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/4u6saejl01okrhtkq3xdh/eagle-d.zip?rlkey=qjx292weid7uj53ok6aomm6px&st=8whvphtl&dl=0"
```

Train
```
bash projects/diffgs/run_fs_sync.sh eagle-d fg crop bob 1
```

During training, you may also use viser to visualzie the 3d asset interactively

https://github.com/user-attachments/assets/63d3106c-f346-4327-aebc-c017dab2568c  

Results will be saved to `logdir/eagle-d-diffgs-fs-fg-b4-bob-r120-mlp/`

https://github.com/user-attachments/assets/08408f89-3744-47cc-bbc6-12433f7fb158



Post-training GUI
```
python projects/diffgs/visergui.py --flagfile=logdir/eagle-d-diffgs-fs-fg-b4-bob-r120-mlp/opts.log --load_suffix latest --data_prefix crop --render_res 512 --lab4d_path ""
```

## Run on mouse dase (to be released)
```
bash projects/csim/run_sync_fg.sh mouse-1 1,2
bash projects/diffgs/run_sync_refine.sh mouse-1 logdir/mouse-1-bob/opts.log fg crop 1,2
```