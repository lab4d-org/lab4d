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

Visualize
```
python projects/diffgs/visergui.py --flagfile=logdir/eagle-d-diffgs-fs-fg-b4-bob-r120-mlp/opts.log --load_suffix latest --data_prefix crop --render_res 512 --lab4d_path ""
```

https://github.com/user-attachments/assets/63d3106c-f346-4327-aebc-c017dab2568c  

