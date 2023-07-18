# WIP by Gengshan Yang
# python scripts/run_rendering_parallel.py logdir/dog-98-category-comp/opts.log 0-2 0,1,2
import sys
import subprocess

# Set the flagfile.
flagfile = sys.argv[1]

# Set the range of inst_ids.
start_inst_id, end_inst_id = map(int, sys.argv[2].split("-"))

# Set the devices id
dev_list = sys.argv[3].split(",")
dev_list = list(map(int, dev_list))
num_devices = len(dev_list)

print(
    "rendering from inst_id",
    start_inst_id,
    "to",
    end_inst_id,
    "on devices",
    dev_list,
)

# Loop over each device.
for device in dev_list:
    # Initialize an empty command list for this device.
    command_for_device = []

    # Loop over the inst_ids assigned to this device.
    for inst_id in range(device, end_inst_id + 1, num_devices):
        # Add the command for this inst_id to the device's command list.
        command_for_device.append(
            f"CUDA_VISIBLE_DEVICES={device} python lab4d/render.py --flagfile={flagfile} --load_suffix latest --inst_id {inst_id} --render_res 256"
        )

        # Add a delay between commands to avoid overloading the device.
        command_for_device.append("sleep 1")

    # Join all commands for this device into a single string.
    command_str = "; ".join(command_for_device)

    # Start a screen session for this device, executing the device's command string.
    subprocess.Popen(
        f'screen -S render-{device} -d -m bash -c "{command_str}"', shell=True
    )
