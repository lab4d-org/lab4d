#!/bin/bash
# Prompt user for the downsize factor
echo "Enter the downsize factor (e.g., 0.5 for half the size):"
read factor
# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found, please install it."
    exit
fi
# Create a directory for the downsized videos
mkdir -p downsized_videos
# Loop through all mp4 files in the current directory
for filename in *.mp4; do
    # Define output file name
    output="downsized_videos/$filename"
    # Run ffmpeg to downsize the video
    ffmpeg -i "$filename" -vf "scale=iw*$factor:ih*$factor" -c:a copy "$output"
    echo "Processed $filename"
done
echo "All videos have been downsized by a factor of $factor and saved in the 'downsized_videos' directory."
