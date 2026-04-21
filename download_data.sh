#!/bin/bash
# download_data.sh - Downloads Tiny ImageNet and stb_image.h for Linux HPC clusters

echo "Downloading Tiny ImageNet dataset..."
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

echo "Extracting Tiny ImageNet dataset..."
unzip -q tiny-imagenet-200.zip

echo "Downloading stb_image.h..."
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

echo "Done! Data is in ./tiny-imagenet-200 and stb_image.h is in current directory."
