#!/usr/bin/env bash

# --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=4 \
# Define the help message
# ^helpMessage=$(cat <<EOF
# ^Usage: $0 [options]

# ^This script runs the Docker image given by name.

# ^Options:
  # ^-h, --help      Display this help message and exit
  # ^-t <name>       Process the name provided with the -t option
  # ^-c <number>  Number of cpus to use

# ^Example:
  # ^$0 -t projectA -cpus 16  Runs an image with name projectA and 16 cpu cores

# ^EOF
# ^)

# ^# Initialize our variables
# ^name=""
# ^c=""

# ^while getopts ":htc:" opt; do
  # ^case ${opt} in
    # ^h )
      # ^echo "$helpMessage"
      # ^exit 0
      # ^;;
    # ^t )
      # ^name="$OPTARG"
      # ^;;
    # ^c )
      # ^c="$OPTARG"
      # ^;;
    # ^\? )
      # ^echo "Invalid Option: -$OPTARG" 1>&2
      # ^exit 1
      # ^;;
    # ^: )
      # ^echo "Invalid Option: -$OPTARG requires an argument" 1>&2
      # ^exit 1
      # ^;;
  # ^esac
# ^done

# ^# Check if the name is provided
# ^if [ -z "$name" ]; then
    # ^echo "No name provided. Use -t option."
# ^else
    # ^echo "Processing name: $name"
# ^fi

	# --mount type=bind,source="/home/lennartz/data/conp-dataset",target=/data/conp-dataset \
docker run \
	-it \
	--net=host \
	--runtime=nvidia \
  --gpus all \
  --cpus=14 \
  --privileged \
	--ipc=host \
	--mount type=bind,source="/home/gomez/data",target=/data/nnUNet_preprocessed \
	--mount type=bind,source="/home/gomez/out",target=/workspace/out \
	--mount type=bind,source="/home/gomez/Data/",target=/data/Data \
	--mount type=bind,source="/home/gomez/DSSQ",target=/src/DSSQ \
	testdocker
	#96740278c511

	# -v "$SSH_AUTH_SOCK:$SSH_AUTH_SOCK" -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \

# miccai23_reboot_commit

