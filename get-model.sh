#! /bin/bash

# Exit on any error
set -e

# Check args
[ -z $1 ] && echo "usage: $0 <model-name| e.g. mobilenet/float/050/model-stride16> " && exit 1

# Define variables
MODEL_NAME=$1
DIR_NAME=$(echo posenet_${MODEL_NAME} | tr "/" "_")

echo MODEL_NAME: ${MODEL_NAME}
mkdir ${DIR_NAME}

# Fetch model.json and weights.bin
pushd ${DIR_NAME}
wget -c https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/${MODEL_NAME}.json -O model.json
cat model.json | jq -r ".weightsManifest | map(.paths) | flatten | @csv" | tr "," "\n" | xargs -I% wget -c https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/${MODEL_NAME%/*}/%
popd

# Convert to the tf_frozen_model format
tfjs_graph_converter ${DIR_NAME} ${DIR_NAME}.pb
