#!/bin/bash

for STR in ./configs/to_be_run/*.json; do
  SUB=$1
  if [[ "$STR" == *"$SUB"* ]]; then
    echo "Training with config $STR";
    python train.py --config $STR;
    python inference.py --config $STR;
    mv -v $STR ./configs/al_run;
  fi;
done

