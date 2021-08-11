#!/bin/bash

micromamba activate deadtrees 
wandb agent $1 --project $2
