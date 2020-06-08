#!/bin/bash

# Change settings here
data_dir="/home/arvindsk/taskonomy_alpha/"
val_models="test_models.txt"
model_name="xception_taskonomy_new"
tasks_to_train_on="dnkts"
number_of_workers="1"
batch_size="5" # 5 is good for xception with 5 tasks.
epochs="1"
partition=1 # 5n is 1.

# Compile command string
command="python train_taskonomy.py "
command+="--data_dir $data_dir "
command+="-vm $val_models "
command+="--arch $model_name "
command+="--tasks $tasks_to_train_on "
command+="--workers $number_of_workers "
command+="--batch-size $batch_size "
command+="--epochs $epochs "
if [ $partition -ge 1 ] 
then
	command+="--partition "
	train_models="train5N_models.txt"
fi

command+="--resume $1 "
command+="-tm $val_models "
command+="--test "

# Run command string
echo "Running command: $command"
eval $command
