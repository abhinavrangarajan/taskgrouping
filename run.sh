#!/bin/bash

# Change settings here
# data_dir="../taskonomy-small/"
#data_dir="/home/ar562/taskonomy-sample-model-1/"
data_dir="/home/naagnes/taskonomy_full_dataset/"
train_models="trainN_models.txt"
val_models="val_models.txt"
#model_name="xception_taskonomy_new"
model_name="resnet18_taskonomy"
#model_name="bugnet_taskonomy"
tasks_to_train_on="dnkts"
model_dir="test_run/"
number_of_workers="1"
experiment_name="dry_run"
batch_size="5" # 5 is good for xception with 5 tasks.
epochs="20"
partition=1

# Compile command string
command="python train_taskonomy.py "
command+="--data_dir $data_dir "
#command+="-tm $train_models "
command+="-vm $val_models "
command+="--arch $model_name "
command+="--tasks $tasks_to_train_on "
command+="--model_dir $model_dir "
command+="--workers $number_of_workers "
command+="--experiment_name $experiment_name "
command+="--batch-size $batch_size "
command+="--epochs $epochs "
command+="--wd 0.01 "
if [ $partition -ge 1 ] 
then
	command+="--partition "
	train_models="train5N_models.txt"
fi

command+="-tm $train_models "
# command+="--resume test_run/dry_run_resnet18_taskonomy_np_dnkts_checkpoint.pth.tar "

# Run command string
echo "Running command: $command"
eval $command
