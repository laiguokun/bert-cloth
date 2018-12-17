#!/bin/bash

bsz_list=(4 10)
epoch_list=(2 3)
lr_list=(5e-5 3e-5 2e-5)

for bsz in "${bsz_list[@]}"
do
    for epoch in "${epoch_list[@]}"
    do
        for lr in "${lr_list[@]}"
        do
                cmd="CUDA_VISIBLE_DEVICES=0,1 nohup python -u main.py --output_dir debug-exp/ --do_train --train_batch_size ${bsz} --do_eval --num_train_epoch ${epoch} --learning_rate ${lr} --bert_model bert-large-uncased > logs/bert-large-${epoch}e-${lr}lr-${bsz}-bsz.out"
                echo $cmd
                eval $cmd
        done
    done
done