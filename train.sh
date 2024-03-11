#!/bin/bash

for variational in False True
do
	python3 train_model.py --out_n $out_n \
			--drop 0.2 \
			--bnorm True \
			--noise 0.005 \
			--alpha 1e-6 \
			--cdec 0.7 \
			--part_lambda 0.005 \
			--init_lr 0.009 \
			--exist_lambda 0.005 \
			--kl_lambda 1e-5 \
			--kdec 1. \
			--encode_parts True \
			--predict_parts True \
			--predict_part_exist True \
			--variational $variational 
done

for out_n in 128 256 512
do
	python3 train_model.py  --out_n $out_n \
			--drop 0.2 \
			--bnorm True \
			--noise 0.005 \
			--alpha 1e-6 \
			--cdec 0.7 \
			--part_lambda 0.005 \
			--init_lr 0.009 \
			--exist_lambda 0.005 \
			--kl_lambda 1e-5 \
			--kdec 1. \
			--encode_parts True \
			--predict_parts True \
			--predict_part_exist True \
			--variational True 
done

for lambda in 0.0001 0.0005 0.005
do
	python3 train_model.py  --out_n 256 \
			--drop 0.2 \
			--bnorm True \
			--noise 0.005 \
			--alpha 1e-6 \
			--cdec 0.7 \
			--part_lambda $lambda \
			--init_lr 0.009 \
			--exist_lambda $lambda \
			--kl_lambda 1e-5 \
			--kdec 1. \
			--encode_parts True \
			--predict_parts True \
			--predict_part_exist True \
			--variational True 
done
