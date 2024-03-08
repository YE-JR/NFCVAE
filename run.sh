#! /bin/bash
# 用来顺序运行10个风场的训练
if [ q$2 = q ]
	then
		model_name="FCVAE"
	else
		model_name=$2
    fi
for zone in Zone1 Zone2 Zone3 Zone4 Zone5 Zone6 Zone7 Zone8 Zone9 Zone10
do 
    echo $model_name $zone 
    python run.py $zone $model_name $1
done