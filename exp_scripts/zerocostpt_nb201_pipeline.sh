#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-0}
gpu=${gpu:-"auto"}
pool_size=${pool_size:-10}
batch_size=${batch_size:-256}
edge_decision=${edge_decision:-'random'}
validate_rounds=${validate_rounds:-100}
metric=${metric:-'jacob'}
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset
echo 'gpu:' $gpu

cd ../nasbench201/
python3 networks_proposal.py \
    --dataset $dataset \
    --save $id --gpu $gpu --seed $seed \
    --edge_decision $edge_decision --proj_crit $metric \
    --batch_size $batch_size\
    --pool_size $pool_size \

cd ../zerocostnas/
python3 post_validate.py\
    --ckpt_path ../experiments/nas-bench-201/prop-$id-$seed-$pool_size-$metric\
    --save $id --seed $seed --gpu $gpu\
    --edge_decision $edge_decision --proj_crit $metric \
    --batch_size $batch_size\
    --validate_rounds $validate_rounds\
