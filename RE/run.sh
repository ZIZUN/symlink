##   ex) bash run.sh base 4
##   ex) bash run.sh base 4 ddp 4
model=$1
bsz=$2
ddp=$3
ngpu_ddp=$4

# (DDP) or (nn.dataparallel, cpu)
if [ "${ddp}" = "ddp" ]
then
    cmd="${cmd}python -m torch.distributed.launch --nproc_per_node=${ngpu_ddp} --master_port=12299"
else
    cmd="${cmd}python"
fi

cmd="${cmd} train.py -c data/semeval/train.json -t data/semeval/dev.json --model=${model}\
            -o output/re_model --batch_size ${bsz}  --epochs 10000 --lr 4e-5 --seed 2437
            --input_seq_len 512 --log_freq 10  --accumulate 1"

if [ "${ddp}" = "ddp" ]
then
    cmd="${cmd} --ddp True"
fi

echo $cmd
$cmd

