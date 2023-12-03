# bash scripts/train.sh lab4d/train.py 0 --seqname 2023-03-26-00-39-17-cat-pikachu
main_func=$1
dev=$2
add_args=${*: 3:$#-1}

ngpu=`echo $dev |  awk -F '[\t,]' '{print NF-1}'`
ngpu=$(($ngpu + 1 ))
echo "using "$ngpu "gpus"

# assign random port
# https://github.com/pytorch/pytorch/issues/73320
# torchrun \
CUDA_VISIBLE_DEVICES=$dev torchrun \
        --nproc_per_node $ngpu --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
        $main_func \
        --ngpu $ngpu \
        $add_args
