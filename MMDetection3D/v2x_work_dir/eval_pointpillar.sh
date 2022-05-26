CUDA_VISIBLE_DEVICES=$1 \
python ../tools/test.py ./config_pointpillar.py \
        ./checkpoints/epoch_80.pth --eval 'mAP'