CUDA_VISIBLE_DEVICES=$1 \
python ../tools/test.py ./config_pointpillar.py \
        ./checkpoints/train_all_3epochs_with_encoder.pth --eval 'mAP'