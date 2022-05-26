# Code for 3D Feature Flow for Vehicle-Infrastructure 3D Object Detection 
This project shows you how we test our model, and we provide 150 instances sampled from our validation dataset for you to redo the experiments in our paper. 
# The structure of this project
    ./test
        /README.md
        /MMDetection3D  
            /feature_flow_work_dir/checkpoints/train_all_2.pth    (checkpoint for FFNet)
            /feature_flow_work_dir/config_pointpillar.py          (config for FFNet)
            /v2x_work_dir/checkpoints/epoch_80.pth                (checkpoint for feature fusion model) 
        /data                                                     (data for testing)            
        /evaluation
            /v2x/script/lidar_feature_flow.sh                     (evaluation script) 
# How to run this code
## Enviriment preparation 
Our evaluation code is run on the revised MMDetection3D, you need get into MMDetection3D folder and run : python setup.py develop.
You might need the following python packages as well:

        tqdm
        numpy
        scipy
        pytorch
        pypcd

## Test the FFNet on VIC-Async-1
To do this, you need to set the "flownet_test_mode" to "FlowPred" in ./test/MMDetection3D/feature_flow_work_dir/config_pointpillar.py.
Then, run the following command in ./test/evaluation/v2x folder:

        bash script/lidar_feature_flow.sh
        
## Test the Feature fusion model on VIC-Async-1
You set "flownet_test_mode" to 'OriginFeat', and run the following command in ./test/evaluation/v2x folder:

        bash script/lidar_feature_flow.sh
## Test the Feature fusion model on VIC-Sync
You set "flownet_test_mode" to 'Async', and run the following command in ./test/evaluation/v2x folder:

        bash script/lidar_feature_flow.sh

## Test the model in different rigions 
We evaluate our model on different rigion including 0-100m, 0-30m, 30-50m, and 50-100m. To do this, you need to change the "extended-range" parameters in  
./test/evaluation/v2x/script/lidar_feature_flow.sh. 

        For 0-100m  :  0 -39.68 -3 100 39.68 1
        For 0-30m   :  0 -39.68 -3 30  39.68 1
        For 30-50m  : 30 -39.68 -3 50  39.68 1
        For 50-100m : 50 -39.68 -3 100 39.68 1

