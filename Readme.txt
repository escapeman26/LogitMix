## Run
>> python main.py \
--network resnet50 \
--dataset cifar100 \
--batch_size 128 \
--mixmethod logitmix_M \
--weights 1 1 1 \
--dist beta \
--alpha 3 \
--loss mse_mixed_logsoftmax \
--gpu 0


## Setting requirements
>> pip install -r requirements.txt
>> Need to modify datapath in 'datasetload.py'

## Reference code
github.com/weiaicunzai/pytorch-cifar100
github.com/pytorch/vision
https://github.com/clovaai/CutMix-PyTorch
