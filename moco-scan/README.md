## MoCo v3 - SCAN

### Introduction
We followed almost the exactly same setting with the original MoCo v3 repo.


### Usage: Self-supervised Pre-Training

Below are two examples for MoCo v3 - SCAN pre-training. 


#### ViT-Small with 1-node (4-GPU) training, batch 600

```bash
python main_moco.py /path/imagenet/
  --multiprocessing-distributed 
  --moco-m-cos 
  --stop-grad-conv1 
  --moco-m-cos 
  --moco-t=.2 
  --optimizer=adamw 
  --lr=1.5e-4 
  --weight-decay=.1 
  --batch-size 600 
  --epoch 100 
  -a vit_small
```

#### ViT-Base with 1-node (4-GPU) training, batch 670

```bash
python main_moco.py /path/imagenet/
  --multiprocessing-distributed 
  --moco-m-cos 
  --stop-grad-conv1 
  --moco-m-cos 
  --moco-t=.2 
  --optimizer=adamw 
  --lr=1.5e-4 
  --weight-decay=.1 
  --batch-size 370 
  --epoch 100 
  -a vit_base
```


#### Notes:
1. The batch size specified by `-b` is the total batch size across all GPUs.
1. The learning rate specified by `--lr` is the *base* lr, and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677) in [this line](https://github.com/facebookresearch/moco-v3/blob/main/main_moco.py#L213).



### Usage: End-to-End Fine-tuning ViT

To perform end-to-end fine-tuning for ViT, use our script to convert the pre-trained ViT checkpoint to [DEiT](https://github.com/facebookresearch/deit) format:
```bash
python convert_to_deit.py \
  --input [your checkpoint path]/[your checkpoint file].pth.tar \
  --output [target checkpoint file].pth
```
Then run the training (in the DeiT repo) with the converted checkpoint:
```bash
python $DEIT_DIR/main.py \
  --resume [target checkpoint file].pth \
  --epochs 150
```

**Note**:
1. We use `--resume` rather than `--finetune` in the DeiT repo, as its `--finetune` option trains under eval mode. When loading the pre-trained model, revise `model_without_ddp.load_state_dict(checkpoint['model'])` with `strict=False`.
1. Our ViT-Small is with `heads=12` in the Transformer block, while by default in DeiT it is `heads=6`. Please modify the DeiT code accordingly when fine-tuning our ViT-Small model. 

### Acknowledgement
[MoCo repo](https://github.com/facebookresearch/moco-v3/tree/main)
