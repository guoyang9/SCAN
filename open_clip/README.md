# OpenCLIP - SCAN
## Data
To download datasets as webdataset, we recommend [img2dataset](https://github.com/rom1504/img2dataset).

### Conceptual Captions

See [cc3m img2dataset example](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md).
In total, we have downloaded 2683008 samples for CC3M, and 8677523 for CC12M.

#### We implement our method using webdataset api only, not for csv as we found it is extremely slow!

## Pre-Training Using A Single Node (CC12M+)

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node 4 -m training.main 
  --save-frequency 4 
  --report-to tensorboard 
  --train-data='/path/to/cc12m/{00000..01242}.tar::/path/to/mscoco/{00000..00059}.tar::/path/to/sbucaptions/{00000..00099}.tar' 
  --dataset-type webdataset 
  --train-num-samples 10108501 
  --batch-size=200 
  --lr=1e-3 
  --wd=0.1 
  --epochs=32 
  --model ViT-B-16
```

Note: 
1) We implement clip loss only, leaving others such as coca untouched.
2) We did not implement gradient checkpointing for the current version.

## Zero-shot ImageNet Evaluation
```bash
python -m training.main \
  --imagenet-val /path/to/imagenet/validation \
  --model ViT-B-32 \
  --pretrained /path/
```

## Downstream Fine-Tuning
As advised by OpenCLIP, we leveraged [WiSE-FT repository](https://github.com/mlfoundations/wise-ft) for fine-tuning the pre-trained models.
The paper on [Robust Fine-tuning of Zero-shot Models](https://arxiv.org/abs/2109.01903).
We slightly changed the settings in wise-ft, and used only their training hyper-paramters.
For example, we abandoned the use for fusion.


## Acknowledgments
[OpenCLIP](https://github.com/mlfoundations/open_clip)

