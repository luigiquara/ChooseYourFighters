# Choose Your Fighters
Re-use pretrained networks available online (*e.g.* from PyTorch Hub, HuggingFace) for different downstream tasks.

+ Load upstream models
+ For each downstream task
  + Select top model(s) according to a given metric, *e.g.* sum of top logits, entropy on predicted probabilities, feature norm, etc.
  + Fewshot finetuning
  + Test

## List of upstream models

| Model | Link |
| ------ | --- |
| NTSNET on CUB200 | [download](https://pytorch.org/hub/nicolalandro_ntsnet-cub200_ntsnet/) |
| GPUNet-D2 on ImageNet | [download](https://pytorch.org/hub/nvidia_deeplearningexamples_gpunet/) |
| SWSL ResNext on IG-940M | [download](https://pytorch.org/hub/facebookresearch_semi-supervised-ImageNet1K-models_resnext/) |
| HybridNets on BDD100k | [download](https://pytorch.org/hub/datvuthanh_hybridnets/) |
| U-Net on Brain MRI | [download](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/) |
| ViT on ImageNet | [download](https://huggingface.co/google/vit-base-patch16-224) |
| ViT on food101 | [download](https://huggingface.co/nateraw/food) |

## List of downstream tasks

| Dataset | Link |
| ------- | ---- |
| Imagewoof | [download](https://huggingface.co/datasets/frgfm/imagewoof) |
| ImageNet-Sketch | [download](https://huggingface.co/datasets/imagenet_sketch) |
| snacks | [download](https://huggingface.co/datasets/Matthijs/snacks) |
| Chest XRay Classification | [download](https://huggingface.co/datasets/keremberke/chest-xray-classification) |
| Oxford Flowers | [download](https://huggingface.co/datasets/nelorth/oxford-flowers) |

Too small?
| Dataset | Link |
| ------- | ---- |
| MedMNIST v2 | [download](https://huggingface.co/datasets/albertvillanova/medmnist-v2) |
| SVHN | [download](https://huggingface.co/datasets/svhn) |

