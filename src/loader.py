'''Download pretrained models from online hubs, remove classification head and return model
'''
import argparse

import torch
from torchvision import transforms

def load_model(model_name: str, device: str):
    '''Load a given model from an online hub. Return the feature exctractor without classifier
    '''

    if model_name == 'NTSNET':
        from PIL import Image
        from nts_net.model import attention_net

        # little mod to original model class
        # to return only the features before classification
        # i.e., ignore the classification head
        class attention_net_mod(attention_net):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = self.model(x)
                return concat_out

        model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                       **{'topN': 6, 'device': device, 'num_classes': 200})
        model = attention_net_mod(model)

        transform = {
            'train': transforms.Compose([
                transforms.Resize((600, 600), Image.BILINEAR),
                transforms.CenterCrop((448, 448)),
                transforms.RandomHorizontalFlip(),  # only if train
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'eval': transforms.Compose([
                transforms.Resize((600, 600), Image.BILINEAR),
                transforms.CenterCrop((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }

    elif model_name == 'GPUNet-D2':
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True, model_type='GPUNet-D2', model_math='fp32')

        transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }

        #TODO: remove classifier
        raise NotImplementedError

    elif model_name == 'SWSL_ResNext':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
        model = torch.nn.Sequential(*(list(model.children())[:-1]))

        transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }

    elif model_name == 'HybridNets':
        from HybridNets.backbone import HybridNetsBackbone

        class HybridNetsBackboneMod(HybridNetsBackbone):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))

            def forward(self, x):
                features, regression, classification, anchors, segmentation = self.model(x)

                # obtain a single value for each channel
                # (batch_size, 160, height, width) -> (batch_size, 160, 1, 1) -> (batch_size, 160)
                pooled = []
                for feature in features: pooled.append(self.avgpool(feature).squeeze())

                # concat all feature maps
                # (5, batch_size, 160) -> (batch_size, 800)
                out = torch.cat([p for p in pooled], dim=1)

                return out

        model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
        model = HybridNetsBackboneMod(model)

        transform = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'test': transforms.Compose([
                transforms.Resize((256, 256)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }

    elif model_name == 'UNet':
        from brain_segmentation_pytorch.unet import UNet

        class UNetMod(UNet):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
            
            def forward(self, x):
                output, features = self.model(x)
                features = self.avgpool(features)
                return features.squeeze()

        #model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        model = torch.hub.load('brain_segmentation_pytorch', 'unet', source='local', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        model = UNetMod(model)

        transform = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'test': transforms.Compose([
                transforms.Resize((256, 256)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }

    elif model_name == 'ViT_IN':
        from transformers import ViTImageProcessor, ViTModel

        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k') # bare model, without classifier

    elif model_name == 'ViT_food101':
        from transformers import ViTImageProcessor, ViTModel

        processor = ViTImageProcessor.from_pretrained('nateraw/food')
        model = ViTModel.from_pretrained('nateraw/food') # bare model, without classifier

    # PyTorch models have the transform variable
    # HuggingFace models have the processor
    if 'transform' in locals(): return 'torch', model, transform
    elif 'processor' in locals(): return 'huggingface', model, processor

def load_dataset(dataset_name: str, *args, transform=None):
    from datasets import load_dataset

    if args: dataset = load_dataset(dataset_name, args[0], cache_dir='/disk4/lquarantiello/huggingface/datasets/')
    else: dataset = load_dataset(dataset_name, cache_dir='/disk4/lquarantiello/huggingface/datasets/')
    o_len_dataset = sum(dataset.num_rows.values())

    # split the dataset, if not already
    # add validation set, if needed
    if not any(['val' in k for k in dataset.keys()]):
        s = dataset['train'].train_test_split(test_size = 0.2)
        dataset['train'] = s['train']
        dataset['val'] = s['test']

    # add test set, if needed
    if 'test' not in dataset.keys():
        s = dataset['train'].train_test_split(test_size=0.2)
        dataset['train'] = s['train']
        dataset['test'] = s['test']
    
    assert sum(dataset.num_rows.values()) == o_len_dataset, 'something is wrong'

    # extract the three sets
    train_ds = dataset['train']
    for k in dataset.keys():
        if 'val' in k: val_ds = dataset[k]
    test_ds = dataset['test']

    if transform:
        import torch

        def preprocess_train(samples):
            samples['t_image'] = [transform['train'](image.convert('RGB')) for image in samples['image']]
            return samples

        def preprocess_eval(samples):
            samples['t_image'] = [transform['eval'](image.convert('RGB')) for image in samples['image']]
            return samples

        def collate_fn(samples):
            t_images = torch.stack([s['t_image'] for s in samples])
            labels = torch.tensor([s['label'] for s in samples])
            #return {'imgs': t_images, 'labels': labels}
            return (t_images, labels)

        train_ds.set_transform(preprocess_train)
        val_ds.set_transform(preprocess_eval)
        test_ds.set_transform(preprocess_eval)

    return {'train': train_ds, 'val': val_ds, 'test': test_ds}, collate_fn