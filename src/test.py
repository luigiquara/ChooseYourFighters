import random

import torch
from torch.utils.data import DataLoader

from loader import Loader

def test_loader():
    source, model, transform = load_model('NTSNET', 'cpu')
    dataset, collate_fn = load_dataset('imagenet_sketch', transform=transform)

    dl = DataLoader(dataset['train'], batch_size=32, collate_fn=collate_fn)
    imgs, labels = next(iter(dl))

def test_shapes():
    output_shapes = []
    models = ['NTSNET', 'SWSL_ResNext', 'HybridNets', 'UNet']#, 'ViT_IN', 'ViT_food101']
    device = 'cpu'

    for model_name in models:
        source, model, transform = load_model(model_name, device)
        dataset, collate_fn = load_dataset('imagenet_sketch', transform=transform)
        dl = DataLoader(dataset['train'], batch_size=2, collate_fn=collate_fn)

        model.eval()
        model.to(torch.device(device))
        for X, y in dl:
            X = X.to(torch.device(device))
            out = model(X)
            output_shapes.append(out.shape)
            break

    for name, s in zip(models, output_shapes): print(f'{name} - output shape: {s}')

def test_hybridnets():
    model_name = 'HybridNets'
    dataset_name = 'imagenet_sketch'

    source, model, transform = load_model(model_name, 'cuda')
    dataset, collate_fn = load_dataset(dataset_name, transform=transform)
    loader = DataLoader(dataset['train'], batch_size=2, collate_fn=collate_fn)

    model.eval()
    model.to(torch.device('cuda'))
    avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    
    for X, y in loader:
        X = X.to(torch.device('cuda'))
        features = model(X)

        # obtain a single value for each channel
        # (batch_size, 160, height, width) -> (batch_size, 160, 1, 1) -> (batch_size, 160)
        pooled = []
        for feature in features: pooled.append(avgpool(feature).squeeze())
        
        # concat all feature maps
        # (5, batch_size, 160) -> (batch_size, 800)
        out = torch.cat([p for p in pooled], dim=1)

        breakpoint()

def test_unet():
    model_name = 'UNet'
    dataset_name = 'imagenet_sketch'

    source, model, transform = load_model(model_name, 'cuda')
    dataset, collate_fn = load_dataset(dataset_name, transform=transform)
    loader = DataLoader(dataset['train'], batch_size=2, collate_fn=collate_fn)

    model.eval()
    model.to(torch.device('cuda'))
    for X, y in loader:
        X = X.to(torch.device('cuda'))

        out = model(X)
        break

def test_load_all_models():
    device = 'cuda'
    l = Loader()
    sources, models, transformations = l.load_all_models(device)

    assert len(sources) == len(models) == len(transformations) == len(l.supported_models), 'test_load_all_models: lengths do not match'

def test_load_all_datasets():
    device = 'cpu'
    l = Loader()
    m_name = random.choice(l.supported_models)

    _, _, transform = l.load_model(m_name, device)
    datasets= l.load_all_datasets()

    assert len(datasets) == len(l.supported_datasets), 'test_load_all_datasets: lengths do not match'
    breakpoint()

test_load_all_datasets()