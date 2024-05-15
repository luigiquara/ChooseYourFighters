import torch
from torch.utils.data import DataLoader

from loader import load_model, load_dataset

def test_loader():
    source, model, transform = load_model('NTSNET', 'cpu')
    dataset, collate_fn = load_dataset('imagenet_sketch', transform=transform)

    dl = DataLoader(dataset['train'], batch_size=32, collate_fn=collate_fn)
    imgs, labels = next(iter(dl))

def test_model():
    output_shapes = []
    models = ['NTSNET', 'SWSL_ResNext', 'HybridNets', 'UNet']#, 'ViT_IN', 'ViT_food101']
    for model_name in models:
        source, model, transform = load_model(model_name, 'cuda')
        dataset, collate_fn = load_dataset('imagenet_sketch', transform=transform)
        dl = DataLoader(dataset['train'], batch_size=2, collate_fn=collate_fn)

        model.eval()
        model.to(torch.device('cuda'))
        for X, y in dl:
            X = X.to(torch.device('cuda'))
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


test_hybridnets()