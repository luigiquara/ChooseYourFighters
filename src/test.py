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
    for model_name in ['NTSNET', 'SWSL_ResNext', 'HybridNets', 'UNet']:#, 'ViT_IN', 'ViT_food101']:
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

    print(output_shapes)

test_model()
