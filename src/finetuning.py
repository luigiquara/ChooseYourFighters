import argparse
from datetime import datetime

from torch import nn
from torch.optim import Adam

from loader import Loader 
from training import Trainer

class BackboneWithClassifier(nn.Module):
    def __init__(self, backbone, n_classes):
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))

def finetune(model, train_loader, val_loader, num_classes, max_epochs, lr, save_path, device, log):
    c_model = BackboneWithClassifier(model, num_classes)

    optimizer = Adam(c_model.parameters(), lr=lr)
    trainer = Trainer(c_model, nn.CrossEntropyLoss(), optimizer, save_path=save_path, device=device, log=log)
    results = trainer.train(train_loader, val_loader, num_classes=num_classes, epochs=max_epochs)

    return results


def run(params):
    l = Loader()

    # load the specified model, or all of them if not specified
    if params.models_name:
        s, n, m, t = l.load_model(params.models_name, params.device)
        sources = [s]
        m_names = [n]
        models = [m]
        transforms = [t]
    else: sources, m_names, models, transforms = l.load_all_model(params.device)

    # load the specified dataset, or all of them if not specified
    if params.datasets_name:
        n, d = l.load_dataset(params.datasets_name, params.dataset_args)
        d_names = [n]
        datasets = [d]
    else: d_names, datasets = l.load_all_datasets()

    # for every combination of loaded models and datasets
    # finetune a classifier and log the results
    for source, m_name, model, transform in zip(sources, m_names, models, transforms):
        for d_name, dataset in zip(d_names, datasets):
            print(f'Using {m_name} on {d_name}')

            if params.log:
                run = wandb.init(
                  project='Fighters',
                  config = {
                    'model': m_name,
                    'dataset': d_name,
                    'max_epochs': param.max_epochs,
                    'batch_size': param.batch_size,
                    'lr': param.lr
                })
                save_path = params.save_root + run.name
            else: save_path = params.save_root + datetime.now().strftime('%H%M%S')

            dataset, collate_fn = l.apply_transform(dataset, transform)
            train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, collate_fn=collate_fn)
            val_loader = DataLoader(dataset['val'], batch_size=params.batch_size, collate_fn=collate_fn)
            num_classes = len(dataset['train'].unique('label'))

            finetune(model, train_loader, val_loader, num_classes, params.max_epochs, params.lr, save_path, params.device, params.log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='*', dest='models_name', type=str)
    parser.add_argument('--dataset', nargs='*', dest='datasets_name', type=str)
    parser.add_argument('--dataset_args', type=str)

    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--no-log', action='store_false', dest='log')
    parser.add_argument('--save_root', type=str, default='./fighters/')

    params = parser.parse_args()
    run(params)