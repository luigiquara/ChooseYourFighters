import argparse
from datetime import datetime
import time
from tqdm import tqdm
import wandb

from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from loader import Loader 
from training import Trainer

class BackboneWithMLPClassifier(nn.Module):
    def __init__(self, backbone, n_classes, in_features=None, source=None):
        super().__init__()
        self.source = source
        self.backbone = backbone
        if in_features: self.classifier = nn.Linear(in_features, n_classes)
        else: self.classifier = nn.LazyLinear(n_classes)

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.source == 'torch': return self.classifier(self.backbone(x))
        elif self.source == 'huggingface':
            output = self.backbone(x)
            sequence_output = output[0]
            logits = self.classifier(sequence_output[:, 0, :])

            return logits

def finetune_mlp(model, source, train_loader, val_loader, collate_fn, num_classes, max_epochs, lr, save_path, device, log):
    print('Finetuning...')
    if source == 'torch': in_features = None
    elif source == 'huggingface': in_features = model.config.hidden_size

    mlp_model = BackboneWithMLPClassifier(model, num_classes, in_features, source)

    optimizer = Adam(mlp_model.parameters(), lr=lr)
    trainer = Trainer(mlp_model, nn.CrossEntropyLoss(), optimizer, save_path=save_path, device=device, log=log)
    results = trainer.train(train_loader, val_loader, num_classes=num_classes, epochs=max_epochs)

    return results

def forward_pass(model, loader, source, device):
    features = []
    targets = []
    with torch.no_grad():
        for X, y in tqdm(loader):
            X = X.to(device)
            model.to(device)

            if source == 'torch': features.append(model(X))
            elif source == 'huggingface':
                output = model(X)
                features.append(output[0][:,0,:])
            
            targets.append(y)

    return torch.cat(features), torch.cat(targets)

def finetune_logReg(model, train_loader, val_loader, source, device, log):
    print('Using Logistic Regression')
    tr_features, tr_targets = forward_pass(model, train_loader, source, device)
    val_features, val_targets = forward_pass(model, val_loader, source, device)

    print('LogisticRegression fitting')
    start = time.time()
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(tr_features.numpy(force=True), tr_targets)
    fitting_time = time.time() - start
    print(f'LogReg fitted in {fitting_time} seconds')

    train_acc = classifier.score(tr_features.numpy(force=True), tr_targets)
    val_acc = classifier.score(val_features.numpy(force=True), val_targets)

    print(f'Training accuracy: {train_acc}\nValidation accuracy: {val_acc}')

    if log:
        wandb.log({'training_accuracy': train_acc, 'validation_accuracy': val_acc, 'fitting_time': fitting_time})


def run(params):
    l = Loader()

    # load the specified model, or all of them if not specified
    print('Loading models')
    if params.models_name:
        sources, m_names, models, transforms = [], [], [], []
        for model_name in params.models_name:
            print(model_name)
            s, n, m, t = l.load_model(model_name, params.device)
            sources.append(s)
            m_names.append(n)
            models.append(m)
            transforms.append(t)
    else: sources, m_names, models, transforms = l.load_all_models(params.device)

    # load the specified dataset, or all of them if not specified
    print('Loading datasets')
    if params.datasets_name:
        d_names, datasets = [], []
        for dataset_name in params.datasets_name:
            if params.dataset_args: n, d = l.load_dataset(dataset_name, params.dataset_args)
            else: n, d = l.load_dataset(dataset_name)

            d_names.append(n)
            datasets.append(d)
    else: d_names, datasets = l.load_all_datasets()
    print(d_names)

    # for every combination of loaded models and datasets
    # finetune a classifier and log the results
    for source, m_name, model, transform in zip(sources, m_names, models, transforms):
        for d_name, dataset in zip(d_names, datasets):
            print(f'Using {m_name} on {d_name}')

            if params.log:
                run = wandb.init(
                  project='Fighters-Fewshot',
                  config = {
                    'model': m_name,
                    'classifier': params.classifier_type,
                    'dataset': d_name,
                    'fewshot': params.fewshot,
                    'max_epochs': params.max_epochs,
                    'batch_size': params.batch_size,
                    'lr': params.lr
                })
                save_path = params.save_root + run.name
            else: save_path = params.save_root + datetime.now().strftime('%H%M%S')

            dataset, collate_fn = l.apply_transform(dataset, transform, source)
            num_classes = len(dataset['train'].unique('label'))
            if params.fewshot: dataset = l.subsets_from_dataset(dataset, 100, 20, 20)

            train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, collate_fn=collate_fn)
            val_loader = DataLoader(dataset['val'], batch_size=params.batch_size, collate_fn=collate_fn)

            if params.classifier_type == 'mlp': finetune_mlp(model, source, train_loader, val_loader, collate_fn, num_classes, params.max_epochs, params.lr, save_path, params.device, params.log)
            elif params.classifier_type == 'logisticRegression': finetune_logReg(model, train_loader, val_loader, source, params.device, params.log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='*', dest='models_name', type=str)
    parser.add_argument('--classifier_type', type=str, default='mlp')
    parser.add_argument('--dataset', nargs='*', dest='datasets_name', type=str)
    parser.add_argument('--dataset_args', type=str, default=None)
    parser.add_argument('--fewshot', action='store_true')
    parser.add_argument('--no-fewshot', action='store_false', dest='fewshot')

    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--no-log', action='store_false', dest='log')
    parser.add_argument('--save_root', type=str, default='/disk4/lquarantiello/chooseyourfighters/fighters/')

    params = parser.parse_args()
    run(params)