
from proofreader.utils.torch import load_model
from proofreader.model.config import *
from proofreader.model.classifier import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import click
from tqdm import tqdm
from proofreader.model.config import *


def test_model(model, test_dataset, use_gpu=True):
    model.eval()
    metrics = {'seen_acc': 0, 'neurite_acc': 0}
    if use_gpu:
        model = model.cuda()
    with torch.no_grad():

        neurites = 0
        seen = 0
        for step, canidiate_batch in tqdm(enumerate(test_dataset)):

            # get batch
            x, y = canidiate_batch
            bid = y[:, 1]
            y = y[:, 0]
            if use_gpu:
                x = x.cuda(0, non_blocking=True)
                y = y.cuda(0, non_blocking=True)

            print(x.shape, y.shape)
            for i in range(x.shape[0]):
                example = x[i].unsqueeze(dim=0)
                print(example.shape)
                y_hat = model(example)
                pred = predict_class(y_hat)
                true = int(pred == y[i])
                metrics['seen_acc'] += true
                seen += 1
                if pred == 1 or y[i] == 1:
                    neurites += 1
                    metrics['neurite_acc'] += true
                    break

    model.training()

    # average metrics
    metrics['seen_acc'] /= seen
    metrics['neurite_acc'] /= neurites
    return metrics


@click.command()
@click.option('--dataset_path', '-dp',
              type=str,
              )
@click.option('--config', '-c',
              type=str,
              )
@click.option('--checkpoint_path', '-cp',
              type=str,
              help='num slices to drop'
              )
def test(dataset_path: str, config: str, checkpoint_path: str):

    x, y = torch.load(dataset_path)
    # x, y = torch.tensor(x), torch.tensor(y)
    # print(x.shape, y.shape)
    test_dataset = SimpleDataset(x, y, shuffle=False)
    config = get_config(config)
    model, _, _ = build_full_model_from_config(config.model, config.dataset)
    model = nn.DataParallel(model)
    model = load_model(model, checkpoint_path)
    metrics = test_model(model, test_dataset, use_gpu=True)
    print(metrics)


if __name__ == '__main__':
    test()
