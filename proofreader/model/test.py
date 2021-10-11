
from proofreader.utils.torch import load_model
from proofreader.model.config import *
from proofreader.model.classifier import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import click
from tqdm import tqdm
from proofreader.model.config import *
import numpy as np

# def test_model(model, test_dataset, use_gpu=True):
#     model.eval()
#     metrics = {'seen_acc': 0, 'neurite_acc': 0}

#     with torch.no_grad():

#         neurites = 0
#         seen = 0
#         batches = 0
#         for step, canidiate_batch in tqdm(enumerate(test_dataset)):

#             # get batch
#             x, y = canidiate_batch
#             bid = y[:, 1]
#             y = y[:, 0]
#             batches += 1
#             if use_gpu:
#                 x = x.cuda(0, non_blocking=True)
#                 y = y.cuda(0, non_blocking=True)

#             for i in range(x.shape[0]):
#                 example = x[i].unsqueeze(dim=0)
#                 y_hat = model(example)
#                 pred = predict_class(y_hat)
#                 true = int(pred == y[i])
#                 metrics['seen_acc'] += true
#                 seen += 1
#                 if pred == 1 or y[i] == 1:
#                     neurites += 1
#                     metrics['neurite_acc'] += true
#                     break

#     model.train()

#     # average metrics
#     metrics['seen_acc'] /= seen
#     metrics['neurite_acc'] /= neurites
#     metrics['num_batches'] = batches
#     return metrics


def test_model(model, test_dataset, batch_size=256, use_gpu=True, rank=0):

    dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                            pin_memory=use_gpu, drop_last=False, shuffle=True)
    ys, preds, bids = [], [], []
    model.eval()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            # get batch
            x, y = batch
            bid = y[:, 1]
            y = y[:, 0]

            if use_gpu:
                x = x.cuda(rank, non_blocking=True)
                y = y.cuda(rank, non_blocking=True)

            y_hat = model(x)
            pred = predict_class(y_hat)
            ys.append(y)
            preds.append(pred)
            bids.append(bid)

    model.train()

    ys, preds, bids = torch.cat(ys), torch.cat(preds), torch.cat(bids)

    uids = np.unique(bids)
    batch_acc = {'neurite_acc': 0, 'seen_acc': 0}
    for uid in uids:
        idxs = bids == uid
        accs = get_accuracy(ys[idxs], preds[idxs], ret_perfect=True)
        batch_acc['neurite_acc'] += accs['perfect']
        batch_acc['seen_acc'] += (accs['total_acc']*len(ys[idxs]))

    batch_acc['neurite_acc'] /= len(uids)
    batch_acc['seen_acc'] /= len(test_dataset)

    return batch_acc


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
    test_dataset = SimpleDataset(x, y, shuffle=False)

    config = get_config(config)
    model, _, _ = build_full_model_from_config(config.model, config.dataset)
    model = model.cuda()
    model = nn.DataParallel(model)
    model = load_model(model, checkpoint_path)

    metrics = test_model(model, test_dataset, use_gpu=True)

    print(metrics)


if __name__ == '__main__':
    test()
