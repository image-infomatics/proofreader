
from proofreader.utils.torch import load_model, SimpleDataset
from proofreader.model.config import *
from proofreader.model.classifier import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import click
from tqdm import tqdm
from proofreader.model.config import *
import numpy as np
from proofreader.utils.vis import *


def single_canidate_prediction(y_hats, ys, bids):
    pass
    # uids = np.unique(bids)
    # batch_acc = {'neurite_acc': 0, 'seen_acc': 0}
    # wrong = []
    # for uid in uids:
    #     idxs = bids == uid
    #     total_canidates = len(ys[idxs])
    #     accs = get_accuracy(ys[idxs], preds[idxs], ret_perfect=True)
    #     batch_acc['neurite_acc'] += accs['perfect']
    #     batch_acc['seen_acc'] += (accs['total_acc']*total_canidates)

    # batch_acc['neurite_acc'] /= len(uids)
    # batch_acc['seen_acc'] /= len(test_dataset)


def test_model(model, test_dataset, batch_size=256, use_gpu=True, rank=0, max_canidates=None):

    dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                            pin_memory=use_gpu, drop_last=False, shuffle=True)

    ys, xs, y_hats, preds, bids = [], [], [], [], []
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
            y_hats.append(y_hat)
            ys.append(y)
            xs.append(x)
            preds.append(pred)
            bids.append(bid)

    model.train()

    ys, xs, y_hats, preds, bids = torch.cat(ys), torch.cat(xs), torch.cat(
        y_hats), torch.cat(preds), torch.cat(bids)

    uids = np.unique(bids)
    batch_acc = {'neurite_acc': 0, 'seen_acc': 0}
    wrong = []
    for uid in uids:
        idxs = bids == uid
        total_canidates = len(ys[idxs])
        accs = get_accuracy(ys[idxs], preds[idxs], ret_perfect=True)
        batch_acc['neurite_acc'] += accs['perfect']
        batch_acc['seen_acc'] += (accs['total_acc']*total_canidates)

    batch_acc['neurite_acc'] /= len(uids)
    batch_acc['seen_acc'] /= len(test_dataset)

    return batch_acc


def build_test_dataset(X, Y):
    # reset batch id
    for i in range(len(X)):
        y = Y[i]
        y[:, 1] = i

    X, Y = torch.cat(X), torch.cat(Y)
    test_dataset = SimpleDataset(X, Y, shuffle=True)

    return test_dataset


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
    x, y = torch.cat(y), torch.cat(x)
    test_dataset = SimpleDataset(x, y, shuffle=True)

    config = get_config(config)
    model, _, _, _ = build_full_model_from_config(config.model, config.dataset)
    model = model.cuda()
    model = nn.DataParallel(model)
    model = load_model(model, checkpoint_path)

    metrics = test_model(model, test_dataset, use_gpu=True)

    print(metrics)


if __name__ == '__main__':
    test()
