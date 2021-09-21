from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from proofreader.data.cremi import prepare_cremi_vols
from proofreader.model.config import *


def cli_main(args):
    pl.seed_everything(1234, workers=True)

    # config
    config_name = args.config
    config = get_config(config_name)

    # data
    train_vols, test_vols = prepare_cremi_vols('./dataset/cremi')
    train_loader, val_loader = build_dataloader_from_config(
        config.dataset, config.augmentor, train_vols)

    # model
    model = build_model_from_config(config.model, config.dataset)

    # training
    trainer = build_trainer_from_config(config.trainer, args)
    trainer.fit(model, train_loader, val_loader)

    # testing
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', default='default', type=str)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    cli_main(args)
