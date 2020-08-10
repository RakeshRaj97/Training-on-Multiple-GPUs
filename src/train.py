# train.py
import os
import pandas as pd
import numpy as np
import argparse
from sklearn import metrics

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from dataset import ClassificationDataset
from engine import Engine
from model import SEResNext50_32x4d

from wtfml.utils import EarlyStopping
import albumentations
import pretrainedmodels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '192.168.44.101'  # replace with your ip address
    os.environ['MASTER_PORT'] = '8880'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
    for i in range(10):
        fold = i
        rank = args.nr * args.gpus + gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
        torch.manual_seed(0)
        training_data_path = "/fred/oz138/test/input/train"
        model_path = "/home/rgopala/Training-on-Multiple-GPUs/model"
        df = pd.read_csv("/fred/oz138/test/input/train_folds.csv")
        epochs = 50
        train_bs = 32
        valid_bs = 16
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        train_aug = albumentations.Compose(
            [
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
                albumentations.Flip(p=0.5),
                albumentations.RandomRotate90(p=0.5)
            ]
        )

        valid_aug = albumentations.Compose(
            [
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            ]
        )

        train_images = df_train.image_name.values.tolist()
        train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
        train_targets = df_train.target.values

        valid_images = df_valid.image_name.values.tolist()
        valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
        valid_targets = df_valid.target.values

        model = SEResNext50_32x4d(pretrained="imagenet", gpu=gpu)
        torch.cuda.set_device(gpu)
        model.cuda(gpu)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            mode="max"
        )

        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level='O1'
        )
        model = DDP(model)

        train_dataset = ClassificationDataset(
            image_paths=train_images,
            targets=train_targets,
            resize=None,
            augmentations=train_aug
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=rank
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_bs,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler
        )

        valid_dataset = ClassificationDataset(
            image_paths=valid_images,
            targets=valid_targets,
            resize=None,
            augmentations=valid_aug
        )

        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=args.world_size,
            rank=rank
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=valid_bs,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=valid_sampler
        )

        es = EarlyStopping(patience=5, mode="max")
        for epoch in range(epochs):
            training_loss = Engine.train(
                train_loader,
                model,
                optimizer,
                fp16=True
            )
            predictions, valid_loss = Engine.evaluate(
                valid_loader,
                model
            )

            predictions = np.vstack(predictions).ravel()
            auc = metrics.roc_auc_score(valid_targets, predictions)
            scheduler.step(auc)
            print(f"epoch={epoch}, auc={auc}")
            es(auc, model, os.path.join(model_path, f"model{fold}.bin"))
            if es.early_stop:
                print("early stopping")
                break


if __name__ == "__main__":

    main()
