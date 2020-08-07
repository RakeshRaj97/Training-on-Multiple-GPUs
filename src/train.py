# train.py
import os
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import albumentations
from sklearn import metrics
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import pretrainedmodels
from dataset import ClassificationDataset
from engine import Engine
from model import SEResNext50_32x4d, EfficientNet
from wtfml.utils import EarlyStopping
from torch.nn import functional as F

# uncomment if using TPU
# import torch_xla
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.distributed.xla_multiprocessing as xmp
# export XLA_USE_BF16=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '192.168.44.171'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained, gpu=None):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)
        self.gpu = gpu

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            out, targets.reshape(-1, 1).type_as(out)
        ).cuda(self.gpu)
        return out, loss

def train(gpu, args):
    fold = 0
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    training_data_path = "/fred/oz138/test/input/train"
    model_path = "/home/rgopala/Training-on-Multiple-GPUs/model1"
    df = pd.read_csv("/fred/oz138/test/input/train_folds.csv")
    # device = 'cuda'
    # device = xm.xla_device()
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

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )

    valid_dataset = ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )

    # valid_sampler = torch.utils.data.distributed.DistributedSampler(
    #     valid_dataset,
    #     num_replicas=xm.xrt_world_size(),
    #     rank=xm.get_ordinal(),
    #     shuffle=False
    # )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )

    model = SEResNext50_32x4d(pretrained="imagenet", gpu=gpu)
    #model = nn.DataParallel(model)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    # model = EfficientNet()
    # model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )

    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level='O1',
        #verbosity=0
    )
    model = DDP(model)

    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(epochs):
        training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            #device,
            #fp16=True
        )
        predictions, valid_loss = Engine.evaluate(
            valid_loader,
            model,
            # device=device,
        )

        predictions = np.vstack(predictions).ravel()
        # print(f"predictions:{predictions.shape}")
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoch={epoch}, auc={auc}")
        es(auc, model, os.path.join(model_path, f"model{fold}.bin"))
        if es.early_stop:
            print("early stopping")
            break


if __name__ == "__main__":
    # train(0)
    # train(1)
    # train(2)
    # train(3)
    # train(4)
    # train(5)
    # train(6)
    # train(7)
    # train(8)
    # train(9)

    main()
