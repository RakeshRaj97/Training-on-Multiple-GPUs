# engine.py
import torch
import torch.nn as nn
from tqdm import tqdm
from average_meter import AverageMeter

try:
    from apex import amp

    _apex_available = True
except ImportError:
    _apex_available = False


class Engine:
    @staticmethod
    def train(
            data_loader,
            model,
            optimizer,
            scheduler=None,
            accumulation_steps=1,
            fp16=False
    ):
        if fp16 and not _apex_available:
            raise Exception("You want to use fp16 but you dont have apex installed")
        if fp16:
            accumulation_steps = 1
        losses = AverageMeter()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.cuda(non_blocking=True)
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()
            _, loss = model(**data)
            with torch.set_grad_enabled(True):
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    if (b_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        if b_idx > 0:
                            optimizer.zero_grad()
            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)
        return losses.avg

    @staticmethod
    def evaluate(
            data_loader,
            model,
            device
    ):
        losses = AverageMeter()
        final_predictions = []
        model.eval()
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                     data[key] = value.to(device)
                predictions, loss = model(**data)
                predictions = predictions.cpu()
                losses.update(loss.item(), data_loader.batch_size)
                final_predictions.append(predictions)
                tk0.set_postfix(loss=losses.avg)
        return final_predictions, losses.avg

