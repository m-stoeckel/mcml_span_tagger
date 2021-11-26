from functools import partial
from typing import List

import torch
from torch import nn as nn

from model.linear import PoolingSpanClassificationModel, unwrap_batch


class PoolingSpanClassificationModelSep(PoolingSpanClassificationModel):
    def __init__(
            self,
            *args,
            gradient_clip_algorithm='norm',
            gradient_clip_val=2.0,
            **kwargs
    ):
        assert not kwargs.get('single_classifier', False), "Separate optimization only works with multiple classifiers"
        super(PoolingSpanClassificationModelSep, self).__init__(*args, **kwargs)
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_val = gradient_clip_val
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx, optimizer_idx):
        input_ids, attention_mask, labels, context_mask = unwrap_batch(batch)

        logits, remedy_logits = self(input_ids, attention_mask, context_mask)

        if self.remedy_solution and len(labels) >= self.max_span_length:
            labels, remedy_labels = labels[:-1], labels[-1]
        else:
            remedy_labels = None

        return self.compute_loss_separate(logits, remedy_logits, labels, remedy_labels)

    def compute_loss_separate(self, logits, remedy_logits, labels, remedy_labels):
        for opt in self.optimizers(use_pl_optimizer=True):
            opt.zero_grad()

        bce_loss = nn.BCEWithLogitsLoss(reduction=self.loss_reduction)
        layer_losses = []
        for layer, (layer_logits, layer_labels) in enumerate(zip(logits, labels)):
            layer_loss = bce_loss(layer_logits, layer_labels)

            self.manual_backward(layer_loss, retain_graph=True)

            layer_losses.append(layer_loss)

        if self.remedy_solution and remedy_labels is not None and remedy_logits is not None:
            remedy_loss = bce_loss(remedy_logits, remedy_labels)

            self.manual_backward(remedy_loss, retain_graph=True)

            layer_losses.append(remedy_loss)

        total_loss = torch.sum(torch.stack(layer_losses))
        self.log("loss", float(total_loss), prog_bar=True, logger=True)

        # Clip gradients
        if self.gradient_clip_algorithm == 'norm':
            for classifier in self.classifiers:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), self.gradient_clip_val)
            if self.reproject_lm:
                torch.nn.utils.clip_grad_norm_(self.reproject.parameters(), self.gradient_clip_val)
        elif self.gradient_clip_algorithm == 'value':
            for classifier in self.classifiers:
                torch.nn.utils.clip_grad_value_(classifier.parameters(), self.gradient_clip_val)
            if self.reproject_lm:
                torch.nn.utils.clip_grad_value_(self.reproject.parameters(), self.gradient_clip_val)

        for opt in self.optimizers(use_pl_optimizer=True):
            opt.step()

        schedulers: List[torch.optim.lr_scheduler.ReduceLROnPlateau] = self.lr_schedulers()
        if self.reproject_lm:
            schedulers, reproject_scheduler = schedulers[:-1], schedulers[-1]
            reproject_scheduler.step(total_loss)

        for sch, loss in zip(schedulers, layer_losses):
            sch.step(loss)

        return total_loss

    def configure_optimizers(self):
        if self.optimizer.lower() == 'adam':
            opt_cls = partial(torch.optim.Adam, lr=self.lr)
        elif self.optimizer.lower() == 'adamw':
            opt_cls = partial(torch.optim.AdamW, lr=self.lr)
        else:
            opt_cls = partial(torch.optim.SGD, lr=self.lr, momentum=self.momentum)

        opts = [
            opt_cls(self.classifiers[i].parameters())
            for i in range(self.max_span_length)
        ]
        lr_scheduler_list = [
            {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opts[i], patience=self.patience, factor=0.5),
                'monitor': f"val/loss",
                'name': f"{opt_cls.__class__.__name__}-layer_{i:02d}",
            }
            for i in range(self.max_span_length)
        ]

        if self.reproject_lm:
            remedy_opt = opt_cls(self.reproject.parameters())
            opts.append(remedy_opt)
            lr_scheduler_list.append({
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(remedy_opt, patience=self.patience, factor=0.5),
                'name': f"{opt_cls.__class__.__name__}-reprojection",
                'monitor': f"val/loss",
            })

        return opts, lr_scheduler_list


class PoolingSpanClassificationModelSepAuto(PoolingSpanClassificationModel):
    def __init__(
            self,
            *args,
            gradient_clip_algorithm='norm',
            gradient_clip_val=2.0,
            **kwargs
    ):
        assert not kwargs.get('single_classifier', False), "Separate optimization only works with multiple classifiers"
        super(PoolingSpanClassificationModelSepAuto, self).__init__(*args, **kwargs)
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_val = gradient_clip_val

    def training_step(self, batch, batch_idx, optimizer_idx):
        input_ids, attention_mask, labels, context_mask = unwrap_batch(batch)

        logits, remedy_logits = self(input_ids, attention_mask, context_mask)

        if self.remedy_solution and len(labels) >= self.max_span_length:
            labels, remedy_labels = labels[:-1], labels[-1]
        else:
            remedy_labels = None

        return self.compute_loss(logits, remedy_logits, labels, remedy_labels)

    def configure_optimizers(self):
        if self.optimizer.lower() == 'adam':
            opt_cls = partial(torch.optim.Adam, lr=self.lr)
        elif self.optimizer.lower() == 'adamw':
            opt_cls = partial(torch.optim.AdamW, lr=self.lr)
        else:
            opt_cls = partial(torch.optim.SGD, lr=self.lr, momentum=self.momentum)

        opts = [
            opt_cls(self.classifiers[i].parameters())
            for i in range(self.max_span_length)
        ]
        lr_scheduler_list = [
            {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opts[i], patience=self.patience, factor=0.5),
                'monitor': f"val/loss-layer_{i:02d}",
                'name': f"{opt_cls.__class__.__name__}-layer_{i:02d}",
                'strict': False
            }
            for i in range(self.max_span_length)
        ]

        if self.reproject_lm:
            remedy_opt = opt_cls(self.reproject.parameters())
            opts.append(remedy_opt)
            lr_scheduler_list.append({
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(remedy_opt, patience=self.patience, factor=0.5),
                'name': f"{opt_cls.__class__.__name__}-reprojection",
                'monitor': f"val/loss",
            })

        return opts, lr_scheduler_list