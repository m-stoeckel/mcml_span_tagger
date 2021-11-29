# encoding: utf-8
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import comet_ml
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import test_tube
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import CometLogger, TestTubeLogger, WandbLogger
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModel, PreTrainedModel

from utils.label_decoder import LabelDecoder
from utils.metrics import multi_label_span_classification_report

nne_entities = [
    'ADDRESSNON', 'AGE', 'AIRPORT', 'ALBUM', 'ANIMATE', 'ARMY', 'ATTRACTION', 'AWARD', 'BAND', 'BOOK', 'BORDER',
    'BRIDGE', 'BUILDING', 'CARDINAL', 'CHANNEL', 'CHEMICAL', 'CITY', 'CITYSTATE', 'CONCERT', 'CONTINENT',
    'CORPJARGON', 'COUNTRY', 'DATE', 'DATEOTHER', 'DAY', 'DISEASE', 'DURATION', 'ELECTRONICS', 'ENERGY',
    'EVENT', 'FACILITY', 'FILM', 'FIRST', 'FOLD', 'FUND', 'GOD', 'GOVERNMENT', 'GPE', 'GRPLOC', 'GRPORG',
    'GRPPER', 'HON', 'HOSPITAL', 'HOTEL', 'HURRICANE', 'INDEX', 'INI', 'IPOINTS', 'LANGUAGE', 'LAW',
    'LOCATIONOTHER', 'MEDIA', 'MIDDLE', 'MONEY', 'MONTH', 'MULT', 'MUSEUM', 'NAME', 'NAMEMOD', 'NATIONALITY',
    'NATURALDISASTER', 'NICKNAME', 'NORPOTHER', 'NORPPOLITICAL', 'NUMDAY', 'OCEAN', 'ORDINAL', 'ORGCORP',
    'ORGEDU', 'ORGOTHER', 'ORGPOLITICAL', 'ORGRELIGIOUS', 'PAINTING', 'PER', 'PERCENT', 'PERIODIC', 'PLAY',
    'PRODUCTDRUG', 'PRODUCTFOOD', 'PRODUCTOTHER', 'QUAL', 'QUANTITY1D', 'QUANTITY2D', 'QUANTITY3D',
    'QUANTITYOTHER', 'RATE', 'REGION', 'REL', 'RELIGION', 'RIVER', 'ROLE', 'SCINAME', 'SEASON', 'SONG', 'SPACE',
    'SPEED', 'SPORTSEVENT', 'SPORTSSEASON', 'SPORTSTEAM', 'STADIUM', 'STATE', 'STATION', 'STREET', 'SUBURB',
    'TEMPERATURE', 'TIME', 'TVSHOW', 'UNIT', 'VEHICLE', 'WAR', 'WEAPON', 'WEIGHT', 'WOA', 'YEAR'
]

genia_entities = [
    'G#DNA',
    'G#RNA',
    'G#cell_line',
    'G#cell_type',
    'G#protein'
]


def to_numpy(tensor: torch.Tensor):
    return tensor.cpu().detach().numpy()


def unwrap_batch(
        batch: Dict[str, Union[torch.Tensor, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int], List[int]]:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    context_mask = batch['context_mask']
    seq_length = batch['seq_length']
    pre_padding = batch['pre_padding']
    word_index = batch['word_index']

    return input_ids, attention_mask, labels, context_mask, seq_length, pre_padding, word_index


class PoolingSpanClassificationModel(pl.LightningModule):
    def __init__(
            self,
            entities_lexicon=None,
            language_model: str = 'xlm-roberta-large',
            max_span_length=8,
            dropout: float = 0.1,
            reproject_lm='linear',
            reproject_lm_dim: int = 256,
            lm_layers: str = 'all',
            lm_layer_aggregation: str = 'mean',
            lm_exclude_embedding=True,
            lm_finetune=False,
            lr: float = 1e-3,
            momentum: float = 0.9,
            patience: int = 2,
            use_cache=False,
            optimizer='adamw',
            feature_pooling='max',
            single_classifier=True,
            loss_reduction='mean',
            subword_pooling=True
    ):
        super().__init__()
        self.entities_lexicon = entities_lexicon
        if entities_lexicon is None:
            self.entities_lexicon = nne_entities
        self.max_span_length = max_span_length
        self.dropout = dropout
        self.lr = lr
        self.momentum = momentum
        self.patience = patience
        self.optimizer = optimizer
        self.feature_pooling = feature_pooling.lower()
        self.single_classifier = single_classifier
        self.reproject_lm = reproject_lm if reproject_lm != 'none' else None
        self.reproject_lm_dim = reproject_lm_dim
        self.lm_finetune = lm_finetune
        self.loss_reduction = loss_reduction
        self.subword_pooling = subword_pooling

        self.lm_exclude_embedding = lm_exclude_embedding
        if lm_layers == 'all':
            self.layer_from = 0
            self.layer_to = None
            self.lm_exclude_embedding = False
        elif ':' in lm_layers:
            self.layer_from, self.layer_to = (int(idx) if idx else None for idx in lm_layers.split(':'))
        else:
            self.layer_from = self.layer_to = int(lm_layers)
        self.lm_layer_aggregation = lm_layer_aggregation

        self._init_lm(language_model)

        self.decoder = LabelDecoder(self.entities_lexicon)

        self.use_cache = use_cache
        if self.use_cache:
            print("Caching only works if the training/validation data is not shuffled! Use with caution.")
        self.cache: Dict[str, torch.Tensor] = {}

        if self.use_cache and self.lm_finetune:
            print("`use_cache` cannot be True if `lm_finetune = True`! Disabling cache.")
            self.use_cache = False

        if self.reproject_lm == 'linear':
            self.reproject = nn.Linear(self.lm_hidden_size, self.reproject_lm_dim)
        elif self.reproject_lm in ('rnn', 'lstm', 'gru'):
            raise NotImplementedError("LSTM reprojection not implemented yet!")

        if self.feature_pooling == 'cat' and self.single_classifier:
            raise ValueError("Cannot use single classifier with 'cat' feature aggregation!")

        self._init_classifier()

        self.save_hyperparameters()

    def _init_lm(self, language_model):
        model_config = AutoConfig.from_pretrained(language_model, output_hidden_states=True)
        self.language_model: PreTrainedModel = AutoModel.from_pretrained(
            language_model,
            config=model_config
        )
        self.language_model.requires_grad_(self.lm_finetune)

        self.lm_hidden_size = model_config.hidden_size
        if self.lm_layer_aggregation == 'cat' and self.layer_from != self.layer_to:
            emb_inc = int(not self.lm_exclude_embedding)
            if self.layer_from is None:
                num_layers = len(list(range(model_config.num_hidden_layers + emb_inc))[:self.layer_to])
            elif self.layer_to is None:
                num_layers = len(list(range(model_config.num_hidden_layers + emb_inc))[self.layer_from:])
            else:
                num_layers = len(list(range(model_config.num_hidden_layers + emb_inc))[self.layer_from:self.layer_to])
            self.lm_hidden_size = self.lm_hidden_size * num_layers

    def _init_classifier(self):
        if self.reproject_lm:
            classifier_input_dim = self.reproject_lm_dim
        else:
            classifier_input_dim = self.lm_hidden_size

        if self.feature_pooling == 'cat':
            self.classifiers = nn.ModuleList([
                nn.Linear(classifier_input_dim * i, self.number_of_classes)
                for i in range(1, self.max_span_length + 1)
            ])
        else:
            if self.feature_pooling == 'exhaustive':
                classifier_input_dim *= 3
            self._init_samesize_classifier(classifier_input_dim)

    def _init_samesize_classifier(self, classifier_input_dim):
        if not self.single_classifier:
            self.classifiers = nn.ModuleList([
                nn.Linear(classifier_input_dim, self.number_of_classes)
                for _ in range(1, self.max_span_length + 1)
            ])
        else:
            self.classifier = nn.Linear(classifier_input_dim, self.number_of_classes)

    @property
    def number_of_classes(self):
        return len(self.entities_lexicon)

    def embed(self, input_ids, attention_mask) -> torch.Tensor:
        if not self.use_cache:
            if self.lm_finetune:
                return self._embed(input_ids, attention_mask)
            else:
                with torch.no_grad():
                    return self._embed(input_ids, attention_mask)
        else:
            return self._get_from_cache(input_ids, attention_mask)

    def _embed(self, input_ids, attention_mask) -> torch.Tensor:
        # If we include the embedding layer, the start of the slice has to be 0
        start = int(self.lm_exclude_embedding)
        embedding: Tuple[torch.Tensor] = self.language_model(input_ids, attention_mask).hidden_states[start:]
        if self.layer_from == self.layer_to:
            embedding: torch.Tensor = embedding[self.layer_from]
        elif self.layer_from is None:
            embedding: torch.Tensor = torch.stack(embedding[:self.layer_to], -1)
        elif self.layer_to is None:
            embedding: torch.Tensor = torch.stack(embedding[self.layer_from:], -1)
        else:
            embedding: torch.Tensor = torch.stack(embedding[self.layer_from:self.layer_to], -1)
        embedding: torch.Tensor = self.aggregate_features(embedding, self.lm_layer_aggregation)
        return embedding

    @torch.no_grad()
    def _get_from_cache(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        key = self._generate_key(input_ids)
        if key not in self.cache:
            embedding = self._embed(input_ids, attention_mask)
            self.cache[key] = embedding.clone().detach().cpu()
            return embedding
        return self.cache[key].to(self.device)

    @staticmethod
    def aggregate_features(tensors: torch.Tensor, aggregation_method: str):
        """

        Args:
            tensors: 4D tensor of shape (batch_size, sequence_length, feature_dim, layers).
            aggregation_method: Either a pooling method `min`, `max` or `mean`; or

        Returns:
            A 3D tensor of shape (batch_size, sequence_length, feature_dim) if
        """
        batch_size, sequence_length = tensors.shape[:2]
        if aggregation_method == 'min':
            output, _ = tensors.min(-1)
            return output
        elif aggregation_method == 'max':
            output, _ = tensors.max(-1)
            return output
        elif aggregation_method == 'mean':
            return tensors.mean(-1)
        elif aggregation_method == 'exhaustive':
            # Create representation for: start token, 'inside' span tokens, end token. See:
            # @InProceedings{Sohrab_2018_Deep,
            #   author    = {Sohrab, Mohammad Golam and Miwa, Makoto},
            #   title     = {Deep Exhaustive Model for Nested Named Entity Recognition},
            #   booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
            #   date      = {2018},
            #   doi       = {10.18653/v1/D18-1309}
            # }
            if len(tensors.size()) == 3:
                start_repr = tensors[:, :, 0]
                inside_repr = torch.mean(tensors, dim=-1)
                end_repr = tensors[:, :, -1]
            else:
                start_repr = tensors[:, :, :, 0]
                inside_repr = torch.mean(tensors, dim=-1)
                end_repr = tensors[:, :, :, -1]

            return torch.cat((start_repr, inside_repr, end_repr), dim=-1)
        elif aggregation_method == 'cat':
            # Instead of using torch.cat with a tuple of tensors, we can use reshape to achieve a tensor of the same
            # shape. It will have a different order for the elements, but this is irrelevant for a DNN feature tensor.
            return tensors.reshape(batch_size, sequence_length, -1)
        else:
            raise ValueError(f"'{aggregation_method}' is not a valid aggregation method!")

    def _generate_key(self, input_ids: torch.Tensor) -> str:
        return "".join(chr(i) for i in input_ids.flatten())

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, context_mask, seq_length, pre_padding, word_index = unwrap_batch(batch)

        logits, context_mask = self(input_ids, attention_mask, context_mask, seq_length, pre_padding, word_index)

        # BUG: Find the cause of this! -> I think I found it!
        # Happend with single item batches with sequences shorter than 2, because of a runaway seq_len - 2
        if not len(logits) or not len(labels):
            from pathlib import Path
            import hashlib

            key = self._generate_key(input_ids)
            digest = hashlib.sha1(key.encode()).hexdigest()

            save_dir = Path('/tmp/ma-stoeckel/broken_batch/')
            save_dir.mkdir(parents=True, exist_ok=True)

            batch_file = save_dir / f"batch_{digest}.pt"
            with open(batch_file, 'wb') as fp:
                torch.save(batch, fp)

            raise RuntimeError(f"Broken batch found! len(logits): {len(logits)} len(labels): {len(labels)}! "
                               f"Saved batch to {batch_file.absolute()}")

        return self.compute_loss(logits, labels)

    def training_step_end(self, loss):
        self.log("train/loss", loss, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(*unwrap_batch(batch))

    def validation_epoch_end(self, validation_step_outputs):
        self.log("val/loss", sum(float(step['loss']) for step in validation_step_outputs), prog_bar=False, logger=True)

        for i, layer_loss in enumerate(pd.DataFrame(step['loss_per_layer'] for step in validation_step_outputs).sum(0)):
            self.log(f"val/loss/layer_{i:02d}", float(layer_loss), prog_bar=False, logger=True)

        self.evaluate_epoch(validation_step_outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self.evaluate(*unwrap_batch(batch))

    def test_epoch_end(self, outputs):
        full_report, layer_reports = self.evaluate_epoch(outputs, 'test')

        experiment = self.logger.experiment
        pd_from_dict = pd.DataFrame.from_dict
        if isinstance(experiment, test_tube.Experiment):
            log_dir = Path(self.logger.save_dir) / self.logger.name / f"version_{self.logger.version}" / "tf"
            writer = SummaryWriter(str(log_dir))
            writer.add_text('full_report', pd_from_dict(full_report, orient='index').to_csv(sep='|'))
            for i, layer_report in enumerate(layer_reports, 1):
                writer.add_text(f"layer_{i}_report", pd_from_dict(layer_report, orient='index').to_csv(sep='|'))
            writer.close()
        elif isinstance(experiment, comet_ml.Experiment):
            experiment.log_html("<h2>Results</h1>")
            experiment.log_html("<h3>All Layers</h3>" + pd_from_dict(full_report, orient='index').to_html())
            for i, layer_report in enumerate(layer_reports, 1):
                experiment.log_html(f"<h3>Layer: {i}</h3>" + pd_from_dict(layer_report, orient='index').to_html())
        elif isinstance(experiment, wandb.wandb_sdk.wandb_run.Run):
            experiment.log({"test/full_report": wandb.Table(dataframe=pd_from_dict(full_report, orient='index'))})
            for i, layer_report in enumerate(layer_reports, 1):
                experiment.log({
                    f"test/layer_report/{i}": wandb.Table(dataframe=pd_from_dict(layer_report, orient='index')),
                    "global_step": self.global_step
                })
        else:
            result_dict = {"All Layers": full_report['micro avg']}
            for i, layer_report in enumerate(layer_reports, 1):
                result_dict[f"Layer {i}"] = layer_report['micro avg']
            print(pd_from_dict(result_dict, orient='index'))

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            context_mask: torch.Tensor,
            seq_length: List[int],
            pre_padding: List[int],
            word_index: List[int],
            **kwargs
    ):
        sequence_output = self.embed(input_ids, attention_mask)

        # sequence_output is now of shape: (batch_size, seq_len, lm_hidden_size)
        sequence_output = F.dropout(sequence_output, self.dropout, training=self.training)

        # If self.reproject_lm, shape changes to: (batch_size, seq_len, reproject_lm_dim)
        if self.reproject_lm:
            sequence_output = torch.relu(self.reproject(sequence_output))

        if self.subword_pooling:
            sequence_output = [
                sequence_embeddings[pre:pre + length][index]
                for sequence_embeddings, length, pre, index in zip(sequence_output, seq_length, pre_padding, word_index)
            ]

            context_mask = [
                mask[pre:pre + length][index]
                for mask, length, pre, index in zip(context_mask, seq_length, pre_padding, word_index)
            ]
        else:
            # Remove pre padding
            sequence_output = [
                sequence_embeddings[pre:pre + length]
                for sequence_embeddings, length, pre in zip(sequence_output, seq_length, pre_padding)
            ]

            context_mask = [
                mask[pre:pre + length]
                for mask, length, pre in zip(context_mask, seq_length, pre_padding)
            ]

        sequence_output = torch.nn.utils.rnn.pad_sequence(sequence_output, True, 0.0)
        context_mask = torch.nn.utils.rnn.pad_sequence(context_mask, True, 0)

        layer_outputs = self.forward_layers(sequence_output, context_mask)
        classifier_outputs = self.forward_classifier(layer_outputs, **kwargs)

        return classifier_outputs, context_mask

    def forward_layers(self, sequence_output, context_mask, **kwargs):
        batch_size, seq_len = sequence_output.size()[:2]

        layer_outputs = []
        for i in range(min(seq_len, self.max_span_length)):
            span_length = i + 1

            layer_output: torch.Tensor = sequence_output.unfold(1, span_length, 1)

            layer_output = self.aggregate_features(layer_output, self.feature_pooling)
            layer_output = torch.relu(layer_output)

            layer_outputs.append(layer_output)
        return layer_outputs

    def forward_classifier(self, layer_outputs: List[torch.Tensor], **kwargs):
        classifier_outputs = []
        for i, layer_output in enumerate(layer_outputs):
            classifier = self.classifier if self.single_classifier else self.classifiers[i]
            classifier_outputs.append(classifier.forward(layer_output))
        return classifier_outputs

    def compute_loss(self, logits, labels, return_loss_per_layer=False):
        bce_loss = nn.BCEWithLogitsLoss(reduction=self.loss_reduction)
        loss_per_layer = [
            bce_loss(layer_logits, layer_labels)
            for layer_logits, layer_labels in zip(logits, labels)
        ]

        if return_loss_per_layer:
            return torch.sum(torch.stack(loss_per_layer)), loss_per_layer
        return torch.sum(torch.stack(loss_per_layer))

    def evaluate(self, input_ids, attention_mask, labels, context_mask, seq_length, pre_padding, word_index, **kwargs):
        logits, context_mask = self(input_ids, attention_mask, context_mask, seq_length, pre_padding, word_index,
                                    **kwargs)
        loss, loss_per_layer = self.compute_loss(logits, labels, True)
        preds, labels = self.collect_preds(logits, labels)
        return {
            'loss': loss.detach(),
            'loss_per_layer': [layer_loss.detach() for layer_loss in loss_per_layer],
            'preds': preds,
            'labels': labels,
            'masks': to_numpy(context_mask)
        }

    def collect_preds(self, logits, labels):
        preds = [to_numpy(torch.round(torch.sigmoid(l))) for l in logits]
        labels = [to_numpy(l) for l in labels]
        return preds, labels

    def evaluate_epoch(self, epoch_step_outputs, split) -> Tuple[dict, List[dict]]:
        pred = []
        layer_wise_pred = [[] for _ in range(self.max_span_length)]
        true = []
        layer_wise_true = [[] for _ in range(self.max_span_length)]
        for step_outputs in epoch_step_outputs:
            _pred = self.decoder.decode(step_outputs['preds'], step_outputs['masks'])
            _true = self.decoder.decode(step_outputs['labels'], step_outputs['masks'])

            self.layer_wise_classes(_pred, layer_wise_pred)
            self.layer_wise_classes(_true, layer_wise_true)

            if len(_pred) > len(_true):
                self.extend_to_same_length(_true, _pred)
            elif len(_true) > len(_pred):
                self.extend_to_same_length(_pred, _true)

            pred.extend([seq for layer in _pred for seq in layer])
            true.extend([seq for layer in _true for seq in layer])

        lexicon = self.entities_lexicon
        full_report = multi_label_span_classification_report(pred, true, lexicon, True)

        full_report_micro_avg = full_report['micro avg']
        self.log(
            f"{split}/f1_micro",
            round(full_report_micro_avg['f1-score'] * 100, 2),
            prog_bar=True,
            logger=True,
        )

        self.log(
            f"{split}/precision_micro",
            round(full_report_micro_avg['precision'] * 100, 2),
            logger=True,
        )

        self.log(
            f"{split}/recall_micro",
            round(full_report_micro_avg['recall'] * 100, 2),
            logger=True,
        )

        layer_reports = []
        val_f1_micro_per_layer = {}
        val_precision_micro_per_layer = {}
        val_recall_micro_per_layer = {}
        val_support_per_layer = {}

        if isinstance(self.logger, TestTubeLogger):
            for layer in range(self.max_span_length):
                layer_report = multi_label_span_classification_report(
                    layer_wise_pred[layer],
                    layer_wise_true[layer],
                    lexicon,
                    True
                )
                micro_avg = layer_report['micro avg']
                val_f1_micro_per_layer[f"layer_{layer:02d}"] = round(micro_avg['f1-score'] * 100, 2)
                val_precision_micro_per_layer[f"layer_{layer:02d}"] = round(micro_avg['precision'] * 100, 2)
                val_recall_micro_per_layer[f"layer_{layer:02d}"] = round(micro_avg['recall'] * 100, 2)
                val_support_per_layer[f"layer_{layer:02d}"] = micro_avg['support']
                layer_reports.append(layer_report)

            self.log(
                f"{split}/f1_micro/",
                val_f1_micro_per_layer,
                logger=True,
            )
            self.log(
                f"{split}/precision_micro/",
                val_precision_micro_per_layer,
                logger=True,
            )
            self.log(
                f"{split}/recall_micro/",
                val_recall_micro_per_layer,
                logger=True,
            )
            self.log(f"{split}/support/", val_support_per_layer, logger=True)

        elif isinstance(self.logger, (CometLogger, WandbLogger)):
            for layer in range(self.max_span_length):
                layer_report = multi_label_span_classification_report(
                    layer_wise_pred[layer],
                    layer_wise_true[layer],
                    lexicon,
                    True
                )
                val_f1_micro_per_layer[f"{split}/f1_micro/layer_{layer:02d}"] = round(
                    layer_report['micro avg']['f1-score'] * 100, 2)
                val_precision_micro_per_layer[f"{split}/precision_micro/layer_{layer:02d}"] = round(
                    layer_report['micro avg']['precision'] * 100, 2)
                val_recall_micro_per_layer[f"{split}/recall_micro/layer_{layer:02d}"] = round(
                    layer_report['micro avg']['recall'] * 100, 2)
                val_support_per_layer[f"{split}/support/layer_{layer:02d}"] = int(layer_report['micro avg']['support'])
                layer_reports.append(layer_report)

            self.logger.log_metrics(
                val_f1_micro_per_layer,
            )
            self.logger.log_metrics(
                val_precision_micro_per_layer,
            )
            self.logger.log_metrics(
                val_recall_micro_per_layer,
            )
            self.logger.log_metrics(
                val_support_per_layer,
            )
        else:
            for layer in range(self.max_span_length):
                layer_reports.append(multi_label_span_classification_report(
                    layer_wise_pred[layer],
                    layer_wise_true[layer],
                    lexicon,
                    True
                ))

        return full_report, layer_reports

    def layer_wise_classes(self, y, layer_wise_):
        for i in range(min(self.max_span_length, len(y))):
            layer_wise_[i].extend([seq for seq in y[i]])

    def extend_to_same_length(self, shorter, longer):
        empty = np.full(len(self.entities_lexicon), False, dtype=bool)
        shorter.extend([
            [
                [empty for _ in y]
                for y in extra_layer
            ]
            for extra_layer in longer[len(shorter):]
        ])

    def configure_optimizers(self):
        if self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,
                                        momentum=self.momentum)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, factor=0.5),
                'monitor': 'val/loss',
            }
        }
