from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from transformers import PreTrainedTokenizerFast

from utils.data.corpus_readers import document_embedding_mrg_reader, rasa_reader, \
    simple_document_mrg_reader as document_mrg_reader, \
    simple_mrg_reader as mrg_reader
from utils.data.dataset import SpanDataset
from utils.data.subword_pooling_dataset import DocumentFeaturesDataset, SubwordPoolingDataset, TokenWindowSpanDataset


class MultiLabelSpanDataModule(pl.LightningDataModule):
    """TODO: Documentation"""

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            data_dir: str = None,
            train_file: str = None,
            test_file: str = None,
            val_file: Optional[str] = None,
            max_span_length=8,
            batch_size: int = 64,
            shuffle_train: bool = False,
            use_cache: bool = True,
            num_workers: int = 0,
            add_super_classes=False,
            offset_entity_end=True,
            subword_pooling=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.pad_token = self.tokenizer.pad_token if self.tokenizer.pad_token is not None else '[PAD]'
        self.pad_token_id = self.tokenizer.pad_token_id

        self.data_dir = Path(data_dir).absolute()
        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file

        self.max_span_length = max_span_length
        self.batch_size = batch_size
        self.shuffle = shuffle_train
        self.use_cache = use_cache
        self.num_workers = num_workers

        self.add_super_classes = add_super_classes
        self.offset_entity_end = offset_entity_end

        self.dataset_cls = SubwordPoolingDataset if subword_pooling else SpanDataset

    def prepare_data(self):
        self.train_dataset = self.separate_and_tokenize(list(self.read(self.data_dir / self.train_file)))

        self.test_dataset = self.separate_and_tokenize(list(self.read(self.data_dir / self.test_file)))

        entity_lexicon = {
            entity['entity_type']
            for dataset in (self.train_dataset['entities'], self.test_dataset['entities'])
            for sample in dataset
            for entity in sample
        }

        if self.val_file is not None:
            self.val_dataset = self.separate_and_tokenize(list(self.read(self.data_dir / self.val_file)))

            entity_lexicon |= {
                entity['entity_type']
                for sample in self.val_dataset['entities']
                for entity in sample
            }

        self.entity_lexicon = list(sorted(entity_lexicon))

    def read(self, path: Path):
        if path.suffix == '.mrg':
            return mrg_reader(path, offset_end=self.offset_entity_end)
        elif path.suffix == '.rasa':
            return rasa_reader(path)
        else:
            raise ValueError(f"Unknown extension '{path.suffix}' for corpus file: '{path.absolute()}'")

    def separate_and_tokenize(self, data_raw):
        sequences = []
        entities = []

        for sample in data_raw:
            sequences.append(sample['tokens'])
            entities.append(sample['entities'])

        batch_encoding = self.tokenizer(
            sequences,
            truncation=True,
            is_split_into_words=True,
            return_token_type_ids=False
        )

        return {
            'sequences': sequences,
            'entities': entities,
            'batch_encoding': batch_encoding
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = self.dataset_cls(
                self.entity_lexicon,
                max_span_length=self.max_span_length,
                pad_token=self.pad_token,
                add_super_classes=self.add_super_classes,
                **self.train_dataset
            )

            if self.val_file is not None:
                self.val_data = self.dataset_cls(
                    self.entity_lexicon,
                    max_span_length=self.max_span_length,
                    pad_token=self.pad_token,
                    add_super_classes=self.add_super_classes,
                    **self.val_dataset
                )
            else:
                lengths = [int(np.ceil(0.9 * len(self.train_data))), int(np.floor(0.1 * len(self.train_data)))]
                self.train_data, self.val_data = random_split(self.train_data, lengths)

        if stage == 'test' or stage is None:
            self.test_data = self.dataset_cls(
                self.entity_lexicon,
                max_span_length=max(99, self.max_span_length),
                pad_token=self.pad_token,
                add_super_classes=self.add_super_classes,
                **self.test_dataset
            )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        collate_fn = partial(
            self.collate,
            num_classes=len(self.entity_lexicon),
            max_span_length=self.train_data.max_span_length,
            pad_token_id=self.pad_token_id
        )
        return DataLoader(
            self.train_data, self.batch_size, *args,
            shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=collate_fn, **kwargs
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        collate_fn = partial(
            self.collate,
            num_classes=len(self.entity_lexicon),
            max_span_length=self.val_data.max_span_length,
            pad_token_id=self.pad_token_id
        )
        return DataLoader(
            self.val_data, self.batch_size, *args, num_workers=self.num_workers, collate_fn=collate_fn, **kwargs
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        collate_fn = partial(
            self.collate,
            num_classes=len(self.entity_lexicon),
            max_span_length=self.test_data.max_span_length,
            pad_token_id=self.pad_token_id
        )
        return DataLoader(
            self.test_data, self.batch_size, *args, num_workers=self.num_workers, collate_fn=collate_fn, **kwargs
        )

    def collate(self, batch, **kwargs):
        return collate(batch, **kwargs)


def collate(batch, num_classes=None, max_span_length=None, pad_token_id=0):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch = {key: [d[key] for d in batch] for key in batch[0].keys()}

    # Pad input ids with tokenizer specific padding value
    batch['input_ids'] = pad_and_stack_tensors(batch['input_ids'], pad_token_id)

    # Pad attention masks with zeros
    batch['attention_mask'] = pad_and_stack_tensors(batch['attention_mask'], 0)

    # Pad context masks with zeros
    batch['context_mask'] = pad_and_stack_tensors(batch['context_mask'], 0)

    # Pad word indices with zeros
    # batch['word_index'] = pad_and_stack_tensors(batch['word_index'], 0)

    # Pad labels with ignore value and transpose to class-first, batch-second order
    batch_labels = []
    for layer in range(max_span_length):
        batch_labels.append([])
        for labels in batch['labels']:
            if len(labels) > layer:
                batch_labels[layer].append(labels[layer].transpose(1, 0))
            else:
                batch_labels[layer].append(torch.empty((num_classes, 0)))

    batch['labels'] = [
        pad_and_stack_tensors(labels, 0).transpose(2, 1) for labels in batch_labels
    ]

    return batch


def pad_and_stack_tensors(batch: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    """
    Pad the tensors in the batch to equal length and stack them on the first dimension.

    :param batch: The batch of tensors to be padded.
    :param padding_value: The padding value to be used.
    :return: The a stacked & padded tensor.
    """
    elem = batch[0]

    max_length = max(el.shape[-1] for el in batch)
    batch = [
        F.pad(tensor, (0, max_length - tensor.shape[-1]), 'constant', padding_value)
        for tensor in batch
    ]

    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    return torch.stack(batch, 0, out=out)  # .squeeze()


class TokenWindowSpanDataModule(MultiLabelSpanDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, *args, window_size=64, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.dataset_cls = partial(TokenWindowSpanDataset, window_size=window_size)

    def separate_and_tokenize(self, data_raw: List[List[Dict]]):
        sequences = []
        entities = []

        sequence_to_document = {}
        document_sequences = defaultdict(list)

        for document_index, document in enumerate(data_raw):
            for sample in document:
                sequence_index = len(sequences)
                sequence_to_document[sequence_index] = document_index
                document_sequences[document_index].append(sequence_index)

                sequences.append(sample['tokens'])
                entities.append(sample['entities'])

        batch_encoding = self.tokenizer(
            sequences,
            truncation=True,
            is_split_into_words=True,
            return_token_type_ids=False
        )

        return {
            'sequences': sequences,
            'entities': entities,
            'batch_encoding': batch_encoding,
            'sequence_to_document': sequence_to_document,
            'document_sequences': document_sequences
        }

    def read(self, path):
        return document_mrg_reader(path)


class DocumentFeaturesSpanDataModule(MultiLabelSpanDataModule):

    def __init__(
            self,
            *args,
            feature_selector='tfidf.8.stack',
            **kwargs
    ):
        self.feature_selector = feature_selector
        super(DocumentFeaturesSpanDataModule, self).__init__(*args, **kwargs)
        self.dataset_cls = DocumentFeaturesDataset

    def read(self, path):
        return document_embedding_mrg_reader(path)

    def separate_and_tokenize(self, data_raw):
        sequences = []
        entities = []
        document_features = []

        for sample in data_raw:
            sequences.append(sample['tokens'])
            entities.append(sample['entities'])
            document_features.append(sample['document_features'][self.feature_selector])

        batch_encoding = self.tokenizer(
            sequences,
            truncation=True,
            is_split_into_words=True,
            return_token_type_ids=False
        )

        return {
            'sequences': sequences,
            'entities': entities,
            'batch_encoding': batch_encoding,
            'document_features': document_features
        }

    def collate(self, batch, **kwargs):
        batch = super(DocumentFeaturesSpanDataModule, self).collate(batch, **kwargs)
        batch['document_features'] = pad_and_stack_tensors(batch['document_features'], 0)
        return batch
