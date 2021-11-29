from typing import Any, Dict, Union
from typing import List

import torch
from tokenizers import Encoding
from transformers import BatchEncoding

from utils.data.dataset import SUBSTITUTION_RULES, SpanDataset


class SubwordPoolingDataset(SpanDataset):

    def __getitem__(self, i) -> Dict[str, Union[torch.Tensor, Any]]:
        encoding: Encoding = self.batch_encoding[i]
        input_ids = torch.tensor(encoding.ids, requires_grad=False)
        attention_mask = torch.tensor(encoding.attention_mask, requires_grad=False)
        context_mask = attention_mask.clone()
        context_mask[0] = 0
        context_mask[-1] = 0

        label_layers, tokenized_seq_len, word_index = self.subword_pooling(i, encoding, context_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_layers,
            'context_mask': context_mask,
            'seq_length': tokenized_seq_len,
            'pre_padding': 1,
            'word_index': word_index,
            'sequences': None
        }

    def subword_pooling(self, i, encoding, context_mask):
        # Number of words is last word_id + 1
        num_words = max(i for i in encoding.word_ids if i is not None) + 1

        # Using FIRST subword pooling
        # Get the first token index for all words
        token_to_word = torch.tensor([encoding.word_to_tokens(word_id)[0] for word_id in range(num_words)])

        # Build index array, to select only the first token of a word
        tokenized_seq_len = int(context_mask.sum())
        word_index = torch.zeros(tokenized_seq_len, dtype=torch.bool)

        # Subtract one for CLS token
        word_index[token_to_word - 1] = True
        label_layers = self.get_label_layers(i, num_words)

        return label_layers, tokenized_seq_len, word_index

    def get_label_layers(self, idx, tokenized_seq_len, token_offset=0):
        label_layers = []
        num_classes = len(self.class_list)
        for l in range(min(tokenized_seq_len, self.max_span_length)):
            n_spans_in_layer = tokenized_seq_len - l
            labels = torch.zeros((n_spans_in_layer, num_classes), requires_grad=False)
            label_layers.append(labels)

        for entity in self.entities[idx]:
            start, end = entity['span']
            entity_type = entity['entity_type']

            layer_index = abs(end - start) - 1

            self.add_entity(label_layers, entity_type, layer_index, start)

            if self.add_super_classes and (entity_type := SUBSTITUTION_RULES.get(entity_type.upper())) is not None:
                self.add_entity(label_layers, entity_type, layer_index, start)

        return label_layers


class TokenWindowSpanDataset(SubwordPoolingDataset):
    def __init__(
            self,
            class_list: List[str],
            sequences: List[List[str]] = None,
            entities: List[List[Dict[str, Any]]] = None,
            batch_encoding: BatchEncoding = None,
            sequence_to_document: Dict[int, int] = None,
            document_sequences: Dict[int, List[int]] = None,
            window_size=64,
            pad_token='PAD',
            max_span_length=8,
            **kwargs
    ):
        self.sequence_to_document = sequence_to_document
        self.document_sequences = document_sequences
        self.window_size = window_size
        super(TokenWindowSpanDataset, self).__init__(
            class_list,
            sequences,
            entities,
            batch_encoding,
            pad_token,
            max_span_length,
            **kwargs
        )

    def __getitem__(self, i) -> Dict[str, Union[torch.Tensor, Any]]:
        encoding: Encoding = self.batch_encoding[i]
        input_ids = encoding.ids

        document_index = self.sequence_to_document.get(i)
        in_document_index = self.document_sequences[document_index].index(i)

        pre = []
        if in_document_index > 0:
            # pre = self.batch_encoding[in_document_index - 1]
            preceding_in_document_index = self.document_sequences[document_index][in_document_index - 1]
            pre = self.batch_encoding[preceding_in_document_index].ids[1:-1][-self.window_size:]

        post = []
        if in_document_index < len(self.document_sequences[document_index]) - 1:
            # post = self.batch_encoding[in_document_index + 1]
            succeeding_in_document_index = self.document_sequences[document_index][in_document_index + 1]
            post = self.batch_encoding[succeeding_in_document_index].ids[1:-1][:self.window_size]

        # The new context mask must exclude CLS and SEP tokens, as well as pre and post context tokens
        raw_length = len(input_ids) - 2
        len_pre = len(pre) or 0
        len_post = len(post) or 0
        context_mask = torch.tensor([0] + [0] * len_pre + [1] * raw_length + [0] * len_post + [0])

        input_ids = torch.tensor([input_ids[0]] + pre + input_ids[1:-1] + post + [input_ids[-1]])
        attention_mask = torch.ones_like(input_ids)

        label_layers, tokenized_seq_len, word_index = self.subword_pooling(i, encoding, context_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_layers,
            'context_mask': context_mask,
            'seq_length': tokenized_seq_len,
            'pre_padding': 1 + len_pre,
            'word_index': word_index,
            'sequences': None
        }


class DocumentFeaturesDataset(SubwordPoolingDataset):
    def __init__(
            self,
            *args,
            document_features,
            **kwargs
    ):
        self.document_features = document_features
        super(DocumentFeaturesDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        sample = super(DocumentFeaturesDataset, self).__getitem__(idx)
        sample['document_features'] = self.document_features[idx]
        return sample


class DocumentFeaturesTokenWindowDataset(TokenWindowSpanDataset):
    def __init__(
            self,
            *args,
            document_features,
            **kwargs
    ):
        self.document_features = document_features
        super(DocumentFeaturesTokenWindowDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        sample = super(DocumentFeaturesTokenWindowDataset, self).__getitem__(idx)
        sample['document_features'] = self.document_features[idx]
        return sample
