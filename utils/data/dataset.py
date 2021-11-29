import logging
from typing import Any, Dict, List, Union

import torch
from tokenizers import Encoding
from torch.utils.data.dataset import Dataset
from transformers import BatchEncoding

SUBSTITUTION_RULES = {
    'AGE': 'TIMEX',
    'AIRPORT': 'FACILITY',
    'ALBUM': 'WOA',
    'ANIMATE': 'MISC',
    'ARMY': 'ORGANISATIONS',
    'ATTRACTION': 'FACILITY',
    'AWARD': 'MISC',
    'BAND': 'ORGANISATIONS',
    'BOOK': 'WOA',
    'BORDER': 'LOCACTION',
    'BRIDGE': 'FACILITY',
    'BUILDING': 'FACILITY',
    'CARDINAL': 'NUMEX',
    'CHANNEL': 'ORGANISATIONS',
    'CITY': 'LOCACTION',
    'CITYSTATE': 'LOCACTION',
    'CONCERT': 'EVENT',
    'CONTINENT': 'LOCACTION',
    'CORPJARGON': 'ORGANISATIONS',
    'COUNTRY': 'LOCACTION',
    'DATE': 'TIMEX',
    'DATEOTHER': 'TIMEX',
    'DAY': 'TIMEX',
    'DISEASE': 'MISC',
    'DURATION': 'TIMEX',
    'ELECTRONICS': 'MISC',
    'ENERGY': 'NUMEX',
    'EVENT': 'EVENT',
    'FACILITY': 'FACILITY',
    'FILM': 'WOA',
    'FIRST': 'PERSON',
    'FOLD': 'NUMEX',
    'FUND': 'ORGANISATIONS',
    'GAME': 'MISC',
    'GOD': 'MISC',
    'GOVERNMENT': 'ORGANISATIONS',
    'GPE': 'LOCACTION',
    'GRPLOC': 'GROUP',
    'GRPORG': 'GROUP',
    'GRPPER': 'GROUP',
    'HON': 'PERSON',
    'HOSPITAL': 'ORGANISATIONS',
    'HOTEL': 'ORGANISATIONS',
    'HURRICANE': 'EVENT',
    'INDEX': 'ORGANISATIONS',
    'INI': 'PERSON',
    'INITIALS': 'PERSON',
    'IPOINTS': 'NUMEX',
    'JARGON': 'ORGANISATIONS',
    'LANGUAGE': 'MISC',
    'LAW': 'MISC',
    'LOCATIONOTHER': 'LOCACTION',
    'MEDIA': 'ORGANISATIONS',
    'MIDDLE': 'PERSON',
    'MONEY': 'NUMEX',
    'MONTH': 'TIMEX',
    'MULT': 'NUMEX',
    'MUSEUM': 'ORGANISATIONS',
    'NAME': 'PERSON',
    'NAMEMOD': 'PERSON',
    'NATIONALITY': 'NORP',
    'NATURALDISASTER': 'EVENT',
    'NICKNAME': 'PERSON',
    'NORPOTHER': 'NORP',
    'NORPPOLITICAL': 'NORP',
    'NUMDAY': 'TIMEX',
    'OCEAN': 'LOCACTION',
    'ORDINAL': 'NUMEX',
    'ORGCORP': 'ORGANISATIONS',
    'ORGEDU': 'ORGANISATIONS',
    'ORGOTHER': 'ORGANISATIONS',
    'ORGPOLITICAL': 'ORGANISATIONS',
    'ORGRELIGIOUS': 'ORGANISATIONS',
    'PAINTING': 'WOA',
    'PERCENT': 'NUMEX',
    'PERIODIC': 'TIMEX',
    'PER': 'PERSON',
    'PERSON': 'PERSON',
    'PLAY': 'WOA',
    'PRODUCTDRUG': 'MISC',
    'PRODUCTFOOD': 'MISC',
    'PRODUCTOTHER': 'MISC',
    'QUANTITY': 'NUMEX',
    'QUANTITY1D': 'NUMEX',
    'QUANTITY2D': 'NUMEX',
    'QUANTITY3D': 'NUMEX',
    'QUANTITYOTHER': 'NUMEX',
    'RATE': 'NUMEX',
    'REGION': 'LOCACTION',
    'REL': 'TIMEX',
    'RELIGION': 'NORP',
    'RIVER': 'LOCACTION',
    'ROLE': 'PERSON',
    'SCINAME': 'MISC',
    'SEASON': 'TIMEX',
    'SONG': 'WOA',
    'SPACE': 'LOCACTION',
    'SPEED': 'NUMEX',
    'SPORTSEVENT': 'EVENT',
    'SPORTSSEASON': 'EVENT',
    'SPORTSTEAM': 'ORGANISATIONS',
    'STADIUM': 'FACILITY',
    'STATE': 'LOCACTION',
    'STATION': 'FACILITY',
    'STREET': 'FACILITY',
    'SUBURB': 'LOCACTION',
    'TEMPERATURE': 'NUMEX',
    'TIME': 'TIMEX',
    'TVSHOW': 'WOA',
    'UNIT': 'NUMEX',
    'VEHICLE': 'MISC',
    'WAR': 'EVENT',
    'WEAPON': 'MISC',
    'WEIGHT': 'NUMEX',
    'WOA': 'WOA',
    'YEAR': 'TIMEX'
}


class SpanDataset(Dataset):
    logger = logging.getLogger('SpanDataset')

    def __init__(
            self,
            class_list: List[str],
            sequences: List[List[str]] = None,
            entities: List[List[Dict[str, Any]]] = None,
            batch_encoding: BatchEncoding = None,
            pad_token='PAD',
            max_span_length=8,
            add_super_classes=False
    ):
        super(SpanDataset, self).__init__()
        self.class_list = class_list
        self.pad_token = pad_token
        self.max_span_length = max_span_length

        self.entities = entities
        self.sequences = sequences
        self.batch_encoding = batch_encoding

        self.max_sequence_length = max(len(seq) for seq in self.sequences)

        self.add_super_classes = add_super_classes

    @staticmethod
    def get_min_max_indices(word_ids):
        min_token_ids, max_token_ids = {}, {}
        for token_id, word_id in enumerate(word_ids):
            max_token_ids[word_id] = token_id
            if word_id not in min_token_ids:
                min_token_ids[word_id] = token_id

        return min_token_ids, max_token_ids

    def get_label_layers(self, idx, tokenized_seq_len, token_offset=0):
        label_layers = []
        num_classes = len(self.class_list)
        for l in range(min(tokenized_seq_len, self.max_span_length)):
            n_spans_in_layer = tokenized_seq_len - l
            labels = torch.zeros((n_spans_in_layer, num_classes), requires_grad=False)
            label_layers.append(labels)

        word_ids = self.batch_encoding.word_ids(idx)
        min_token_index, max_token_index = self.get_min_max_indices(word_ids)
        for entity in self.entities[idx]:
            start, end = entity['span']
            entity_type = entity['entity_type']

            start_token_index = min_token_index[start] - 1 + token_offset
            end_token_index = max_token_index[end - 1] + token_offset

            layer_index = abs(end_token_index - start_token_index) - 1

            self.add_entity(label_layers, entity_type, layer_index, start_token_index)

            if self.add_super_classes and (entity_type := SUBSTITUTION_RULES.get(entity_type.upper())) is not None:
                self.add_entity(label_layers, entity_type, layer_index, start_token_index)

        return label_layers

    def add_entity(self, label_layers, entity_type, layer_index, start_token_index):
        if layer_index < self.max_span_length:
            label_layers[layer_index][start_token_index][self.class_list.index(entity_type)] = 1

    def __getitem__(self, i) -> Dict[str, Union[torch.Tensor, Any]]:
        encoding: Encoding = self.batch_encoding[i]
        input_ids = torch.tensor(encoding.ids, requires_grad=False)
        attention_mask = torch.tensor(encoding.attention_mask, requires_grad=False)
        context_mask = attention_mask.clone()
        context_mask[0] = 0
        context_mask[-1] = 0

        tokenized_seq_len = int(context_mask.sum())

        label_layers = self.get_label_layers(i, tokenized_seq_len)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_layers,
            'context_mask': context_mask,
            'seq_length': int(context_mask.sum()),
            'pre_padding': 1,
            'word_index': None,
            'sequences': None,
        }

    def __len__(self):
        return len(self.sequences)
