from collections import defaultdict
from typing import Iterable

import numpy as np


class LabelDecoder:

    def __init__(self, entities: Iterable[str]):
        self.entity_array = np.array(entities, dtype=str)

    def decode(self, batch, label_masks):
        """

        :param batch: The batch of
        :param label_masks:
        :return:
        """
        return [self.decode_layer(layer, layer_index, label_masks) for (layer_index, layer) in enumerate(batch)]

    def decode_layer(self, layer, layer_index, label_masks):
        decoded_layer = []
        for (sequence, mask) in zip(layer, label_masks):
            nonzero = np.nonzero(mask)[0]
            sequence_start = nonzero[0]
            sequence_end = nonzero[-1] - layer_index + 1
            if sequence_end > sequence_start:
                decoded_layer.append([
                    indicators == 1
                    for indicators in sequence[sequence_start:sequence_end]
                ])
            else:
                decoded_layer.append([])
        return decoded_layer

    def decode_remedy(self, y_remedy):
        """Deprecated and up for removal."""
        longest_span = 0
        sequences_tags = []
        for sequence in y_remedy:
            # Keep a list for each entity class for both entities that are already finished, and those entities
            # that may be continued with the next token
            sequence_entities = defaultdict(list)
            current_entities = defaultdict(list)
            previous_entities = np.full_like(sequence[0].reshape(-1, 2)[:, 0], False, dtype=bool)

            # As the modified remedy solution predicts multiple labels per token, sequences can both begin and continue
            # on any given token. As such, all entities, that have begone on a previous token, and have not ended yet,
            # are updated each round. This may lead to misclassifications with partially overlapping entities.
            # However, the NNE dataset does not contain any partially overlapping annotations.
            for offset, logits in enumerate(sequence):
                np_logits = logits.reshape(-1, 2)
                begin_entities = np_logits[:, 0] == 1

                # Inside tag calculation
                # An inside tag requires a preceding begin tag -> multiplication with previous_entities.
                inside_entities = (np_logits[:, 1] == 1) * previous_entities

                # Add each beginning entity to its respective current list.
                begin_entities_list = self.entity_array[begin_entities].tolist()
                for begin_entity in begin_entities_list:
                    current_entities[begin_entity].append([offset, offset + 1])
                    longest_span = max(longest_span, 1)

                # Extend all continued entities by increasing their span-end offset.
                inside_entities_list = self.entity_array[inside_entities].tolist()
                for inside_entity in inside_entities_list:
                    for entity in current_entities[inside_entity]:
                        entity[1] = offset + 1
                        longest_span = max(longest_span, abs(entity[1] - entity[0]))

                previous_entities = begin_entities | inside_entities

                # Push all entities, that neither started nor continued on this token, to the sequence_entities lists
                for missing_entity in self.entity_array[np.logical_not(previous_entities)].tolist():
                    sequence_entities[missing_entity].extend(current_entities[missing_entity])
                    current_entities[missing_entity] = []

            # Push remaining entities to sequence_entities lists
            for key, value in current_entities.items():
                sequence_entities[key].extend(value)

            sequence_tags = {}
            for entity_name, entities in sequence_entities.items():
                for start, end in entities:
                    length = end - start
                    if length not in sequence_tags:
                        sequence_tags[length] = [[] for _ in range(len(sequence) - (length - 1))]
                    sequence_tags[length][start].append(entity_name)

            sequences_tags.append(sequence_tags)

        return self._decode_labels(y_remedy, sequences_tags, longest_span)

    def _decode_labels(self, y_remedy, sequences_tags, longest_span):
        decoded_labels = []
        empty = np.full_like(self.entity_array, False, dtype=bool)
        for i in range(1, longest_span + 1):
            decoded_labels_for_order = []
            for sequence, sequence_tags in zip(y_remedy, sequences_tags):
                sequence_length = max(0, len(sequence) - (i - 1))
                if i in sequence_tags:
                    span = [
                        np.in1d(self.entity_array, iob2_tag)
                        for iob2_tag in sequence_tags[i]
                    ]
                else:
                    span = [empty for _ in range(sequence_length)]
                decoded_labels_for_order.append(span)
            decoded_labels.append(decoded_labels_for_order)
        return decoded_labels
