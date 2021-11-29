from typing import List

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.linear import PoolingSpanClassificationModel


class SimpleRNNSpanClassificationModel(PoolingSpanClassificationModel):
    def __init__(
            self,
            *args,
            rnn_type='lstm',
            rnn_hidden_size=128,
            rnn_bidirectional=True,
            rnn_num_layers=1,
            rnn_dropout=0.0,
            **kwargs
    ):
        """
        Abstract base model for the RNN span classification models.

        :param args: Positional arguments for the PoolingSpanClassificationModel.
        :param rnn_type: The type of RNN to use. May be 'lstm' or 'gru'. Default: 'lstm'
        :param rnn_hidden_size: The hidden size of the RNN. Default: 128
        :param rnn_bidirectional: If True, a bi-directional version of the chosen RNN will be used. Default: True
        :param rnn_num_layers: The number of RNN layers. Default: 1
        :param rnn_dropout: The dropout between RNN layers.
        :param kwargs: Keyword arguments for the PoolingSpanClassificationModel.
        """
        self.rnn_cls = {
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }[rnn_type.lower()]

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_dropout = rnn_dropout

        super(SimpleRNNSpanClassificationModel, self).__init__(*args, **kwargs)

    def _init_classifier(self):
        if self.reproject_lm:
            rnn_input_dim = self.reproject_lm_dim
        else:
            rnn_input_dim = self.lm_hidden_size

        self.rnn = self.rnn_cls(
            rnn_input_dim,
            self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=self.rnn_bidirectional,
            dropout=self.rnn_dropout,
            batch_first=True,
        )

        self._init_samesize_classifier(self.rnn_feature_dim)

    @property
    def num_directions(self):
        return 1 + int(self.rnn_bidirectional)

    @property
    def rnn_feature_dim(self):
        return self.rnn_hidden_size * self.num_directions

    def rnn_forward_pass(self, rnn: torch.nn.RNNBase, rnn_input, lengths):
        # Sort input by length
        sorted_lengths, sorted_indices = lengths.flatten().sort(descending=True)
        sorted_input = rnn_input[sorted_indices]

        # Filter padding-only spans
        non_padding_spans = sorted_lengths.gt(0)
        span_input = sorted_input[non_padding_spans]
        span_lengths = sorted_lengths[non_padding_spans].cpu()

        # Pack -> Forward Pass -> Unpack
        packed_input = pack_padded_sequence(span_input, span_lengths, batch_first=True)
        rnn_output, _ = rnn.forward(packed_input)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=rnn_input.size(1))

        # Add padding-only sequences back
        if rnn_output.size(0) < rnn_input.size(0):
            missing_spans = rnn_input.size(0) - rnn_output.size(0)
            rnn_output = torch.cat(
                (rnn_output, torch.zeros(missing_spans, *rnn_output.shape[1:], device=self.device)),
                dim=0
            )

        # Reverse sort
        _, reverse_indices = sorted_indices.sort(descending=False)
        rnn_output = rnn_output[reverse_indices]

        return rnn_output


class RNNSpanClassificationModel(SimpleRNNSpanClassificationModel):
    """
    This model classifies spans by the RNN representation of their last token. With a bi-directional RNN the
    representation of the last token of the forward pass is concatenated with the representation of the first
    token of the backward pass.
    """

    def forward_layers(self, sequence_output: torch.Tensor, context_mask: torch.Tensor, **kwargs):
        batch_size, seq_len = sequence_output.size()[:2]

        layer_outputs = []
        for i in range(min(seq_len, self.max_span_length)):
            span_length = i + 1
            n_spans_in_layer = seq_len - i

            # Shape: batch_size, seq_len - span_length, input_feature_dim, span_length
            unfolded_input: torch.Tensor = sequence_output.unfold(1, span_length, 1)
            _seq_len, feature_dim = unfolded_input.shape[1:3]

            # Transpose features to last dimension
            unfolded_input = unfolded_input.transpose(3, 2)

            # Reshape unfolded sequences to 3D tensor
            # Shape: batch_size * _seq_len, span_length, feature_dim
            unfolded_input = unfolded_input.reshape(-1, span_length, feature_dim)

            # Sort span inputs descending by length of non-padding tokens
            unfolded_masks: torch.Tensor = context_mask.unfold(1, span_length, 1)
            unfolded_lengths = unfolded_masks.sum(-1).flatten()

            # Get the representations of last and first token for forward and backward directions, respectively
            layer_output = self.rnn_head_tail(unfolded_input, unfolded_lengths)

            # Reshape output back to batch × spans × features
            layer_output = layer_output.view(batch_size, n_spans_in_layer, self.rnn_feature_dim)

            layer_outputs.append(layer_output)
        return layer_outputs

    def rnn_head_tail(self, rnn_input, lengths):
        # Sort input by length
        sorted_lengths, sorted_indices = lengths.sort(descending=True)
        sorted_inputs = rnn_input[sorted_indices]

        # Filter padding-only spans
        non_padding_spans = sorted_lengths.gt(0)
        span_input = sorted_inputs[non_padding_spans]
        span_lengths = sorted_lengths[non_padding_spans]

        # Pack sequences for proper bidirectional processing of paddings
        packed_spans = pack_padded_sequence(span_input, span_lengths.cpu(), batch_first=True)
        packed_output, _ = self.rnn.forward(packed_spans)

        # Unpack sequences
        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=span_input.size(1))

        # Separate directions
        _batch_size = span_input.size(0)
        _n_seqs = span_input.size(1)
        rnn_output = rnn_output.view(_batch_size, _n_seqs, self.num_directions, self.rnn_hidden_size)

        # Get RNN features of last span sub-token
        layer_output = rnn_output[:, -1, 0]
        if self.rnn_bidirectional:
            layer_output = torch.hstack((layer_output, rnn_output[:, 0, 1]))
        layer_output = torch.relu(layer_output)

        # Add padding-only sequences back
        if layer_output.size(0) < rnn_input.size(0):
            missing_spans = rnn_input.size(0) - layer_output.size(0)
            layer_output = torch.cat(
                (layer_output, torch.zeros(missing_spans, self.rnn_feature_dim, device=self.device)),
                dim=0
            )

        # Reverse sort
        _, reverse_indices = sorted_indices.sort(descending=False)
        layer_output = layer_output[reverse_indices]

        return layer_output


class PyramidModel(SimpleRNNSpanClassificationModel):
    """
    This model is an implementation of the *Pyramid* hierarchical span classification model (Wang et al., 2020).
    """

    def _init_classifier(self):
        if self.reproject_lm:
            rnn_input_dim = self.reproject_lm_dim
        else:
            rnn_input_dim = self.lm_hidden_size

        self._init_pyramid(rnn_input_dim)

        self._init_samesize_classifier(self.rnn_feature_dim)

    def _init_pyramid(self, rnn_input_dim):
        self.pyramid_rnn = self.rnn_cls(
            rnn_input_dim,
            self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=self.rnn_bidirectional,
            dropout=self.rnn_dropout,
            batch_first=True,
        )
        self.conv = torch.nn.Conv1d(self.rnn_feature_dim, self.rnn_feature_dim, 2)
        self.layer_norm = torch.nn.LayerNorm(self.rnn_feature_dim)

    def forward_layers(self, sequence_output: torch.Tensor, context_mask: torch.Tensor, **kwargs):
        batch_size, seq_len = sequence_output.size()[:2]

        layer_outputs = []
        for _ in range(min(seq_len, self.max_span_length)):
            pyramid_repr, sequence_output, context_mask = self.pyramid_forward(sequence_output, context_mask)
            layer_outputs.append(pyramid_repr)

            # When the sequence length has reached one, we are done
            if context_mask.size(1) <= 1:
                break

        return layer_outputs

    def pyramid_forward(self, sequence_output, context_mask):
        pyramid_repr = self.layer_norm(sequence_output)
        # TODO: Pyramid adds another dropout layer here, but I believe it's unnecessary?
        pyramid_repr = self.rnn_forward_pass(self.pyramid_rnn, pyramid_repr, context_mask.sum(-1))
        pyramid_repr = F.dropout(pyramid_repr, self.dropout, self.training)

        # Find valid spans in next layer
        context_mask = context_mask.unfold(1, 2, 1).min(-1).values

        # Convolve neighboring spans
        sequence_output = self.conv(pyramid_repr.transpose(2, 1)).transpose(2, 1)

        return pyramid_repr, sequence_output, context_mask


class LocalizedPyramidModel(PyramidModel, RNNSpanClassificationModel):
    """
    A :class:`PyramidModel` variant that also passes the local span representations from the
    :class:`RNNSpanClassificationModel` to the classifier at each step.
    """

    def _init_classifier(self):
        if self.reproject_lm:
            rnn_input_dim = self.reproject_lm_dim
        else:
            rnn_input_dim = self.lm_hidden_size

        self.rnn = self.rnn_cls(
            rnn_input_dim,
            self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=self.rnn_bidirectional,
            dropout=self.rnn_dropout,
            batch_first=True,
        )

        self._init_pyramid(rnn_input_dim)

        self._init_samesize_classifier(self.rnn_feature_dim * 2)

    def forward_layers(self, sequence_output: torch.Tensor, context_mask: torch.Tensor, **kwargs):
        pyramid_outputs = PyramidModel.forward_layers(self, sequence_output, context_mask, **kwargs)
        local_layer_outputs = RNNSpanClassificationModel.forward_layers(self, sequence_output, context_mask, **kwargs)

        return [
            torch.cat((pyramid, local), dim=-1)
            for pyramid, local in zip(pyramid_outputs, local_layer_outputs)
        ]


class PassThroughPyramidModel(PyramidModel):
    def __init__(self, *args, feature_pooling='max', **kwargs):
        """
        A :class:`PyramidModel` variant that also passes the aggregated input token representations to the classifier.
        Supports 'exhaustive' feature aggregation as well as 'min', 'max' and 'mean' pooling.

        See also: :class:`ExhaustiveRegionClassificationModel`

        :param args: Positional arguments for the PyramidModel.
        :param feature_pooling: The feature pooling method. Default: 'max'
        :param kwargs: Keyword arguments for the PyramidModel.
        """
        if feature_pooling == 'cat':
            raise ValueError(f"Feature pooling method 'cat' is not valid for {self.__class__.__name__}!")
        super().__init__(*args, feature_pooling=feature_pooling, **kwargs)

    def _init_samesize_classifier(self, classifier_input_dim):
        if self.reproject_lm:
            lm_output_dim = self.reproject_lm_dim
        else:
            lm_output_dim = self.lm_hidden_size

        if self.feature_pooling == 'exhaustive':
            lm_output_dim *= 3

        super(PassThroughPyramidModel, self)._init_samesize_classifier(
            classifier_input_dim + lm_output_dim
        )

    def forward_layers(self, sequence_output: torch.Tensor, context_mask: torch.Tensor, **kwargs):
        batch_size, seq_len = sequence_output.size()[:2]

        layer_outputs = []
        _orig = sequence_output
        for i in range(min(seq_len, self.max_span_length)):
            span_length = i + 1
            pass_through_repr = self.aggregate_features(_orig.unfold(1, span_length, 1), self.feature_pooling)
            pass_through_repr = torch.relu(pass_through_repr)

            pyramid_repr, sequence_output, context_mask = self.pyramid_forward(sequence_output, context_mask)

            # Combine ERC and Pyramid representations
            layer_output = torch.cat((pass_through_repr, pyramid_repr), dim=-1)
            layer_outputs.append(layer_output)

            # When the sequence length has reached one, we are done
            if context_mask.size(1) <= 1:
                break

        return layer_outputs


class HierarchicalRNNSpanClassificationModel(RNNSpanClassificationModel):
    def __init__(self, *args, hierarchical_feature_pooling='max', hierarchical_feature_dim=-1, **kwargs):
        """
        This model passes the output of the classifier between layers, as opposed to the :class:`PyramidModel` where
        span representations are passed between layers. It supports a linear encoding layer as well as min, max and mean
        pooling.

        :param args: Positional arguments for the RNNSpanClassificationModel.
        :param hierarchical_feature_pooling: The method for hierarchical feature pooling. Can be 'min', 'max', 'mean'
            or 'linear'. Default: 'max'
        :param hierarchical_feature_dim: The linear layer feature size, when choosing the 'linear' pooling method.
        :param kwargs: Keyword arguments for the RNNSpanClassificationModel.
        """
        self.hierarchical_feature_pooling = hierarchical_feature_pooling
        self.hierarchical_feature_dim = hierarchical_feature_dim

        super().__init__(*args, **kwargs)

    def _init_samesize_classifier(self, classifier_input_dim):
        if self.hierarchical_feature_pooling == 'linear':
            if self.hierarchical_feature_dim < 0:
                self.hierarchical_feature_dim = self.number_of_classes * abs(self.hierarchical_feature_dim)
            self.hierarchical_linear = nn.Linear(self.number_of_classes * 2, self.hierarchical_feature_dim)
        else:
            self.hierarchical_feature_dim = self.number_of_classes * (1 + int(self.feature_pooling == 'cat'))

        super(HierarchicalRNNSpanClassificationModel, self)._init_samesize_classifier(
            classifier_input_dim + self.hierarchical_feature_dim
        )

    def hierarchical_aggregation(self, prev_layer_outputs: torch.Tensor):
        prev_layer_outputs = prev_layer_outputs.unfold(1, 2, 1)

        if self.hierarchical_feature_pooling == 'linear':
            batch_size, seq_len, feature_dim = prev_layer_outputs.shape[:3]
            prev_layer_outputs = prev_layer_outputs.reshape(batch_size, seq_len, 2 * feature_dim)
            hierarchical_repr = self.hierarchical_linear.forward(torch.tanh(prev_layer_outputs))
        else:
            hierarchical_repr = self.aggregate_features(prev_layer_outputs, self.hierarchical_feature_pooling)

        hierarchical_repr = torch.tanh(hierarchical_repr)
        return hierarchical_repr

    def forward_classifier(self, layer_outputs: List[torch.Tensor], **kwargs):
        classifier_outputs: List[torch.Tensor] = []
        hierarchical_repr = torch.zeros(
            (*layer_outputs[0].shape[:2], self.hierarchical_feature_dim),
            device=self.device
        )
        for i, layer_output in enumerate(layer_outputs):
            classifier = self.classifier if self.single_classifier else self.classifiers[i]
            classifier_outputs.append(classifier.forward(torch.cat((layer_output, hierarchical_repr), dim=-1)))

            if hierarchical_repr.size(1) <= 1:
                break

            hierarchical_repr = self.hierarchical_aggregation(classifier_outputs[i])
        return classifier_outputs


class ExhaustiveRegionClassificationModel(SimpleRNNSpanClassificationModel):
    """
    This model is an implementation of the Exhaustive Region Classification model (Sohrab and Miwa, 2018).
    The model uses a concatenation of three features: the first and last representations of the
    Bi-LSTM over the span in question and an average of all token representations of that span.

    In total, a final representation R(i, j) equal to 3 times the size of the RNN hidden size is passed to the
    classifier.
    """

    def _init_samesize_classifier(self, classifier_input_dim):
        super(ExhaustiveRegionClassificationModel, self)._init_samesize_classifier(classifier_input_dim * 3)

    def forward_layers(self, sequence_output: torch.Tensor, context_mask: torch.Tensor, **kwargs):
        batch_size, seq_len = sequence_output.size()[:2]

        rnn_output = self.rnn_forward_pass(self.rnn, sequence_output, context_mask.sum(-1))

        layer_outputs = []
        for i in range(min(seq_len, self.max_span_length)):
            span_length = i + 1

            exhaustive_repr = self.aggregate_features(rnn_output.unfold(1, span_length, 1), 'exhaustive')
            exhaustive_repr = torch.relu(exhaustive_repr)
            layer_outputs.append(exhaustive_repr)

        return layer_outputs


class ExhaustivePyramidalModel(PyramidModel):
    """
    This model combines the hierarchical representations from the :class:`PyramidModel` with the local span
    representations from :class:`ExhaustiveRegionClassificationModel`. It differs from the
    :class:`PassThroughPyramidModel` in that the input tokens are encoded in a separate RNN for the ERC representations
    and then aggregated in each layer.
    """

    def _init_classifier(self):
        if self.reproject_lm:
            rnn_input_dim = self.reproject_lm_dim
        else:
            rnn_input_dim = self.lm_hidden_size

        self.rnn = self.rnn_cls(
            rnn_input_dim,
            self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=self.rnn_bidirectional,
            dropout=self.rnn_dropout,
            batch_first=True,
        )

        self.pyramid_rnn = self.rnn_cls(
            rnn_input_dim,
            self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=self.rnn_bidirectional,
            dropout=self.rnn_dropout,
            batch_first=True,
        )
        self.conv = torch.nn.Conv1d(self.rnn_feature_dim, self.rnn_feature_dim, 2)
        self.layer_norm = torch.nn.LayerNorm(self.rnn_feature_dim)

        self._init_samesize_classifier(self.rnn_hidden_size * self.num_directions * 4)

    def forward_layers(self, sequence_output: torch.Tensor, context_mask: torch.Tensor, **kwargs):
        batch_size, seq_len = sequence_output.size()[:2]

        encoded_sequence = self.rnn_forward_pass(self.rnn, sequence_output, context_mask.sum(-1))

        layer_outputs = []
        for i in range(min(seq_len, self.max_span_length)):
            span_length = i + 1
            exhaustive_repr = self.aggregate_features(encoded_sequence.unfold(1, span_length, 1), 'exhaustive')
            exhaustive_repr = torch.relu(exhaustive_repr)

            pyramid_repr, sequence_output, context_mask = self.pyramid_forward(sequence_output, context_mask)

            # Combine ERC and Pyramid representations
            layer_output = torch.cat((exhaustive_repr, pyramid_repr), dim=-1)
            layer_outputs.append(layer_output)

            # When the sequence length has reached one, we are done
            if context_mask.size(1) <= 1:
                break

        return layer_outputs
