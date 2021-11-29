from typing import Any, Dict, List, Tuple, Union

import torch

from model.recurrent import RNNSpanClassificationModel


def unwrap_batch(
        batch: Dict[str, Union[torch.Tensor, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int], List[int], torch.Tensor]:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    context_mask = batch['context_mask']
    seq_length = batch['seq_length']
    pre_padding = batch['pre_padding']
    word_index = batch['word_index']
    document_features = batch['document_features']

    return input_ids, attention_mask, labels, context_mask, seq_length, pre_padding, word_index, document_features


class ContextualRNNSpanClassificationModel(RNNSpanClassificationModel):
    def __init__(
            self,
            *args,
            contextual_embeddings_dim=3200,
            reproject_contextual_embeddings=True,
            reproject_contextual_embeddings_dim=64,
            **kwargs
    ):
        self.reproject_contextual_embeddings = reproject_contextual_embeddings
        if self.reproject_contextual_embeddings:
            self.contextual_hidden_size = reproject_contextual_embeddings_dim
        else:
            self.contextual_hidden_size = contextual_embeddings_dim

        super(ContextualRNNSpanClassificationModel, self).__init__(*args, **kwargs)

        if self.reproject_contextual_embeddings:
            self.reproject_contextual = torch.nn.Linear(contextual_embeddings_dim, reproject_contextual_embeddings_dim)

    def _init_samesize_classifier(self, classifier_input_dim):
        classifier_input_dim = classifier_input_dim + self.contextual_hidden_size
        super(ContextualRNNSpanClassificationModel, self)._init_samesize_classifier(classifier_input_dim)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, context_mask, *_batch, document_features = unwrap_batch(batch)

        if self.reproject_contextual_embeddings:
            document_features = self.reproject_contextual.forward(document_features.view(input_ids.shape[0], -1))

        logits = self(input_ids, attention_mask, context_mask, *_batch, document_features=document_features)

        return self.compute_loss(logits, labels)

    def forward_classifier(self, layer_outputs: List[torch.Tensor], document_features: torch.Tensor = None, **kwargs):
        document_features = document_features.unsqueeze(1)
        classifier_outputs = []
        for i, layer_output in enumerate(layer_outputs):
            layer_output = torch.cat((layer_output, document_features.expand(-1, layer_output.shape[1], -1)), -1)
            classifier = self.classifier if self.single_classifier else self.classifiers[i]
            classifier_outputs.append(classifier.forward(layer_output))
        return classifier_outputs

    def validation_step(self, batch, batch_idx):
        input_ids, *_batch, document_features = unwrap_batch(batch)

        if self.reproject_contextual_embeddings:
            document_features = self.reproject_contextual.forward(document_features.view(input_ids.shape[0], -1))

        results = self.evaluate(input_ids, *_batch, document_features=document_features)

        self.log("val_loss", float(results['loss']), logger=True)
        for i, layer_loss in enumerate(results['loss_per_layer']):
            self.log(f"val_loss-layer_{i:02d}", float(layer_loss), logger=True)
        return results

    def test_step(self, batch, batch_idx):
        input_ids, *_batch, document_features = unwrap_batch(batch)

        if self.reproject_contextual_embeddings:
            document_features = self.reproject_contextual.forward(document_features.view(input_ids.shape[0], -1))

        return self.evaluate(input_ids, *_batch, document_features=document_features)
