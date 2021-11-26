import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from transformers import AutoTokenizer

from model.linear import genia_entities
from model.recurrent import PyramidModel
from utils.data.datamodule import MultiLabelSpanDataModule

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    max_span_length = 8

    nne_datamodule = MultiLabelSpanDataModule(
        tokenizer,
        "data/genia/",
        "train.mrg",
        "test.mrg",
        "dev.mrg",
        max_span_length=max_span_length,
        batch_size=16,
        remedy_solution=False,
        shuffle_train=True,
        offset_entity_end=False
    )

    tagger = PyramidModel(
        genia_entities,
        'bert-base-multilingual-cased',
        max_span_length=max_span_length,
        single_classifier=True,
        remedy_solution=False,
        loss_reduction='sum',
        lr=1e-3,
        use_cache=False
    )

    # logger = TestTubeLogger('tb_logs', name='MultiLabelSpan-TokenWindow-test')
    trainer = pl.Trainer(
        gpus=1,
        # logger=logger,
        check_val_every_n_epoch=1,
        limit_train_batches=0.5,
        limit_val_batches=0.2,
        limit_test_batches=0.2,
        max_epochs=5,
        gradient_clip_val=2.0,
        gradient_clip_algorithm='norm',
        log_every_n_steps=50,
    )
    trainer.fit(tagger, datamodule=nne_datamodule)
    trainer.test(tagger, datamodule=nne_datamodule)
