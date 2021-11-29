import argparse
import logging
import os
from functools import partial
from typing import Union

from utils.cli import add_boolean_argument, add_choice_argument

console_logger = logging.getLogger("pytorch_lightning")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'variant',
        type=str,
        choices=('rnn', 'rnn_exhaustive', 'rnn_pyramid', 'rnn_pyramid_local', 'rnn_exhpyr', 'rnn_passpyr',
                 'rnn_hierarchical', 'rnn_tfidf', 'multi_linear')
    )
    parser.add_argument('--language_model', '--language-model', type=str, default='xlm-roberta-large')
    parser.add_argument('--max_span_length', '--max-span-length', type=int, default=8)
    parser.add_argument('--num_workers', '--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--apex', action='store_const', const='apex', default='native')
    add_boolean_argument(parser, 'single_classifier', True)

    add_choice_argument(parser, 'corpus', ('nne', 'genia'), 'nne')

    parser.add_argument('--check_val_every_n_epoch', '--check-val-every-n-epoch', type=int, default=1)
    add_choice_argument(parser, 'logger', ('neptune', 'comet', 'testtube', 'wandb', 'none'), default='wandb')
    add_boolean_argument(parser, 'offline_logging', False)

    add_boolean_argument(parser, 'shuffle_train', True)
    add_boolean_argument(parser, 'use_cache', False)

    parser.add_argument('--subword_pooling', choices=('none', 'first'), default='first')

    add_choice_argument(parser, 'dataset', ('base', 'TokenWindow'), 'base')
    parser.add_argument('--token_window', '--token-window', type=int, default=64)
    parser.add_argument('--add_super_classes', '--add-super-classes', action='store_true', default=False)

    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', '--batch-size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--early_stopping', '--early-stopping', type=int, default=10)
    add_boolean_argument(parser, 'stochastic_weight_avg', True)
    add_choice_argument(parser, 'optimizer', ('sgd', 'adam', 'adamw'), 'adamw')
    parser.add_argument('--gradient_clip_algorithm', '--gradient-clip-algorithm', type=str, default='norm',
                        choices=('norm', 'value'))
    parser.add_argument('--gradient_clip_val', '--gradient-clip-val', type=float, default=2.0)
    parser.add_argument('--loss_reduction', '--loss-reduction', type=str, default='mean', choices=('mean', 'sum'))

    reproject_group = parser.add_mutually_exclusive_group()
    reproject_group.add_argument(
        '--reproject_lm',
        dest='reproject_lm',
        nargs='?',
        const='linear',
        choices=('none', 'linear', 'rnn', 'lstm', 'gru'),
    )
    reproject_group.add_argument('--no-reproject_lm', dest='reproject_lm', action='store_const', const='none')
    parser.add_argument('--reproject_lm_dim', '--reproject-lm-dim', type=int, default=256)
    parser.set_defaults(**{'reproject_lm': 'linear'})

    add_boolean_argument(parser, 'lm_finetune', False)
    add_boolean_argument(parser, 'lm_exclude_embedding', False)
    parser.add_argument('--lm_layer_aggregation', '--lm-layer-aggregation', choices=('cat', 'mean', 'max', 'min'),
                        default='mean')
    parser.add_argument(
        '--lm_layers', '--lm-layers',
        type=str,
        default='all',
        help="The LM layers to use. May be a single layer or a range of layers as given by '-2:' or '8:12'. "
             "Defaults to 'all', which is equal to '0:'."
    )

    # For Multi-Linear and Hierarchical RNN tagger
    parser.add_argument('--feature_pooling', '--feature-pooling', type=str, default='max',
                        choices=('cat', 'mean', 'max', 'min', 'exhaustive'))

    # Multi-Linear tagger arguments
    parser.add_argument('--force_auto_opt', '--force-auto-opt', action='store_true', default=False)
    add_boolean_argument(parser, 'optimize_separately', False)

    # RNN tagger arguments
    add_choice_argument(parser, 'rnn_type', ('lstm', 'gru'), 'lstm')
    add_boolean_argument(parser, 'rnn_bidirectional', True)
    parser.add_argument('--rnn_hidden_size', '--rnn-hidden-size', type=int, default=128)
    parser.add_argument('--rnn_num_layers', '--rnn-num-layers', type=int, default=1)
    parser.add_argument('--rnn_dropout', '--rnn-dropout', type=float, default=0.0)
    parser.add_argument('--hierarchical', action='store_true', default=False)

    parser.add_argument('--hierarchical_feature_pooling', type=str, default='max',
                        choices=('cat', 'mean', 'max', 'min', 'linear'))
    parser.add_argument('--hierarchical_feature_dim', type=int, default=-1,
                        help="If negative value, will infer from number of classes as `n * abs(val)`.")

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()

    # Post-correct arguments
    args.use_cache = args.use_cache and not args.lm_finetune and not args.shuffle_train
    args.dataset = 'base' if args.variant == 'rnn_tfidf' else args.dataset

    return args


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Post-correct arguments
    args.use_cache = args.use_cache and not args.lm_finetune and not args.shuffle_train
    args.dataset = 'base' if args.variant == 'rnn_tfidf' else args.dataset

    # Pyramidal model require same size input from LM and lower layers
    if args.variant in ('rnn_pyramid', 'rnn_passpyr', 'rnn_exhpyr', 'rnn_pyramid_local'):
        bidi = (1 + int(args.rnn_bidirectional))
        rnn_size = bidi * args.rnn_hidden_size
        if args.reproject_lm is not None and args.reproject_lm != 'none':
            if rnn_size != args.reproject_lm_dim:
                raise ValueError(
                    f"Reprojection size and RNN size must match for pyramidal models, but got "
                    f"{bidi}×rnn_hidden_size={rnn_size} & reproject_lm_dim={args.reproject_lm_dim}!"
                )
        else:
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(args.language_model)
            if rnn_size != model_config.hidden_size:
                raise ValueError(
                    f"RNN size must match embedding size, but got "
                    f"{bidi}×rnn_hidden_size={rnn_size} & language_model.hidden_size={model_config.hidden_size}!"
                )

        if args.lm_layer_aggregation == 'cat':
            console_logger.warning(f"Detected pyramidal model but lm_layer_aggregation='cat'! Use with caution, "
                                   f"{bidi}×rnn_hidden_size must match concatenated language model embeddings size.")

    import comet_ml
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ProgressBar
    from pytorch_lightning.loggers import CometLogger, NeptuneLogger, TestTubeLogger, WandbLogger
    from pytorch_lightning.loggers.base import DummyLogger
    from transformers import AutoTokenizer

    from utils.data.datamodule import DocumentFeaturesSpanDataModule, MultiLabelSpanDataModule, \
        TokenWindowSpanDataModule
    from model.contextual import ContextualRNNSpanClassificationModel
    from model.linear import PoolingSpanClassificationModel, genia_entities, nne_entities
    from model.recurrent import ExhaustivePyramidalModel, ExhaustiveRegionClassificationModel, \
        HierarchicalRNNSpanClassificationModel, \
        LocalizedPyramidModel, PassThroughPyramidModel, PyramidModel, RNNSpanClassificationModel

    non_default_values = [
        "_".join((key, str(value)))
        for key, value in vars(args).items()
        if parser.get_default(key) != value and 'log' not in key and key != 'variant'
    ]
    non_default_values = [f"variant_{args.variant}"] + list(sorted(non_default_values))
    non_default_values = "-".join(non_default_values)

    experiment_name = f"{non_default_values}" if non_default_values else f"{args.variant}"
    if args.logger == 'comet':
        comet_ml.init(
            disable_auto_logging=True,
            logging_console=False
        )
        logger = CometLogger(
            save_dir='comet_logs/',
            api_key=os.getenv('COMET_API_KEY'),
            rest_api_key=os.getenv('COMET_API_KEY'),
            project_name='master-thesis',
            workspace='m-stoeckel',
            experiment_name=experiment_name,
            offline=args.offline_logging
        )
    elif args.logger == 'neptune':
        logger = NeptuneLogger(
            api_key=os.getenv('NEPTUNE_API_TOKEN'),
            project_name='m-stoeckel/Master-Thesis',
            experiment_name=experiment_name,
            tags=[args.variant, args.corpus, args.dataset, args.subword_pooling]
        )
    elif args.logger == 'wandb':
        logger = WandbLogger(
            project='master-thesis',
            entity='m-stoeckel',
            name=experiment_name,
            log_model=False
        )
    elif args.logger == 'testtube':
        logger = TestTubeLogger(
            'tb_logs',
            experiment_name
        )
    else:
        logger = DummyLogger()

    logger.log_hyperparams(vars(args))
    logger.log_hyperparams({
        'name': experiment_name
    })

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    if args.corpus == "nne":
        entities = nne_entities
        if args.variant == 'rnn_tfidf':
            nne_datamodule = DocumentFeaturesSpanDataModule(
                tokenizer,
                "data/nne_raw/",
                "train/",
                "test/",
                "dev/",
                max_span_length=args.max_span_length,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle_train=args.shuffle_train,
                add_super_classes=args.add_super_classes,
                subword_pooling=args.subword_pooling,
            )
        elif args.variant == 'rnn_exhaustive' or args.dataset == 'base':
            nne_datamodule = MultiLabelSpanDataModule(
                tokenizer,
                "data/nne_concat/",
                "train.mrg",
                "test.mrg",
                "dev.mrg",
                max_span_length=args.max_span_length,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle_train=args.shuffle_train,
                add_super_classes=args.add_super_classes,
                subword_pooling=args.subword_pooling,
            )
        else:  # args.dataset == 'TokenWindow'
            assert args.token_window > 0

            nne_datamodule = TokenWindowSpanDataModule(
                tokenizer,
                "data/nne_raw/",
                "train/",
                "test/",
                "dev/",
                window_size=args.token_window,
                max_span_length=args.max_span_length,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle_train=args.shuffle_train,
                add_super_classes=args.add_super_classes,
                subword_pooling=args.subword_pooling,
            )
    else:  # GENIA
        nne_datamodule = MultiLabelSpanDataModule(
            tokenizer,
            "data/genia/",
            "train.mrg",
            "test.mrg",
            "dev.mrg",
            max_span_length=args.max_span_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_train=args.shuffle_train,
            add_super_classes=args.add_super_classes,
            subword_pooling=args.subword_pooling,
            offset_entity_end=False
        )
        entities = genia_entities

    if args.variant.startswith('rnn'):
        if args.variant == 'rnn_hierarchical':
            tagger_cls = partial(
                HierarchicalRNNSpanClassificationModel,
                hierarchical_feature_pooling=args.hierarchical_feature_pooling,
                hierarchical_feature_dim=args.hierarchical_feature_dim,
            )
        elif args.variant == 'rnn_tfidf':
            tagger_cls = ContextualRNNSpanClassificationModel
        elif args.variant == 'rnn_exhaustive':
            tagger_cls = ExhaustiveRegionClassificationModel
        elif args.variant == 'rnn_pyramid':
            tagger_cls = PyramidModel
        elif args.variant == 'rnn_passpyr':
            tagger_cls = PassThroughPyramidModel
        elif args.variant == 'rnn_exhpyr':
            tagger_cls = ExhaustivePyramidalModel
        elif args.variant == 'rnn_pyramid_local':
            tagger_cls = LocalizedPyramidModel
        else:
            tagger_cls = RNNSpanClassificationModel

        tagger_cls = partial(
            tagger_cls,
            rnn_type=args.rnn_type,
            rnn_hidden_size=args.rnn_hidden_size,
            rnn_bidirectional=args.rnn_bidirectional,
            rnn_num_layers=args.rnn_num_layers,
            rnn_dropout=args.rnn_dropout,
        )

        trainer_cls = partial(
            pl.Trainer,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            gradient_clip_val=args.gradient_clip_val,
            stochastic_weight_avg=args.stochastic_weight_avg,
        )
    else:  # args.variant == 'multi_linear':
        tagger_cls = PoolingSpanClassificationModel
        trainer_cls = partial(
            pl.Trainer,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            gradient_clip_val=args.gradient_clip_val,
            stochastic_weight_avg=args.stochastic_weight_avg,
        )

    tagger: Union[
        HierarchicalRNNSpanClassificationModel, RNNSpanClassificationModel, PoolingSpanClassificationModel
    ] = tagger_cls(
        language_model=args.language_model,
        entities_lexicon=entities,
        dropout=args.dropout,
        reproject_lm=args.reproject_lm,
        reproject_lm_dim=args.reproject_lm_dim,
        lr=args.lr,
        momentum=args.momentum,
        patience=args.patience,
        max_span_length=args.max_span_length,
        optimizer=args.optimizer,
        feature_pooling=args.feature_pooling,
        single_classifier=args.single_classifier,
        lm_layers=args.lm_layers,
        lm_layer_aggregation=args.lm_layer_aggregation,
        lm_exclude_embedding=args.lm_exclude_embedding,
        lm_finetune=args.lm_finetune,
        use_cache=args.use_cache,
        loss_reduction=args.loss_reduction,
        subword_pooling=args.subword_pooling
    )

    if isinstance(logger, CometLogger):
        summary = tagger.summarize('top')
        logger.experiment.log_html(
            f"<h2>Model Summary</h2>\n"
            f"<pre>{summary}</pre>"
        )
    elif isinstance(logger, NeptuneLogger):
        summary = tagger.summarize('top')
        logger.experiment.log_text(
            "model_summary",
            f"==Model Summary==\n"
            f"{summary}"
        )
    elif isinstance(logger, WandbLogger):
        import wandb

        summary = tagger.summarize('top')
        logger.experiment.log({
            "model_summary": wandb.Html(
                f"<h2>Model Summary</h2>\n"
                f"<pre>{summary}</pre>"
            )
        })

    if not args.subword_pooling:
        console_logger.warning("Subword Pooling is disabled. "
                               "This is not recommended and will inevitably hurt performance!")

    trainer: pl.Trainer = trainer_cls(
        gpus=args.gpus,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        logger=logger,
        amp_backend=args.apex,
        callbacks=[
            ProgressBar(),
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping('val/loss', patience=args.early_stopping)
        ],
        checkpoint_callback=False,
        weights_summary='top',
        log_every_n_steps=50
    )

    trainer.fit(tagger, datamodule=nne_datamodule)
    trainer.test(tagger, datamodule=nne_datamodule)
