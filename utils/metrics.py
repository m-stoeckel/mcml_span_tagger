"""
Given a list of span->class mappings,
transform the spans into BIOES format annotations for each token
and compute Precision, Recall and F1-Score for these tokens.
"""
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Set, Tuple, Union

import numpy as np
import seqeval
from seqeval.reporters import StringReporter


class DictReporter(seqeval.reporters.DictReporter):
    def write_foot(self, *args, **kwargs):
        return self.write(*args, **kwargs)


class LatexReporter(StringReporter):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.foot_buffer = []
        self.body_buffer = []
        self.digits = kwargs.get('digits', 2)
        self.name_width = kwargs.get('name_width', 10)
        self.score_width = kwargs.get('score_width', 21)
        self.support_width = kwargs.get('support_width', 16)
        self.row_fmt = '\t{:<{name_width}s}' + ' & {:<{score_width}}' * 3 + ' & {:<{support_width}} \\\\'

    def report(self):
        report = self.write_header()
        report += '\n'.join(self.foot_buffer)
        report += '\n\t\\bottomrule\\endfoot\n'
        report += '\n'.join(self.body_buffer)
        report += '\n\\end{longtable}\n'
        return report

    def _write(self, row_name, precision, recall, f1, support):
        value_fmt = '\\npp{{{:02.'f'{self.digits}''f}}}'
        precision = value_fmt.format(precision * 100)
        recall = value_fmt.format(recall * 100)
        f1 = value_fmt.format(f1 * 100)
        support = f'\\np{{{support}}}'
        return self.row_fmt.format(
            *[row_name, precision, recall, f1, support],
            name_width=self.name_width,
            score_width=self.score_width,
            support_width=self.support_width,
            digits=self.digits
        )

    def write(self, row_name: str, precision: float, recall: float, f1: float, support: int):
        row = self._write(row_name, precision, recall, f1, support)
        self.body_buffer.append(row)

    def write_foot(self, row_name: str, precision: float, recall: float, f1: float, support: int):
        row = self._write(row_name, precision, recall, f1, support)
        self.foot_buffer.append(row)

    def write_header(self):
        report = '\\begin{longtable}{lrrrr}\n' \
                 '\t\\caption[Short Caption]{Long Caption.\\label{tab:}} \\\\ % TODO: Caption\n' \
                 '\t\\toprule\n'

        headers = ['Category', 'Precision', 'Recall', 'F1-Score', 'Support']
        head_fmt = '\t{:<{name_width}}'
        head_fmt += ' & {:<{score_width}}' * 3
        head_fmt += ' & {:<{support_width}} \\\\'
        report += head_fmt.format(
            *headers,
            name_width=self.name_width,
            score_width=self.score_width,
            support_width=self.support_width
        )
        report += '\n\t\\midrule\\endhead\\midrule\n'
        return report

    def write_blank(self):
        self.body_buffer.append('')


def _bioes_encode_impl(start, end, in_ex):
    assert in_ex in (0, 1)
    start, end = sorted((start, end))
    if end - start == in_ex:
        yield start, 'S'
    else:
        yield start, 'B'
        if end - start > 1 + in_ex:
            for idx in range(start + 1, end - in_ex):
                yield idx, 'I'
        yield end - in_ex, 'E'


def bioes_encode(start: int, end: int) -> Iterator[Tuple[int, str]]:
    """
    :param start: Token start index.
    :param end: Token end index (exclusively).
    :returns: A list of BIOES encoded

    Example:

    >>> tokens = "This is Goethe University in Frankfurt / Main .".split()
    >>> annotations = [(2, 4, "ORG"), (2, 3, "PER"), (3, 4, "ORG"), (5, 8, "LOC")]
    >>> for start, end, category in annotations:
    ...     for idx, tag in bioes_encode(start, end):
    ...         print(tokens[idx], f'{tag}-{category}')
    Goethe B-ORG
    University E-ORG
    Goethe S-PER
    University S-ORG
    Frankfurt B-LOC
    / I-LOC
    Main E-LOC
    """
    yield from _bioes_encode_impl(start, end, 1)


bioes_encode_exclusively = bioes_encode


def bioes_encode_inclusively(start: int, end: int) -> Iterator[Tuple[int, str]]:
    """
    :param start: Token start index.
    :param end: Token end index (inclusively).
    :returns: A list of BIOES encoded

    Example:

    >>> tokens = "This is Goethe University in Frankfurt / Main .".split()
    >>> annotations = [(2, 3, "ORG"), (2, 2, "PER"), (3, 3, "ORG"), (5, 7, "LOC")]
    >>> for start, end, category in annotations:
    ...     for idx, tag in bioes_encode_inclusively(start, end):
    ...         print(tokens[idx], f'{tag}-{category}')
    Goethe B-ORG
    University E-ORG
    Goethe S-PER
    University S-ORG
    Frankfurt B-LOC
    / I-LOC
    Main E-LOC
    """
    yield from _bioes_encode_impl(start, end, 0)


def encode_sequence(sequence, largest_offset):
    bioes_sequence = [defaultdict(set) for _ in range(largest_offset)]
    for start, end, category in sequence:
        for idx, tag in bioes_encode(start, end):
            bioes_sequence[idx][category].add(tag)

    return bioes_sequence


def calc_fb_score(precision, recall, beta):
    precision, recall = np.array(precision), np.array(recall)
    f1_score = np.zeros_like(precision)
    p_r_sum = precision + recall
    f1_score[p_r_sum > 0] = (1. + beta ** 2.) * precision[p_r_sum > 0] * recall[p_r_sum > 0] / p_r_sum[p_r_sum > 0]
    return f1_score


def calc_f1_score(precision, recall) -> Union[np.ndarray, float]:
    return calc_fb_score(precision, recall, 1)


def compute_confusion_matrix_values_bioes(prediction, ground_truth, category_names):
    num_categories = len(category_names)

    true_positive = np.zeros(num_categories, dtype=int)
    false_positive = np.zeros(num_categories, dtype=int)
    false_negative = np.zeros(num_categories, dtype=int)

    for p_sequence, t_sequence in zip(prediction, ground_truth):
        largest_offset = max(max(tup[1] for tup in p_sequence), max(tup[1] for tup in t_sequence))

        # BIOES encoded sequences, one entry for each token
        bioes_p_sequence = encode_sequence(p_sequence, largest_offset)
        bioes_t_sequence = encode_sequence(t_sequence, largest_offset)

        for preds, truths in zip(bioes_p_sequence, bioes_t_sequence):
            preds: Dict[str, Set[str]]
            truths: Dict[str, Set[str]]

            for idx, category in enumerate(category_names):
                true_positive[idx] += len(preds[category].intersection(truths[category]))
                false_positive[idx] += len(preds[category].difference(truths[category]))
                false_negative[idx] += len(truths[category].difference(preds[category]))

    return true_positive, false_positive, false_negative


def compute_confusion_matrix_values_spans(predictions, ground_truths, category_indices):
    num_categories = category_indices.size

    true_positives = np.zeros(num_categories, dtype=int)
    false_positives = np.zeros(num_categories, dtype=int)
    false_negatives = np.zeros(num_categories, dtype=int)

    for p_sequence, t_sequence in zip(predictions, ground_truths):
        for preds, truths in zip(p_sequence, t_sequence):
            tp, fp, fn = tp_fp_fn(preds, truths)

            true_positives[tp] += 1
            false_positives[fp] += 1
            false_negatives[fn] += 1

    return true_positives, false_positives, false_negatives


def tp_fp_fn(preds, truths):
    tp = np.logical_and(preds, truths)
    fp = np.logical_and(preds, np.logical_not(truths))
    fn = np.logical_and(truths, np.logical_not(preds))
    return tp, fp, fn


def calculate_metrics(true_positives, false_positives, false_negatives):
    shape = true_positives.shape
    precision, recall = np.zeros(shape), np.zeros(shape)

    # Sums
    pred_positive = true_positives + false_positives
    act_positive = true_positives + false_negatives

    # Precision, Recall, F1-Score per category
    precision[pred_positive > 0] = true_positives[pred_positive > 0] / pred_positive[pred_positive > 0]
    recall[act_positive > 0] = true_positives[act_positive > 0] / act_positive[act_positive > 0]
    f1_score = calc_f1_score(precision, recall)

    # Macro metrics
    precision_macro = float(precision.mean())
    recall_macro = float(recall.mean())
    f1_score_macro = float(calc_f1_score(precision_macro, recall_macro))

    # Micro metrics
    precision_micro, recall_micro = 0., 0.
    if pred_positive.sum():
        precision_micro = float(true_positives.sum() / pred_positive.sum())
    if act_positive.sum():
        recall_micro = float(true_positives.sum() / act_positive.sum())
    f1_score_micro = float(calc_f1_score(precision_micro, recall_micro))

    # Weighted metrics
    if true_positives.sum():
        precision_weighted = float((precision * true_positives / true_positives.sum()).sum())
        recall_weighted = float((recall * true_positives / true_positives.sum()).sum())
        f1_score_weighted = float(calc_f1_score(precision_weighted, recall_weighted))
    else:
        precision_weighted = 0.
        recall_weighted = 0.
        f1_score_weighted = 0.

    # Packing return values
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'macro': (precision_macro, recall_macro, f1_score_macro),
        'micro': (precision_micro, recall_micro, f1_score_micro),
        'weighted': (precision_weighted, recall_weighted, f1_score_weighted)
    }


def build_reporter(metrics, category_names, support, digits=2, output_dict=False):
    if output_dict:
        reporter = DictReporter()
    else:
        name_width = max(map(len, category_names))
        avg_width = len('weighted avg')
        name_width = max(name_width, avg_width, digits)
        scores = np.hstack([metrics['precision'], metrics['recall'], metrics['f1_score']])
        score_width = 11 + int(scores.max() >= 1)
        support_width = len(str(support.max())) + 5
        reporter = LatexReporter(
            digits=digits,
            name_width=name_width,
            score_width=score_width,
            support_width=support_width
        )

    # Report building
    for idx, name in enumerate(category_names):
        reporter.write(name, metrics['precision'][idx], metrics['recall'][idx], metrics['f1_score'][idx],
                       support[idx])

    reporter.write_foot('Macro Avg' if not output_dict else 'macro avg', *metrics['macro'], support.sum())
    reporter.write_foot('Micro Avg' if not output_dict else 'micro avg', *metrics['micro'], support.sum())
    reporter.write_foot('Weighted Avg' if not output_dict else 'weighted avg', *metrics['weighted'], support.sum())

    return reporter


def multi_label_span_classification_report(
        predictions: List[List[np.ndarray]],
        ground_truths: List[List[np.ndarray]],
        category_names: List[str],
        output_dict=False,
        digits=2
):
    # category_names = sorted(category_names)
    category_indices = np.arange(len(category_names))

    true_positives, false_positives, false_negatives = compute_confusion_matrix_values_spans(
        predictions, ground_truths, category_indices
    )

    metrics = calculate_metrics(
        true_positives,
        false_positives,
        false_negatives
    )

    reporter = build_reporter(metrics, category_names, true_positives, digits=digits, output_dict=output_dict)

    return reporter.report()


def multi_label_bioes_classification_report(
        predictions: List[List[Tuple[int, int, str]]],
        ground_truths: List[List[Tuple[int, int, str]]],
        category_names: Iterable[str],
        output_dict=False,
        digits=2
):
    category_names = sorted(category_names)

    true_positive, false_positive, false_negative = compute_confusion_matrix_values_bioes(
        predictions, ground_truths, category_names
    )

    metrics = calculate_metrics(
        true_positive,
        false_positive,
        false_negative
    )

    reporter = build_reporter(metrics, category_names, true_positive, digits, output_dict)

    return reporter.report()


def test_bioes():
    categories = ["ORG", "OTH", "LOC", "PER"]
    report = multi_label_bioes_classification_report(
        [[(2, 4, "ORG"), (2, 3, "PER"), (5, 8, "OTH")]],
        [[(2, 4, "ORG"), (2, 3, "PER"), (3, 4, "ORG"), (5, 8, "LOC")]],
        categories
    )
    print("Using BIOES tags")
    print(report, end="\n\n")


def transform_to_binary_array(ll: List[str], category_names: List[str]):
    indices = [category_names.index(el) for el in ll]
    ret = np.full(len(category_names), False)
    ret[indices] = True
    return ret


def test_spans():
    categories = ["ORG", "OTH", "LOC", "PER"]
    predictions = [
        [
            transform_to_binary_array(token, list(sorted(categories)))
            for token in sequence
        ]
        for sequence in [
            [[], [], ["PER"], [], [], [], [], [], []],
            [[], ["ORG"], [], [], [], [], [], []],
            [[], [], [], [], [], ["OTH"], []]
        ]
    ]
    ground_truths = [
        [
            transform_to_binary_array(token, list(sorted(categories)))
            for token in sequence
        ]
        for sequence in [
            [[], [], ["PER"], ["ORG"], [], [], [], [], []],
            [[], ["ORG"], [], [], [], [], [], []],
            [[], [], [], [], [], ["LOC"], []]
        ]
    ]

    report = multi_label_span_classification_report(
        predictions,
        ground_truths,
        categories,
        output_dict=True
    )
    import pandas as pd
    print(pd.DataFrame.from_dict(report, orient='index').round(4).to_csv())


if __name__ == '__main__':
    # test_bioes()
    test_spans()
