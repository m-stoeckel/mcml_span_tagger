import bisect
import itertools
import json
import re
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import torch

recurdict = lambda: defaultdict(recurdict)


def wikipedia_article_data_reader(path):
    """
    New data reader for automatically annotated Wikipedia articles.
    """
    # for example in json.load(open(path, 'r'))['rasa_nlu_data']['common_examples']:
    path = Path(path)
    with path.open('r', encoding='utf-8') as fp:
        for jsonline in fp:
            if not jsonline.strip().startswith("{\"id\":"):
                continue

            article = json.loads(jsonline)

            entities = [
                {
                    'title': entity['title'],
                    'category': entity['category'],
                    'start': entity['start'],
                    'end': entity['end']
                }
                for entity in article['entities']
            ]

            sentences = list(sorted(
                [
                    {
                        'start': entity['start'],
                        'end': entity['end']
                    }
                    for entity in article['sentences']
                ],
                key=lambda d: d['start']
            ))

            sentences_starts = [sentence['start'] for sentence in sentences]

            entity_map = defaultdict(list)
            for entity in entities:
                start = entity['start']
                sentence_index = bisect.bisect(sentences_starts, start) - 1
                entity_map[sentence_index].append(entity)

            for idx, sentence in enumerate(sentences):
                sentence_entities = entity_map[idx]

                if not sentence_entities:
                    continue

                sentence_start = sentence['start']
                sentence_end = sentence['end']
                full_text = article['text']
                sentence_text = full_text[sentence_start:sentence_end]

                tokens = sentence_text.split()
                token_indices = [match.span() for match in re.finditer(r"\S+", sentence_text)]
                start_indices, stop_indices = zip(*token_indices)

                for entity in sentence_entities:
                    entity.update({
                        'title': full_text[entity['start']:entity['end']],
                        'start': entity['start'] - sentence_start,
                        'end': entity['end'] - sentence_start,
                    })
                    entity['span'] = (start_indices.index(entity['start']), stop_indices.index(entity['end']) + 1)

                yield {
                    'tokens': tokens,
                    'entities': sentence_entities
                }


def rasa_reader(path: Union[str, PathLike]):
    path = Path(path)
    with path.open('r', encoding='utf-8') as fp:
        rasa_nlu_data = json.load(fp)["rasa_nlu_data"]

    for example in rasa_nlu_data["common_examples"]:
        text: str = example['text']
        entities: list = example['entities']

        tokens = text.split()
        token_indices = [match.span() for match in re.finditer(r"\S+", text)]
        start_indices, stop_indices = zip(*token_indices)

        for entity in entities:
            entity_title = text[entity['start']:entity['end']]
            entity_span = (start_indices.index(entity['start']), stop_indices.index(entity['end']) + 1)
            entity_type = entity['entity']

            entity.clear()
            entity.update({
                'title': entity_title,
                'span': entity_span,
                'entity_type': entity_type
            })

        yield {
            'tokens': tokens,
            'entities': entities
        }


def simple_mrg_reader(path: Union[str, PathLike], offset_end=True) -> Iterable[Dict[str, Any]]:
    """
    Reads a MRG format file, like in the PTB. By default with `offset_end=True`, format should be **end-inclusive**: ::

        Goethe University Frankfurt .
        NN NN NN .
        0,1 PER|0,2 ORG|0,3 ORG|3,3 LOC
        
    If `offset_end=False`, format should be **end-exclusive**: ::

        Goethe University Frankfurt .
        NN NN NN .
        0,2 PER|0,3 ORG|0,4 ORG|3,4 LOC

    :param path: The path to the .mrg file.
    :param offset_end: If True, will add +1 to the end index of entities.
    :return: Yields samples as a dict of 'tokens' (list of strings) and 'entities' (list of entity dicts).
    """
    path = Path(path)
    with path.open('r', encoding='utf-8') as fp:
        lines = fp.readlines()

    for idx in range(0, len(lines), 4):
        text, pos, tags = lines[idx:idx + 3]
        text, pos, tags = text.strip(), pos.strip(), tags.strip()

        entities = []
        if len(tags) > 0:
            for tag in tags.split("|"):
                offsets, category = tag.split()
                start, end = offsets.split(",")
                entities.append(
                    {
                        'entity_type': category,
                        'span': [int(start), int(end) + int(offset_end)]
                    }
                )

        yield {
            'tokens': list(text.split()),
            'entities': entities
        }


def iter_mrg(path: Union[str, PathLike]):
    yield from (file for file in Path(path).iterdir() if file.name.endswith(".mrg"))


def simple_document_mrg_reader(path):
    for file in iter_mrg(path):
        yield list(simple_mrg_reader(file.absolute()))


def simple_contextual_mrg_reader(path, window_size=5):
    for file in iter_mrg(path):
        yield from contextual_mrg_reader(file.absolute(), window_size)


def contextual_mrg_reader(file: Path, window_size=5):
    dataset = list(simple_mrg_reader(file))

    pre_buffer = []
    current = dataset.pop(0)
    post_buffer = dataset[:window_size]

    yield {
        'tokens': current['tokens'],
        'entities': current['entities'],
        'preceding': pre_buffer,
        'succeeding': post_buffer
    }

    while len(dataset) > 0:
        pre_buffer.append(current)
        pre_buffer = pre_buffer[-window_size:]
        current = dataset.pop(0)
        post_buffer = dataset[:window_size]

        yield {
            'tokens': current['tokens'],
            'entities': current['entities'],
            'preceding': pre_buffer[:],
            'succeeding': post_buffer[:]
        }


def get_file_pairs(path: Union[str, PathLike]) -> Iterable[Tuple[Path]]:
    file_stem_dict = defaultdict(list)
    for file in Path(path).iterdir():
        file_stem_dict[file.stem.split('-')[0]].append(file)

    yield from (tuple(sorted(files, key=lambda f: len(f.stem))) for files in file_stem_dict.values())


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def _document_embedding_mrg_reader(path, reader: Callable):
    file_tuples: Iterable[Tuple[Path]] = get_file_pairs(path)
    for tup in file_tuples:
        file, feature_files = tup[0], tup[1:]

        document_features = recurdict()
        for feature_file in feature_files:
            # Eg: 'wsj_0001-features_tfidf_8_stack' -> ('tfidf', '8', 'stack')
            selection_method, size, accumulation_method = feature_file.stem.split("-")[1].split("_")[1:]

            with feature_file.open('rb') as fp:
                feature = torch.load(fp)

            document_features[".".join((selection_method, size, accumulation_method))] = feature

        for sample in reader(file):
            sample['document_features'] = default_to_regular(document_features)
            yield sample


def document_embedding_mrg_reader(path):
    yield from _document_embedding_mrg_reader(path, simple_mrg_reader)


def contextualized_document_embedding_mrg_reader(path):
    yield from _document_embedding_mrg_reader(path, contextual_mrg_reader)


if __name__ == '__main__':
    print(next(itertools.islice(
        simple_mrg_reader("../../../mcml_classifier/data/nne_concat/dev.mrg"), 0, 2)
    ))
    print(next(itertools.islice(
        simple_contextual_mrg_reader("../../../mcml_classifier/data/nne_raw/dev/"), 0, 2)
    ))
    print(next(itertools.islice(
        document_embedding_mrg_reader("../../../mcml_classifier/data/nne_raw/dev/"), 0, 2)
    ))
    print(next(itertools.islice(
        rasa_reader("../../../mcml_classifier/data/genia/genia.dev.rasa"), 0, 2)
    ))
