import argparse
from typing import Any, Iterable


def add_boolean_argument(parser: argparse.ArgumentParser, name: str, default=False):
    name = name.lstrip('-')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def add_choice_argument(
        parser: argparse.ArgumentParser,
        dest: str,
        choices: Iterable[Any],
        default: Any,
        type=str,
        required=False
):
    dest = dest.lstrip('--')
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--' + dest, type=type, choices=choices, dest=dest)
    for choice in choices:
        group.add_argument('--' + str(choice), const=choice, action='store_const', dest=dest)
    parser.set_defaults(**{dest: default})
