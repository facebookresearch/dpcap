# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from enum import Enum

import torch
import torch.nn.functional as F


class Trie:
    def __init__(self):
        self.children = {}
        self.is_terminal = False

    def add(self, sequence):
        if len(sequence) >= 1:
            c = sequence[0]
            if c not in self.children:
                self.children[c] = Trie()
            self.children[c].add(sequence[1:])
        else:
            self.is_terminal = True

    def print(self, prefix=""):
        if self.is_terminal:
            print(prefix)
        for c, t in self.children.items():
            t.print(prefix + "\t" + str(c))

    def depth(self):
        max_depth = 0
        for t in self.children.values():
            max_depth = max(max_depth, 1 + t.depth())

        return max_depth

    def __eq__(self, other):
        if self.is_terminal != other.is_terminal:
            return False
        c1 = self.children.keys()
        c2 = other.children.keys()
        if set(c1) != set(c2):
            return False
        else:
            return all([self.children[c] == other.children[c] for c in c1])

    def __repr__(self):
        return "\n".join(self.__repr())

    def __repr(self, prefix=""):
        returns = []
        if self.is_terminal:
            returns.append(prefix)
        for c, t in self.children.items():
            returns.extend(t.__repr(prefix + "\t" + str(c)))

        return returns

class RowState(Enum):
    BEGIN_FIELD = 1
    END_FIELD = 2
    MID_FIELD = 3
    END_ROW = 4


class RowGuide:
    def __init__(self, trie):
        self.trie = trie
        self.end_tokens = [102, 1012, 0]# for now just(EOS, PAD, .)then we can add [:, , ;]: [1010, 1025, 1024]
        self.values = []
        self.current_seq = trie

    def next(self, distribution):
        assert distribution is None or distribution.ndim == 1
        possible_values = list(self.current_seq.children.keys())
        if self.current_seq.is_terminal:
            possible_values += self.end_tokens #end of sentence, dot, , , ;, 

        next_token = torch.argmax(distribution[possible_values])
        next_token = possible_values[next_token]
        self.values.append(next_token)

        if next_token in self.end_tokens:
            return next_token, True

        self.current_seq = self.current_seq.children[next_token]

        return next_token, False


def get_trie(dic):
    l = []
    for value in dic.values():
        for x in value:
            l.append(tuple(x))
    values = set(l)
    t = Trie()
    for value in values:
        t.add(value)

    return t
