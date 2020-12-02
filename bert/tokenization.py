# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""
import sentencepiece as spm


class SentencePieceTokenizer:
    """Runs Google's SentencePiece tokenization."""

    def __init__(self, model_file, lowercase=False):
        self.sp = spm.SentencePieceProcessor()
        self.lowercase = lowercase
        self.sp.Load(model_file)

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        return self.sp.EncodeAsPieces(text)

    def ids_to_tokens(self, ids):
        return [self.sp.IdToPiece(int(id)) for id in ids]

    def tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]

    def create_word2id_id2word(self):
        word2id = {
            self.sp.id_to_piece(id): id
            for id in range(self.sp.get_piece_size())
        }
        id2word = dict(zip(word2id.values(), word2id.keys()))
        return word2id, id2word

    def get_word2id(self):
        word2id, _ = self.create_word2id_id2word()
        return word2id

    def get_id2word(self):
        _, id2word = self.create_word2id_id2word()
        return id2word