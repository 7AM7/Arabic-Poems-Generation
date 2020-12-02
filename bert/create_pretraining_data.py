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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""
import os
import collections
import random
import time
from multiprocessing import Pool

import tensorflow as tf

from bert.tokenization import SentencePieceTokenizer

flags = tf.flags
FLAGS = flags.FLAGS


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(
        self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next
    ):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (
            " ".join([str(x) for x in self.tokens])
        )
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions])
        )
        s += "masked_lm_labels: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_labels])
        )
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def print_example(instance, features):
    tf.logging.info("*** Example ***")
    tf.logging.info(
        "tokens: %s"
        % " ".join([str(x) for x in instance.tokens])
    )

    for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
            values = feature.int64_list.value
        elif feature.float_list.value:
            values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values]))
        )


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def transform(instance, tokenizer, max_seq_length, max_predictions_per_seq):
    """Transform instance to inputs for MLM and NSP."""
    input_ids = tokenizer.tokens_to_ids(instance.tokens)
    assert len(input_ids) <= max_seq_length
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    return features


def convert_to_tfexample(
    instances, tokenizer, max_seq_length, max_predictions_per_seq
):
    """Create TF example files from `TrainingInstance`s."""
    tf_examples = []
    for inst_index, instance in enumerate(instances):
        features = transform(instance, tokenizer, max_seq_length, max_predictions_per_seq)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        tf_examples.append(tf_example)

        if inst_index < 2:
            print_example(instance, features)

    return tf_examples


def tokenize_lines(x):
    """Worker function to tokenize lines based on the tokenizer, and perform vocabulary lookup."""
    lines, tokenizer = x
    results = []
    for line in lines:
        if not line:
            break
        line = line.strip()
        # Empty lines are used as document delimiters
        if not line:
            results.append([])
        else:
            tokens = tokenizer.tokenize(line)
            if tokens:
                results.append(tokens)
    return results


def write_to_files(features, output_file):
    """Create TF example files from `TrainingInstance`s."""
    total_written = len(features)
    writer = tf.python_io.TFRecordWriter(output_file)
    writer.write(features.SerializeToString())
    writer.close()

    tf.logging.info('Wrote %d total instances', total_written)


def create_training_instances(x):
    """Create `TrainingInstance`s from raw text."""
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.

    (input_files, tokenizer, max_seq_length,
     dupe_factor, short_seq_prob, masked_lm_prob,
     max_predictions_per_seq, whole_word_mask,
     nworker, worker_pool, output_file) = x
    time_start = time.time()
    if nworker > 1:
        assert worker_pool is not None

    all_documents = [[]]
    for input_file in input_files:
        with tf.io.gfile.GFile(input_file, "r") as reader:
            lines = reader.readlines()
            num_lines = len(lines)
            num_lines_per_worker = (num_lines + nworker - 1) // nworker
            process_args = []

            # tokenize in parallel
            for worker_idx in range(nworker):
                start = worker_idx * num_lines_per_worker
                end = min((worker_idx + 1) * num_lines_per_worker, num_lines)
                process_args.append((lines[start:end], tokenizer))

            if worker_pool:
                tokenized_results = worker_pool.map(
                    tokenize_lines, process_args)
            else:
                tokenized_results = [tokenize_lines(process_args[0])]

            for tokenized_result in tokenized_results:
                for line in tokenized_result:
                    if not line:
                        if all_documents[-1]:
                            all_documents.append([])
                    else:
                        all_documents[-1].append(line)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]

    # generate training instances
    vocab_words = list(tokenizer.word2id.keys())
    instances = []
    if worker_pool:
        process_args = []
        for document_index in range(len(all_documents)):
            process_args.append((
                    all_documents, document_index,
                    max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq,
                    whole_word_mask, vocab_words))
        for _ in range(dupe_factor):
            instances_results = worker_pool.map(
                create_instances_from_document, process_args)
            for instances_result in instances_results:
                instances.extend(instances_result)

        tfexample_instances = worker_pool.apply(
            convert_to_tfexample, (instances, tokenizer,
                                   max_seq_length, max_predictions_per_seq))
    else:
        for _ in range(dupe_factor):
            for document_index in range(len(all_documents)):
                instances.extend(
                    create_instances_from_document(
                        (all_documents, document_index, max_seq_length, short_seq_prob,
                         masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_words)))
        tfexample_instances = convert_to_tfexample(instances, tokenizer,
                                                   max_seq_length, max_predictions_per_seq)

    features = tfexample_instances
    # write output to files. Used when pre-generating files
    if output_file:
        tf.logging.info('*** Writing to output file %s ***', output_file)
        write_to_files(features, output_file)
        features = None

    time_end = time.time()
    tf.logging.info('Process %d files took %.1f s',
                  len(input_files), time_end - time_start)
    return features


def create_instances_from_document(x):
    """Creates `TrainingInstance`s for a single document."""
    (all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_words) = x
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or random.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = random.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                    masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob,
                    max_predictions_per_seq,
                    whole_word_mask,
                    vocab_words
                )
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                )
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(
    tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_words
):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with '_'. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (
            whole_word_mask
            and len(cand_indexes) >= 1
            and token.startswith("_")
        ):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    time_start = time.time()
    random.seed(FLAGS.random_seed)

    # create output dir
    output_dir = os.path.expanduser(FLAGS.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer = SentencePieceTokenizer(
        model_file=FLAGS.sentencepiece_file,
        lowercase=FLAGS.do_lower_case,
    )

    input_files = []
    for input_pattern in FLAGS.input_file.split(','):
        input_files.extend(tf.gfile.Glob(input_pattern))

    # Print files
    for input_file in input_files:
        tf.logging.info('\t%s', input_file)

    num_inputs = len(input_files)
    num_outputs = min(FLAGS.num_outputs, len(input_files))
    tf.logging.info('*** Reading from %d input files ***', num_inputs)

    # calculate the number of splits
    file_splits = []
    split_size = (num_inputs + num_outputs - 1) // num_outputs
    for i in range(num_outputs):
        split_start = i * split_size
        split_end = min(num_inputs, (i + 1) * split_size)
        file_splits.append(input_files[split_start:split_end])

    # prepare workload
    count = 0
    process_args = []
    for i, file_split in enumerate(file_splits):
        output_file = os.path.join(
            output_dir, 'part-{}.tfrecord'.format(str(i).zfill(3)))
        count += len(file_split)
        process_args.append((file_split, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
                             FLAGS.short_seq_prob, FLAGS.masked_lm_prob,
                             FLAGS.max_predictions_per_seq, FLAGS.do_whole_word_mask,
                             1, None, output_file))
    # sanity check
    assert count == len(input_files)

    # dispatch to workers
    nworker = FLAGS.num_workers
    if nworker > 1:
        pool = Pool(nworker)
        pool.map(create_training_instances, process_args)
    else:
        for process_arg in process_args:
            create_training_instances(process_arg)

    time_end = time.time()
    tf.logging.info('Time cost=%.1f', time_end - time_start)


if __name__ == "__main__":
    flags.DEFINE_string(
        "input_file", None, 'Input files, separated by comma. For example, "~/data/*.txt"'
    )

    flags.DEFINE_string(
        "output_dir", None, "Output TF records directory"
    )

    flags.DEFINE_string("sentencepiece_file", None, "The sentecepiece mode path")

    flags.DEFINE_bool(
        "do_lower_case",
        True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.",
    )

    flags.DEFINE_bool(
        "do_whole_word_mask",
        False,
        "Whether to use whole word masking rather than per-WordPiece masking.",
    )

    flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

    flags.DEFINE_integer(
        "max_predictions_per_seq",
        20,
        "Maximum number of masked LM predictions per sequence.",
    )

    flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

    flags.DEFINE_integer(
        "num_workers",
        8,
        "Number of workers for parallel processing, where each generates an output file.")

    flags.DEFINE_integer(
        "num_outputs",
        1,
        "Number of workers for parallel processing, where each generates an output file.")

    flags.DEFINE_integer(
        "dupe_factor",
        10,
        "Number of times to duplicate the input data (with different masks).",
    )

    flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

    flags.DEFINE_float(
        "short_seq_prob",
        0.1,
        "Probability of creating sequences which are shorter than the " "maximum length.",
    )
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("sentencepiece_file")
    tf.app.run()
