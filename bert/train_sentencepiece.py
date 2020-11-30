import os
import logging
import argparse

import sentencepiece as spm
import tensorflow as tf

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename="log/ar_training_run_1.log",
)


class TrainSentencePiece:
    def __init__(
        self,
        input_file,
        save_dir,
        model_prefix='tokenizer',
        vocab_size=64000,
        model_type="unigram",
        max_num_sentences=12800000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    ):
        """
        Initialize TrainSentencePiece Model.
        :param input_file: (str) of input file
        :param save_dir: (str) of directory to save vocabularies
        :param model_prefix: (str) of model prefix/name
        :param vocab_size: (int) of vocab size
        :param model_type: (str) of sentencepiece model type (e.g.: BPE, unigram)
        :param max_num_sentences: (int) of maximum size of sentences the trainer loads from dataset.
        """
        self.input_file = input_file
        self.model_prefix = model_prefix
        self.save_dir = save_dir
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.max_num_sentences = max_num_sentences
        self.special_tokens = special_tokens

        self.train_model()

    def train_model(self):
        """ Training SentencePiece Model."""
        # Check if save directory exists.
        if not os.path.exists(self.save_dir):
            raise ValueError("Could not find save dir : {}".format(self.save_dir))

        logging.info("Training SentencePiece model/vocab ...")
        combined_model_file = os.path.join(self.save_dir, self.model_prefix)
        # Check if a cached combined model file exists.
        if os.path.exists(combined_model_file + ".model"):
            logging.info("Found existing SentencePiece model/vocab ...")
        else:
            logging.info(
                "Could not find existing SentencePiece model/vocab. Training ..."
            )

            spm.SentencePieceTrainer.train(
                input=self.input_file,
                input_sentence_size=self.max_num_sentences,
                model_prefix=combined_model_file,
                control_symbols=self.special_tokens,
                # amount of characters covered by the model.
                character_coverage=1.0,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
                shuffle_input_sentence=True,
            )

            logging.info("SentencePiece model/vocab Trained Successfully.")

def main():
    parser = argparse.ArgumentParser(description='Train Sentencepiece Model')
    parser.add_argument(
        "-i",
        "--input_file",
        help="input data file path",
        required=True,
        type=str
    )

    parser.add_argument(
        "-s",
        "--save_folder",
        help="Path to save the model files",
        required=True,
        type=str
    )

    parser.add_argument(
        "-p",
        "--model_prefix",
        help="model prefix or model names",
        required=False,
        default="tokenizer",
        type=str
    )

    parser.add_argument(
        "-v",
        "--vocab_size",
        help="vocab size",
        required=False,
        default=64000,
        type=int
    )

    parser.add_argument(
        "-m",
        "--model_type",
        help="sentencepiece model type (e.g.: BPE, unigram)",
        required=False,
        default="unigram",
        type=str
    )

    parser.add_argument(
        "-z",
        "--max_num_sentences",
        help="the maximum size of sentences the trainer loads from dataset",
        required=False,
        default=12800000,
        type=int
    )

    parser.add_argument(
        "-k",
        "--special_tokens",
        help="sentencepiece model special tokens (e.g.: [<pad>,<s>])",
        required=False,
        default=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        type=lambda s: [str(item) for item in s.split(',')]
    )

    args = parser.parse_args()

    TrainSentencePiece(
        input_file=tf.gfile.Glob(args.input_file),
        save_dir=args.save_folder,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        max_num_sentences=args.max_num_sentences,
        special_tokens=args.special_tokens
    )


if __name__ == "__main__":
    main()