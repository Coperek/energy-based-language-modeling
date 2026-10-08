# TODO: dataset interface changed, so this must change as well

from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

TOKENIZER_PATH = str(Path(__file__).parent.parent.joinpath("tokenizer.json"))


class BPETokenizer:
    def __init__(self, unk_token="<unk>", vocab_size=32000,
                 train_samples=1_000_000, train_batch_size=128,):
        self.special_tokens = {}

        if Path(TOKENIZER_PATH).exists():
            self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        else:
            self.tokenizer = Tokenizer(
                BPE(fuse_unk=True, unk_token=unk_token)
            )

            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()

            print("Tokenizer creation in progress...")
            data_path = input("[Tokenizer training] Enter the path to the data directory or file:")

            if not Path(data_path).exists():
                raise Exception(f"Path {data_path} does not exist!")

            data_generator = dataset_interface.FileReader.create(data_path).read_batches(
                batch_size=train_batch_size,
                max_batches=train_samples//train_batch_size
            )

            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=[unk_token]
            )

            print("Please wait... (this can take a while)")
            self.tokenizer.train_from_iterator(data_generator, trainer)
            self.tokenizer.save(TOKENIZER_PATH)

            print(f"Tokenizer training finished. Vocab size: {self.tokenizer.get_vocab_size()}")

        if unk_token:
            self.register_special_token(unk_token)

    def encode(self, text):
        if hasattr(text, "__iter__") and not isinstance(text, str):
            return [encoding.ids for encoding in self.tokenizer.encode_batch(text, add_special_tokens=False)]
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        reversed_vocab = {v: k for k, v in self.tokenizer.get_vocab(with_added_tokens=True).items()}
        space_char = reversed_vocab[self.tokenizer.encode(" ").ids[0]]
        reversed_special_tokens = {v: k for k, v in self.special_tokens.items()}
        return [
            (
                reversed_special_tokens[_id] if (_id in self.special_tokens.values())
                else reversed_vocab[_id].replace(space_char, " ")
            )
            for _id in ids
        ]

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True) + len(self.special_tokens)
        # special tokens are completely outside the vocabulary

    def register_special_token(self, token_name: str) -> int:
        self.special_tokens.update({token_name: self.get_vocab_size()})
        return self.special_tokens[token_name]

# TODO: special tokens: <|im_start|>system <|im_start|>user <|im_start|>assistant