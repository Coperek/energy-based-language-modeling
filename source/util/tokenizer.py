from pathlib import Path
from tqdm import tqdm

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

import pyarrow as pa
import pyarrow.parquet as pq


TOKENIZER_DEFAULT_PATH = str(Path(__file__).parent.parent.joinpath("tokenizer.json"))


class BPETokenizer:
    def __init__(
            self,
            training_data_batch_iterator,
            samples_count=None,
            vocab_size=32000,
            load_if_exists=True,
            eos_token="<eos>",
            unk_token="<unk>",
            path=TOKENIZER_DEFAULT_PATH,
    ):

        self.special_tokens = {}

        if Path(path).exists() and load_if_exists:
            self.tokenizer = Tokenizer.from_file(path)
        else:
            self.tokenizer = Tokenizer(
                BPE(fuse_unk=True, unk_token=unk_token)
            )

            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()

            print("Tokenizer training in progress...")

            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=[unk_token]
            )

            print("Please wait... (this can take a while)")

            self.tokenizer.train_from_iterator(training_data_batch_iterator, trainer, length=samples_count)
            self.tokenizer.save(path)

            print(f"Tokenizer training finished. Vocab size: {self.tokenizer.get_vocab_size()}")

        self.eos_token = eos_token
        if not self.eos_token:
            raise ValueError("EOS token must be provided")
        self.register_special_token(eos_token)

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

    def encode_stream(self, iterator):
        """
        Encodes data provided by iterator and yields tokens ids one by one,
        automatically injects EoS tokens.
        """
        for text in iterator:
            if hasattr(text, "__iter__") and not isinstance(text, str):
                # batch of texts
                for ids in self.encode(text):
                    for _ in ids:
                        yield _
                    yield self.special_tokens[self.eos_token]
            else:
                for _ in self.encode(text):
                    yield _
                yield self.special_tokens[self.eos_token]

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True) + len(self.special_tokens)

    def register_special_token(self, token_name: str) -> int:
        # special tokens are completely outside the regular vocabulary
        self.special_tokens.update({token_name: self.get_vocab_size()})
        return self.special_tokens[token_name]

    def create_tokenized_dataset(
            self,
            data_iterator,
            output_dir: str,
            total_data_samples: int,
            batch_size=4096,
            max_tokens_per_file: int = 67108864,
            max_files: int = 64,
            datatype: pa.DataType = pa.int32(),
    ):

        out_dir_obj = Path(output_dir)

        if not out_dir_obj.exists():
            raise ValueError("Output directory does not exist")
        if not out_dir_obj.is_dir():
            raise ValueError("Output path must be a directory")

        column_name = "tokens"
        schema = pa.schema([
            (column_name, datatype)
        ])

        finished = False
        with (tqdm(total=total_data_samples, unit="doc") as pbar):

            file_i = 0
            while file_i < max_files:

                path = output_dir
                path += f"tokenized_dataset_{file_i}.parquet" \
                    if path.endswith("/") \
                    else f"/tokenized_dataset_{file_i}.parquet"

                tokens_written = 0
                with pq.ParquetWriter(path, schema=schema) as writer:
                    while tokens_written < max_tokens_per_file:

                        texts = []
                        for i in range(batch_size):
                            try:
                                texts.append(next(data_iterator))
                            except StopIteration:
                                finished = True
                                break

                        batch_tokens = self.encode(texts)
                        unified_tokens = []
                        for tokens in batch_tokens:
                            unified_tokens.extend(tokens)
                            unified_tokens.append(self.special_tokens[self.eos_token])

                        table = pa.table({column_name: unified_tokens}, schema=schema)
                        writer.write(table)

                        tokens_written += len(unified_tokens)
                        pbar.update(len(batch_tokens))

                        if finished:
                            return

                file_i += 1
