from util import dataset_interface as data
import pyarrow as pa
from util import tokenizer


def create_dataset(seq_length: int):
    path = input("Enter data directory path: ")

    reader = data.ParquetReader(
        path,
        data.ReaderConfig(output_mode="python"),
        columns="text"
    )

    _BATCH_SIZE = 4096

    t = tokenizer.BPETokenizer(reader.read_batched(_BATCH_SIZE), samples_count=len(reader))
    t.create_tokenized_dataset(
        reader.read_batched(_BATCH_SIZE),
        "../data/tokenized/train",
        len(reader),
        seq_length,
        datatype=pa.uint16()
    )


def probe_dataset():
    train_reader = data.ParquetReader(
        "../data/tokenized/train",
        config=data.ReaderConfig(
            output_mode="python",
        ),
        columns=0,
    )
    t = tokenizer.BPETokenizer(None, load_if_exists=True)

    print(f"Windows count: {len(train_reader)}")

    for i, item in zip(range(10), train_reader.read()):
        print("Tokens:")
        print(item)
        print("Decoded:")
        print(t.decode(item))


if __name__ == "__main__":
    probe_dataset()