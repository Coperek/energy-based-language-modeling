from util import tokenizer
from util import dataset_interface as data

if __name__ == '__main__':
    t = tokenizer.BPETokenizer()

    #encoded = t.encode(input("Enter string:"))
    #print(encoded)
    #print(t.decode(encoded))

    for word in data.ParquetReader("test.parquet", data.ReaderConfig(window_size=3, window_overlap=0, drop_last=False, shuffle_mode="none")).read_batched(2, drop_last=False):
        for i in word:
            print(i.shape)
            print(i)
            print()
