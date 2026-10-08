from util import tokenizer

if __name__ == '__main__':
    t = tokenizer.BPETokenizer()

    encoded = t.encode(input("Enter string:"))
    print(encoded)
    print(t.decode(encoded))
