VOCAB = ""
VOCAB_SIZE = 0


def string_to_integer():
    return {ch: i for i, ch in enumerate(VOCAB)}


def integer_to_string():
    return {i: ch for i, ch in enumerate(VOCAB)}


def encode(chars):
    return [string_to_integer()[c] for c in chars]


def decode(chars):
    return ''.join([integer_to_string()[i] for i in chars])


if __name__ == "__main__":
    with open('../tiny-shakespeare.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    VOCAB = sorted(list(set(text)))
    VOCAB_SIZE = len(VOCAB)
