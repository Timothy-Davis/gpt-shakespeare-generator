import torch

VOCAB = ""
VOCAB_SIZE = 0
TRAINING_SIZE = 0.9
BLOCK_SIZE = 8
BATCH_SIZE = 4
torch.manual_seed(1337)


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

    data = torch.tensor(encode(text), dtype=torch.long)
    train_data = data[:int(len(data)*TRAINING_SIZE)]
    test_data = data[int(len(data)*TRAINING_SIZE):]

    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else test_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x, y

    xb, yb = get_batch('train')
    for b in range(BATCH_SIZE):
        for t in range(BLOCK_SIZE):
            context = xb[b, :t+1]
            target = yb[b, t]
            print(f"When input is {context.tolist()} the target is: {target}")
