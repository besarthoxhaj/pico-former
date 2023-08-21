import sentencepiece as spm


class Tokenizer:
  def __init__(self):
    f = open('./names.txt', 'r')
    names = f.read().splitlines()
    self.vocab = ['<pad>', '<eos>', '<sos>'] + sorted(set(''.join(names)))
    self.stoi = {c:i for i, c in enumerate(self.vocab)}
    self.itos = {i:c for i, c in enumerate(self.vocab)}
    self.vocab_size = len(self.vocab)
    f.close()

  def encode(self, name):
    return [self.stoi[c] for c in name]

  def decode(self, tokens):
    return ''.join([self.itos[t] for t in tokens if self.itos[t] not in ('<sos>', '<eos>', '<pad>')])


class TinyTokenizer:
  def __init__(self):
    print('TinyTokenizer.__init__')

  def encode(self, txt):
    return txt

  def decode(self, tks):
    return tks


if __name__ == '__main__':
  tknz = Tokenizer()
  print('tknz.vocab', tknz.vocab)
  print('tknz.stoi', tknz.stoi)
  print('tknz.itos', tknz.itos)