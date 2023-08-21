import torch
import tokenizer


class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    f = open('./names.txt', 'r')
    self.names = f.read().split('\n')
    self.tknz  = tokenizer.Tokenizer()
    f.close()

  def __len__(self):
    return len(self.names)

  def __getitem__(self, idx):
    name  = self.names[idx]
    input = [self.tknz.stoi['<sos>']] + self.tknz.encode(name)
    label = (input[1:]) + [self.tknz.stoi['<eos>']]
    masks = [1] * len(input)
    return {
      'plain': name,
      'input': torch.tensor(input),
      'label': torch.tensor(label),
      'masks': torch.tensor(masks),
    }

  # The input batch is a list of tensors with different
  # lengths. We use pad_sequence to pad the tensors with
  # 0s so that they all have the same length.
  def collate_fn(self, batch):
    input_pad = torch.nn.utils.rnn.pad_sequence([item['input'] for item in batch], batch_first=True, padding_value=0)
    label_pad = torch.nn.utils.rnn.pad_sequence([item['label'] for item in batch], batch_first=True, padding_value=0)
    masks_pad = torch.nn.utils.rnn.pad_sequence([item['masks'] for item in batch], batch_first=True, padding_value=0)

    return {
      'plain': [item['plain'] for item in batch],
      'input': input_pad,
      'label': label_pad,
      'masks': masks_pad,
    }


if __name__ == '__main__':
  ds = Dataset()
  emma = ds[0]
  print('emma', emma)

  # 'plain': 'emma'
  # 'input': tensor([ 7, 15, 15,  3])
  # 'label': tensor([15, 15,  3,  1])
  # 'masks': tensor([ 1,  1,  1,  1])