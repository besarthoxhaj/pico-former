import torch
import t5
import dataset
import tokenizer


torch.manual_seed(42)
myT5 = t5.T5()
myT5.num_params()


tk = tokenizer.Tokenizer()
ds = dataset.Dataset()
opt = torch.optim.SGD(myT5.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()


x_src = torch.randint(0, t5.VOCB, (2, 12))
x_tgt = torch.randint(0, t5.VOCB, (2, 12))


for idx, epoch in enumerate(range(101)):

  y = x_tgt.clone() # TODO: shift right
  p = myT5(x_src, x_tgt)

  p = p.reshape(-1, p.size(-1))
  y = y.reshape(-1)

  l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)

  if idx % 100 == 0: print(f"Loss: {l.item():.4f}")
  l.backward()
  opt.step()
  opt.zero_grad()