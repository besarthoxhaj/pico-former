import torch
import t5
import dataset
import tokenizer


torch.manual_seed(42)
myT5 = t5.T5()
myT5.num_params()


tk = (tokenizer.LangTokenizer()).load()
ds = dataset.LangDataset()
dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ds.collate_fn)
opt = torch.optim.SGD(myT5.parameters(), lr=0.01)


for epoch in range(5):

  org = "hello world"
  src = torch.tensor([tk.encode(org)])
  trs = myT5.translate(src)
  print(f"{org} - {tk.decode(trs.tolist()[0])}")

  for idx, batch in enumerate(dl):

    c = batch['contx']
    x = batch['input']
    y = batch['label']
    p = myT5(c, x)

    p = p.view(-1, p.size(-1))
    y = y.view(-1)
    l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)
    if idx % 1000 == 0: print(f"Loss: {l.item():.4f}")
    l.backward()
    opt.step()
    opt.zero_grad()