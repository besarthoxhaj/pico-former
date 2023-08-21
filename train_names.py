import torch
import gpt
import dataset
import tokenizer


torch.manual_seed(42)
myGPT = gpt.GPT()
myGPT.num_params()


tk = tokenizer.Tokenizer()
ds = dataset.Dataset()
dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=ds.collate_fn)
opt = torch.optim.SGD(myGPT.parameters(), lr=0.01)


for epoch in range(10):

  sos = torch.tensor([[2]])
  name = myGPT.generate(sos)
  name = name[0].tolist()
  print("Name:", tk.decode(name))

  for idx, batch in enumerate(dl):

    # print("batch['input']", batch['input'].shape, batch['input'])
    # print("batch['label']", batch['label'].shape, batch['label'])
    # print("batch['masks']", batch['masks'].shape, batch['masks'])

    x = batch['input']
    y = batch['label']
    p = myGPT(x)

    # Cross Entropy Loss expects a 2D tensor of size (N, C)
    # and a 1D tensor of size (N) as input, where N is the
    # batch size and C is the number of classes. E.g.
    #
    # p -> (8, 16, 29) -> (8*16, 29) -> (128, 29)
    # y -> (8, 16)     -> (8*16)     -> (128)

    p = p.view(-1, p.size(-1))
    y = y.view(-1)

    l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)

    # Initially, a randomly initialized model will predict
    # on average, a uniform distribution over the vocabulary.
    # Thus the probability of a token is 1/vocab_size ~ 1/29.
    # Therefore the loss will be -ln(1/29) ~ 3.37 for each
    # token. The loss for each token is then averaged over the
    # sequence length i.e.
    #
    # loss = -ln(1/29) ~ 3.37

    if idx % 1000 == 0: print("Loss:", l.item())
    l.backward()
    opt.step()
    opt.zero_grad()