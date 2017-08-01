# Tudor Berariu @ Bitdefender, 2017

import torch
import torch.nn as nn
from torch.autograd import Variable


class RelationalReasoningModule(nn.Module):

    def __init__(self, g, depth, embedding_size=None):
        super(RelationalReasoningModule, self).__init__()
        self.d  = depth
        self.e = embedding_size
        self.g = g

        self._cached_idxs = None
        self._cached_size = None


    def forward(self, x, emb=None):
        B, D, H, W = x.size()
        assert D == self.d

        if emb is not None:
            _B, E = emb.size()
            assert B == _B and E == self.e

        px_no = H * W
        pairs_no = px_no * px_no
        real_pairs_no = px_no * (px_no - 1) // 2

        x = x.view(B, D, px_no).transpose(1, 2)
        prod_sz = torch.Size([B * real_pairs_no, D])

        if self._cached_idxs is None or self._cached_size != px_no:
            _mask = x.data.new().long().resize_(px_no, px_no).fill_(0)
            for i in range(px_no-1):
                _mask[i,(i+1):].fill_(1)
            _mask = Variable(_mask.view(-1).nonzero().squeeze())
            self._cached_size = px_no
            self._cached_idxs = _mask
        idxs = self._cached_idxs

        a = x.unsqueeze(2).expand(B, px_no, px_no, D).contiguous() \
             .view(B, pairs_no, D) \
             .index_select(1, idxs).view(prod_sz)

        b = x.unsqueeze(1).expand(B, px_no, px_no, D).contiguous() \
             .view(B, pairs_no, D) \
             .index_select(1, idxs).view(prod_sz)

        x_pairs = torch.cat([a, b], 1)

        if emb is not None:
            emb = emb.unsqueeze(1).expand(B, real_pairs_no, E)\
                 .contiguous().view(B * real_pairs_no, E)

            g_input = torch.cat([x_pairs, emb], 1)
        else:
            g_input = x_pairs

        return self.g(g_input).view(B, real_pairs_no, -1).sum(1).squeeze(1)


class RelationalReasoningModuleWithLoop(nn.Module):

    def __init__(self, g, depth, embedding_size=None):
        super(RelationalReasoningModuleWithLoop, self).__init__()
        self.d  = depth
        self.e = embedding_size
        self.g = g


    def forward(self, x, emb=None):
        B, D, H, W = x.size()
        assert D == self.d

        if emb is not None:
            _B, E = emb.size()
            assert B == _B and E == self.e

        px_no = H * W
        real_pairs_no = px_no * (px_no - 1) // 2
        x = x.view(B, D, px_no).transpose(1, 2)
        pairs = []
        for i in range(px_no - 1):
            ith = torch.cat([x.narrow(1, i, 1).expand(B, px_no - 1 - i, D),
                             x.narrow(1, i + 1, px_no - 1 - i)],
                            2).view(B, (px_no - 1 - i), 2 * D)
            pairs.append(ith)

        x_pairs = torch.cat(pairs, 1).view(B * real_pairs_no, 2 * D)
        if emb is not None:
            emb = emb.unsqueeze(1).expand(B, real_pairs_no, E)\
                 .contiguous().view(B * real_pairs_no, E)
            g_input = torch.cat([x_pairs, emb], 1)
        else:
            g_input = x_pairs

        return self.g(g_input).view(B, real_pairs_no, -1).sum(1).squeeze(1)


def main():
    B = 128   # batch size
    D = 16    # volume depth (# of feature maps)
    H = 8     # volume height
    W = 8     # volume width
    E = 128   # embedding size
    O = 32
    use_cuda = True

    x = Variable(torch.randn(B, D, H, W))

    e = Variable(torch.randn(B, E))
    g = nn.Sequential(nn.Linear(D * 2, 256), nn.ReLU(),  nn.Linear(256, O))
    g_e = nn.Sequential(nn.Linear(D * 2 + E, 256), nn.Linear(256, O))

    if use_cuda:
        x = x.cuda()
        e = e.cuda()
        g.cuda()
        g_e.cuda()

    # -- Without embedding
    print("Without embedding, no lopps:")
    r = RelationalReasoningModule(g, D)
    y = r(x)
    print(y)
    del r, y

    # -- Without embedding
    print("Without embedding, with loops:")
    r = RelationalReasoningModuleWithLoop(g, D)
    y = r(x)
    print(y)
    del r, y

    # -- With embedding
    print("With embedding, no lopps:")
    r = RelationalReasoningModule(g_e, D, E)
    y = r(x, e)
    print(y)
    del r, y

    # -- With embedding
    print("With embedding, with loops:")
    r = RelationalReasoningModuleWithLoop(g_e, D, E)
    y = r(x, e)
    print(y)
    del r, y

if __name__ == "__main__":
    main()
