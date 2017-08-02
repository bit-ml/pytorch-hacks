# Tudor Berariu @ Bitdefender, 2017

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RelationalReasoningModuleN(nn.Module):

    def __init__(self, g, n, depth, embedding_size=None):
        super(RelationalReasoningModuleN, self).__init__()
        self.d = depth
        self.n = n
        self.e = embedding_size
        self.g = g

        self._cached_idxs = None
        self._cached_size = None

    @staticmethod
    def get_indices(n, px_no):
        def _get_indices(_n, _start, _px_no):
            if _n == 1:
                return range(_start, _px_no)
            else:
                result = []
                for j in range(_start, px_no - _n + 1):
                    result.extend(map(lambda x: x + j * _px_no ** (_n - 1),
                                      _get_indices(_n - 1, j + 1, _px_no)))
                return result

        return list(_get_indices(n, 0, px_no))

    def forward(self, x, emb=None):
        B, D, H, W = x.size()
        assert D == self.d

        if emb is not None:
            _B, E = emb.size()
            assert B == _B and E == self.e

        n = self.n

        px_no = H * W
        pairs_no = px_no ** n
        real_pairs_no = int(np.prod(range(max(1, px_no - n + 1), px_no + 1)) /
                            np.prod(range(1, n + 1)))
        x = x.view(B, D, px_no).transpose(1, 2)

        if self._cached_idxs is None or self._cached_size != (n, px_no):
            idxs = RelationalReasoningModuleN.get_indices(n, px_no)
            idxs = torch.LongTensor(idxs)
            if x.is_cuda:
                idxs = idxs.cuda()
            idxs = Variable(idxs)
            self._cached_size = (n, px_no)
            self._cached_idxs = idxs
        idxs = self._cached_idxs
        assert real_pairs_no == len(idxs)

        full_sz = torch.Size([B] + [px_no] * n + [D])
        prod_sz = torch.Size([B, pairs_no, D])
        real_prod_sz = torch.Size([B * real_pairs_no, D])
        x_js = []
        for j in range(n):
            x_j = x
            for d in [2] * (n - j - 1) + [1] * j:
                x_j = x_j.unsqueeze(d)
            x_j = x_j.expand(full_sz).contiguous().view(prod_sz)\
                     .index_select(1, idxs).view(real_prod_sz)
            x_js.append(x_j)

        x_pairs = torch.cat(x_js, 1)

        if emb is not None:
            emb = emb.unsqueeze(1).expand(B, real_pairs_no, E)\
                 .contiguous().view(B * real_pairs_no, E)

            g_input = torch.cat([x_pairs, emb], 1)
        else:
            g_input = x_pairs

        return self.g(g_input).view(B, real_pairs_no, -1).sum(1).squeeze(1)


def main():
    B = 32   # batch size
    D = 16    # volume depth (# of feature maps)
    H = 7     # volume height
    W = 7     # volume width
    E = 128   # embedding size
    O = 32
    use_cuda = True

    x = Variable(torch.randn(B, D, H, W))

    e = Variable(torch.randn(B, E))
    g_2 = nn.Sequential(nn.Linear(D * 2, 256), nn.ReLU(),  nn.Linear(256, O))
    g_e_2 = nn.Sequential(nn.Linear(D * 2 + E, 256), nn.Linear(256, O))
    g_3 = nn.Sequential(nn.Linear(D * 3, 256), nn.ReLU(), nn.Linear(256, O))
    g_e_3 = nn.Sequential(nn.Linear(D * 3 + E, 256), nn.Linear(256, O))

    if use_cuda:
        x = x.cuda()
        e = e.cuda()
        g_2.cuda()
        g_e_2.cuda()
        g_3.cuda()
        g_e_3.cuda()

    # -- Without embedding (n = 2)
    print("Without embedding, n = 2:")
    r = RelationalReasoningModuleN(g_2, 2, D)
    y = r(x)
    print(y)
    del r, y

    # -- Without embedding (n = 3)
    print("Without embedding, n = 3:")
    r = RelationalReasoningModuleN(g_3, 3, D)
    y = r(x)
    print(y)
    del r, y

    # -- With embedding
    print("With embedding, n = 2:")
    r = RelationalReasoningModuleN(g_e_2, 2, D, E)
    y = r(x, e)
    print(y)
    del r, y

    # -- With embedding
    print("With embedding, with loops:")
    r = RelationalReasoningModuleN(g_e_3, 3, D, E)
    y = r(x, e)
    print(y)
    del r, y

if __name__ == "__main__":
    main()
