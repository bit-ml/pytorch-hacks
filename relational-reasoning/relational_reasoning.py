# Tudor Berariu @ Bitdefender, 2017

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RelationalReasoningModuleNoTranspose(nn.Module):

    def __init__(self, g, depth, embedding_size=None, full_pairs=False):
        super(RelationalReasoningModuleNoTranspose, self).__init__()
        self.d = depth
        self.e = embedding_size
        self.g = g
        self.full_pairs = full_pairs

        self._router = None  # Lazy creation (not to worry about cuda or sizes)
        self._router_size = None

    def forward(self, x, emb=None):
        B, D, H, W = x.size()
        assert D == self.d

        if emb is not None:
            _B, E = emb.size()
            assert B == _B and E == self.e

        full_pairs = self.full_pairs

        N = H * W
        P = N * N
        R = N * (N - 1) // 2

        x = x.view(B, D * N)

        pairs_no = P if full_pairs else R
        router_size = torch.Size([pairs_no * D * 2, D * N])

        if self._router is None or self._router_size != router_size:
            router = x.data.new().resize_(router_size).fill_(0)
            offset = 0
            for i in range(N):
                _s = 0 if full_pairs else (i + 1)
                for j in range(_s, N):
                    for k in range(D):
                        router[offset * (2 * D) + k, k * N + i] = 1
                        router[offset * (2 * D) + D + k, k * N + j] = 1
                    offset += 1
            self._router = Variable(router)
            self._router_size = router_size

        router = self._router

        x_pairs = F.linear(input=x, weight=router).view(B * pairs_no, -1)

        if emb is not None:
            emb = emb.unsqueeze(1).expand(B, pairs_no, E) \
                 .contiguous().view(B * pairs_no, E)
            g_input = torch.cat([x_pairs, emb], 1)
        else:
            g_input = x_pairs
        return self.g(g_input).view(B, pairs_no, -1).sum(1).squeeze(1)


class RelationalReasoningModule(nn.Module):

    def __init__(self, g, depth, embedding_size=None, full_pairs=False):
        super(RelationalReasoningModule, self).__init__()
        self.d = depth
        self.e = embedding_size
        self.g = g
        self.full_pairs = full_pairs

        self._cached_idxs = None
        self._cached_size = None

    def forward(self, x, emb=None):
        B, D, H, W = x.size()
        assert D == self.d

        if emb is not None:
            _B, E = emb.size()
            assert B == _B and E == self.e

        N = H * W
        P = N * N
        R = N * (N - 1) // 2
        full_pairs = self.full_pairs
        pairs_no = P if full_pairs else R

        x = x.view(B, D, N).transpose(1, 2)

        if full_pairs:
            a = x.unsqueeze(2).expand(B, N, N, D).contiguous().view(B * P, D)
            b = x.unsqueeze(1).expand(B, N, N, D).contiguous().view(B * P, D)
        else:
            if self._cached_idxs is None or self._cached_size != N:
                _mask = x.data.new().long().resize_(N, N).fill_(0)
                for i in range(N - 1):
                    _mask[i, (i + 1):].fill_(1)
                _mask = Variable(_mask.view(-1).nonzero().squeeze())
                self._cached_size = N
                self._cached_idxs = _mask
            idxs = self._cached_idxs

            prod_sz = torch.Size([B * R, D])
            a = x.unsqueeze(2).expand(B, N, N, D).contiguous().view(B, P, D)\
                 .index_select(1, idxs).view(prod_sz)
            b = x.unsqueeze(1).expand(B, N, N, D).contiguous().view(B, P, D)\
                 .index_select(1, idxs).view(prod_sz)

        x_pairs = torch.cat([a, b], 1)

        if emb is not None:
            emb = emb.unsqueeze(1).expand(B, pairs_no, E)\
                 .contiguous().view(B * pairs_no, E)
            g_input = torch.cat([x_pairs, emb], 1)
        else:
            g_input = x_pairs
        return self.g(g_input).view(B, pairs_no, -1).sum(1).squeeze(1)


class RelationalReasoningModuleWithLoop(nn.Module):

    def __init__(self, g, depth, embedding_size=None, full_pairs=False):
        super(RelationalReasoningModuleWithLoop, self).__init__()
        self.d = depth
        self.e = embedding_size
        self.g = g
        self.full_pairs = full_pairs

    def forward(self, x, emb=None):
        B, D, H, W = x.size()
        assert D == self.d

        if emb is not None:
            _B, E = emb.size()
            assert B == _B and E == self.e

        N = H * W
        P = N * N
        R = N * (N - 1) // 2
        full_pairs = self.full_pairs
        pairs_no = P if full_pairs else R

        x = x.view(B, D, N).transpose(1, 2)
        pairs = []

        if full_pairs:
            for i in range(N):
                ith = torch.cat([x.narrow(1, i, 1).expand(B, N, D), x], 2)
                ith = ith.view(B, N, 2 * D)
                pairs.append(ith)
        else:
            for i in range(N - 1):
                ith = torch.cat([x.narrow(1, i, 1).expand(B, N - 1 - i, D),
                                 x.narrow(1, i + 1, N - 1 - i)],
                                2).view(B, (N - 1 - i), 2 * D)
                pairs.append(ith)

        x_pairs = torch.cat(pairs, 1).view(B * pairs_no, 2 * D)
        if emb is not None:
            emb = emb.unsqueeze(1).expand(B, pairs_no, E)\
                 .contiguous().view(B * pairs_no, E)
            g_input = torch.cat([x_pairs, emb], 1)
        else:
            g_input = x_pairs
        return self.g(g_input).view(B, pairs_no, -1).sum(1).squeeze(1)


def compare(args):
    B = args.batch_size
    D = args.channels_no
    H = args.height
    W = args.width
    O = args.output_size
    full_pairs = args.full_pairs
    E = args.embedding_size

    x = Variable(torch.randn(B, D, H, W))
    if E:
        e = Variable(torch.randn(B, E))
        g = nn.Sequential(nn.Linear(D * 2 + E, 256), nn.Linear(256, O))
    else:
        g = nn.Sequential(nn.Linear(D * 2, 256), nn.ReLU(), nn.Linear(256, O))

    if args.use_cuda:
        x = x.cuda()
        if args.use_embeddings:
            e = e.cuda()
        g.cuda()

    for M in [RelationalReasoningModule,
              RelationalReasoningModuleWithLoop,
              RelationalReasoningModuleNoTranspose]:
        r = M(g, D, embedding_size=E, full_pairs=full_pairs)
        if args.use_cuda:
            r.cuda()
        y = r(x, e) if E else r(x)
        print("Output:")
        print(y)
        del r, y


def main():
    from argparse import Namespace
    args = Namespace()

    args.batch_size = 128
    args.channels_no = 16
    args.height = 8
    args.width = 8
    args.embedding_size = None  # None or some positive integer
    args.full_pairs = False     # True or False
    args.output_size = 8
    args.use_cuda = False

    compare(args)

if __name__ == "__main__":
    main()
