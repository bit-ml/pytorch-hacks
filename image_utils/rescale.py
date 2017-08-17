# Tudor Berariu @ Bitdefender, 2017

import torch


class BilinearResizer(object):

    def __init__(self, orig_size=None, new_size=None):
        if orig_size is not None and new_size is not None:
            self._compute_idxs(orig_size, new_size)
        else:
            self.orig_size = None
            self.new_size = None

    def _compute_idxs(self, orig_size, new_size):
        assert len(orig_size) == 3 and len(new_size) == 3 and \
               orig_size[0] == new_size[0]

        d, o_h, o_w = orig_size
        _, n_h, n_w = new_size

        r_idx = torch.linspace(0, o_h - 1, n_h).cuda()
        i_up = torch.floor(r_idx)
        i_down = i_up + 1.
        f_down = (r_idx - i_up).view(1, n_h, 1)
        f_up = (i_down - r_idx).view(1, n_h, 1)

        self.i_up = i_up.long().cuda()
        self.i_down = i_down.clamp(0, o_h - 1).long().cuda()
        self.f_down = f_down.cuda()
        self.f_up = f_up.cuda()

        r_idx = torch.linspace(0, o_w - 1, n_w).cuda()
        i_left = torch.floor(r_idx)
        i_right = i_left + 1.
        f_left = (r_idx - i_right).view(1, 1, n_w)
        f_right = (i_left - r_idx).view(1, 1, n_w)

        self.i_left = i_left.long().cuda()
        self.i_right = i_right.clamp(0, o_w - 1).long().cuda()
        self.f_left = f_left.cuda()
        self.f_right = f_right.cuda()

        self.orig_size = orig_size
        self.new_size = new_size

    def __call__(self, orig_img, new_size):
        orig_size = orig_img.size()
        if not(self.orig_size == orig_size and self.new_size == new_size):
            self._compute_idxs(orig_size, new_size)

        d, o_h, o_w = orig_size
        _, n_h, n_w = new_size

        z = orig_img.index_select(1, self.i_up) *\
            self.f_up.expand(d, n_h, o_w) +\
            orig_img.index_select(1, self.i_down) *\
            self.f_down.expand(d, n_h, o_w)

        y = z.index_select(2, self.i_left) * self.f_left.expand(d, n_h, n_w) +\
            z.index_select(2, self.i_right) * self.f_right.expand(d, n_h, n_w)

        return y


class NNResizer(object):

    def __init__(self, orig_size=None, new_size=None):
        if orig_size is not None and new_size is not None:
            self._compute_idxs(orig_size, new_size)
        else:
            self.orig_size = None
            self.new_size = None

    def _compute_idxs(self, orig_size, new_size):
        assert len(orig_size) == 3 and len(new_size) == 3 and \
               orig_size[0] == new_size[0]

        d, o_h, o_w = orig_size
        _, n_h, n_w = new_size

        self.i_v = torch.linspace(0, o_h - 1, n_h).cuda().round().long()
        self.i_h = torch.linspace(0, o_w - 1, n_w).cuda().round().long()

        self.orig_size = orig_size
        self.new_size = new_size

    def __call__(self, orig_img, new_size):
        orig_size = orig_img.size()
        if not(self.orig_size == orig_size and self.new_size == new_size):
            self._compute_idxs(orig_size, new_size)
        return orig_img.index_select(1, self.i_v).index_select(2, self.i_h)


def bl_resize(orig_img, new_size):
    orig_size = orig_img.size()
    assert len(orig_size) == 3 and len(new_size) == 3 and\
           orig_size[0] == new_size[0]

    d, o_h, o_w = orig_size
    _, n_h, n_w = new_size

    r_idx = torch.linspace(0, o_h - 1, n_h).cuda()  # TODO: Try to remove cuda!
    i_up = torch.floor(r_idx)
    i_down = i_up + 1.
    f_down = (r_idx - i_up).view(1, n_h, 1)
    f_up = (i_down - r_idx).view(1, n_h, 1)

    i_up = i_up.long().cuda()
    i_down = i_down.clamp(0, o_h - 1).long().cuda()
    f_down = f_down.cuda()
    f_up = f_up.cuda()

    z = orig_img.index_select(1, i_up) * f_up.expand(d, n_h, o_w) +\
        orig_img.index_select(1, i_down) * f_down.expand(d, n_h, o_w)

    r_idx = torch.linspace(0, o_w - 1, n_w).cuda()  # TODO: Try to remove cuda!
    i_left = torch.floor(r_idx)
    i_right = i_left + 1.
    f_left = (r_idx - i_right).view(1, 1, n_w)
    f_right = (i_left - r_idx).view(1, 1, n_w)

    i_left = i_left.long().cuda()
    i_right = i_right.clamp(0, o_w - 1).long().cuda()
    f_left = f_left.cuda()
    f_right = f_right.cuda()

    return z.index_select(2, i_left) * f_left.expand(d, n_h, n_w) +\
           z.index_select(2, i_right) * f_right.expand(d, n_h, n_w)


def nn_resize(orig_img, new_size):
    orig_size = orig_img.size()
    assert len(orig_size) == 3 and len(new_size) == 3 and\
           orig_size[0] == new_size[0]

    d, o_h, o_w = orig_size
    _, n_h, n_w = new_size

    r_idx = torch.linspace(0, o_h - 1, n_h).cuda()  # TODO: Try to remove cuda!
    i_v = torch.round(r_idx).long().cuda()
    r_idx = torch.linspace(0, o_w - 1, n_w).cuda()  # TODO: Try to remove cuda!
    i_h = torch.round(r_idx).long().cuda()

    y = orig_img.index_select(1, i_v).index_select(2, i_h)

    return y


def main():
    orig_size, new_size = torch.Size([1, 7, 7]), torch.Size([1, 5, 5])
    test_img = torch.randn(orig_size).cuda()

    bl_img = bl_resize(test_img, new_size)
    nn_img = nn_resize(test_img, new_size)

    print(test_img)
    print(bl_img)

    assert bl_img.size() == new_size and nn_img.size() == new_size

    bl_resizer = BilinearResizer(orig_size, new_size)
    for _ in range(100):
        test_img = torch.randn(orig_size).cuda()
        bl_img = bl_resizer(test_img, new_size)
        assert bl_img.size() == new_size

    nn_resizer = NNResizer(orig_size, new_size)
    for _ in range(100):
        test_img = torch.randn(orig_size).cuda()
        nn_img = nn_resizer(test_img, new_size)
        assert nn_img.size() == new_size

    print("Ok")

if __name__ == "__main__":
    main()
