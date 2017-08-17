# Tudor Berariu @ Bitdefender, 2017

import torch

def bl_resize(orig_img, new_size):
    orig_size = orig_img.size()
    assert len(orig_size) == 3 and len(new_size) == 3 and\
           orig_size[0] == new_size[0]

    d, o_h, o_w = orig_size
    _, n_h, n_w = new_size

    r_idx = torch.linspace(0, o_h - 1, n_h).cuda()  # TODO: Try to remove cuda!
    i_up = torch.floor(r_idx)
    i_down = torch.ceil(r_idx)
    f_down = (r_idx - i_up).view(1, n_h, 1)
    f_up = (i_down - r_idx).view(1, n_h, 1)

    i_up = i_up.long().cuda()
    i_down = i_down.long().cuda()
    f_down = f_down.cuda()
    f_up = f_up.cuda()

    z = orig_img.index_select(1, i_up) * f_up.expand(d, n_h, o_w) +\
        orig_img.index_select(1, i_down) * f_down.expand(d, n_h, o_w)

    r_idx = torch.linspace(0, o_w - 1, n_w).cuda()  # TODO: Try to remove cuda!
    i_left = torch.floor(r_idx)
    i_right = torch.ceil(r_idx)
    f_left = (r_idx - i_right).view(1, 1, n_w)
    f_right = (i_left - r_idx).view(1, 1, n_w)

    i_left = i_left.long().cuda()
    i_right = i_right.long().cuda()
    f_left = f_left.cuda()
    f_right = f_right.cuda()

    y = z.index_select(2, i_left) * f_left.expand(d, n_h, n_w) +\
        z.index_select(2, i_right) * f_right.expand(d, n_h, n_w)

    return y

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
    orig_size, new_size = torch.Size([3, 1024, 1024]), torch.Size([3, 33, 233])
    test_img = torch.randn(orig_size).cuda()

    bl_img = bl_resize(test_img, new_size)
    nn_img = nn_resize(test_img, new_size)

    assert bl_img.size() == new_size and nn_img.size() == new_size

    print("Ok")

if __name__ == "__main__":
    main()
