"""This script demonstrates multi-process optimization using one of
the following:
 - Adadelta
 - Adagrad
 - Adam
 - ASGD
 - SGD
 - Rprop
 - RMSprop

This script doesn't work with:
 - LBFGS   - it requires a closure function
 - Adamax  - uses a new tensor for `exp_inf`

The async version uses locks in order to replicate the serial
behaviour. Real algorithms probably won't use those locks!

"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from torch.autograd import Variable
import sys
from copy import deepcopy
from argparse import ArgumentParser

class SimpleModel(nn.Module):
    """A 2-layer neural network."""

    def __init__(self, in_size, out_size):
        super(SimpleModel, self).__init__()
        h_size = in_size + (out_size - in_size) // 2
        self.l1 = nn.Linear(in_size, h_size)
        self.l2 = nn.Linear(h_size, out_size)

    def forward(self, inputs):
        return F.tanh(self.l2(F.relu(self.l1(inputs))))

def show_parameters_summary(model):
    """Displays a fingerprint of the parameters of a given model."""

    for name, p in model.state_dict().items():
        sys.stdout.write(" | {:s} : {:f}".format(name, p.abs().sum()))
    sys.stdout.write("\n")
    sys.stdout.flush()

def show_state_summary(optimizer):
    """Displays a fingerprint of the state of an optimizer."""

    s = optimizer.state_dict()["state"]
    for param, info in s.items():
        sys.stdout.write(" | ".join(["{:s}@{:d}={:f}".format(
            name, id(x), float(x) if not torch.is_tensor(x) else x.abs().sum()
        ) for name, x in info.items()]))
        sys.stdout.write("\n")
    sys.stdout.write("\n")
    sys.stdout.flush()

def train_single_process(initial_parameters, inputs, targets, args):

    torch.manual_seed(1)

    IN_SIZE = args.in_size
    OUT_SIZE = args.out_size
    STEPS_NO = args.steps_no

    my_model = SimpleModel(IN_SIZE, OUT_SIZE)
    if args.gpu:
        my_model.cuda()

    my_model.load_state_dict(initial_parameters)

    optimizer = getattr(optim, args.algorithm)(my_model.parameters(),
                                               **eval(args.optim_args))

    sys.stdout.write("Initial parameters:")
    show_parameters_summary(my_model)

    sys.stdout.write("\n --------- MODEL ---------\n\n")

    for step in range(STEPS_NO):
        if args.verbose:
            sys.stdout.write(" ---> STEP {:d} <--- \n".format(step))

        optimizer.zero_grad()
        x = Variable(inputs[step])
        y = my_model(x)
        t = Variable(targets[step], requires_grad=False)
        loss = F.smooth_l1_loss(y, t)
        loss.backward()
        optimizer.step()

        if args.verbose:
            sys.stdout.write("Loss @ step {:d}: {:f}\n".format(step,
                                                               loss.data[0]))

            sys.stdout.write("Optimizer state after step {:d}:\n".format(step))
            show_state_summary(optimizer)

            sys.stdout.write("Params after step {:d}:".format(step))
            show_parameters_summary(my_model)

    sys.stdout.write("Params after training:")
    show_parameters_summary(my_model)


def to_shared_state(state):
    """Takes an optimizer state, puts scalars in mp.Value and shares tensors."""

    new_state = {}
    for k, v in state.items():
        if type(v) == dict:
            new_state[k] = to_shared_state(v)
        elif torch.is_tensor(v):
            v.share_memory_()
            new_state[k] = v
        elif type(v) == int:
            new_state[k] = mp.Value('i', v)
        elif type(v) == float:
            new_state[k] = mp.Value('f', v)
        else:
            raise NotImplemented()
    return new_state

def repair_ids(shared_state, original_id_to_label, label_to_id):
    """Given an optimizer state and some new ids for the parameters,
    creates a new state with the new ids linked to the same tensors.

    Useful when using an optimizer state used on some old model on a
    new one.

    """

    new_state = {}
    for param_id, state in shared_state.items():
        new_param_id = label_to_id[original_id_to_label[param_id]]
        new_state[new_param_id] = state
    return new_state

def from_shared_state(shared_state):
    """Used on what you get from `to_shared_state`, it only extracts
    tensors from mp.Value

    """
    new_state = {}
    for param_id, state in shared_state.items():
        new_state[param_id] = {}
        for name, value in state.items():
            if type(value) == Synchronized:
                new_state[param_id][name] = value.value
            elif torch.is_tensor(value):
                new_state[param_id][name] = value
            else:
                raise NotImplemented()
    return new_state

def update_from_shared_values(optimizer_state, shared_state):
    for param_id, state in shared_state.items():
        assert param_id in optimizer_state
        for name, value in state.items():
            if type(value) == Synchronized:
                optimizer_state[param_id][name] = value.value
            elif torch.is_tensor(value):
                optimizer_state[param_id][name] = value

def update_shared_values(optimizer_state, shared_state):
    for param_id, state in shared_state.items():
        assert param_id in optimizer_state
        for name, value in state.items():
            if type(value) == Synchronized:
                value.value = optimizer_state[param_id][name]



class Worker(mp.Process):
    def __init__(self, pid, workers_no, shared_stuff):
        super(Worker, self).__init__()
        self.my_pid = pid
        self.workers_no = workers_no
        self.shared_stuff = shared_stuff

    def run(self):

        shared_model = self.shared_stuff["shared_model"]
        shared_optimizer_state = self.shared_stuff["shared_optimizer_state"]
        lock = self.shared_stuff["lock"]
        crt_step = self.shared_stuff["crt_step"]
        steps_no = self.shared_stuff["steps_no"]
        inputs = self.shared_stuff["inputs"]
        targets = self.shared_stuff["targets"]
        algorithm = self.shared_stuff["algorithm"]
        optim_args = self.shared_stuff["optim_args"]
        verbose = self.shared_stuff.get("verbose", None)
        original_id_to_label = self.shared_stuff["original_id_to_label"]

        workers_no = self.workers_no
        my_pid = self.my_pid

        my_optimizer = getattr(optim, algorithm)(shared_model.parameters(),
                                                 **eval(optim_args))

        # Decouple gradients
        loss = F.smooth_l1_loss(shared_model(Variable(inputs[0])),
                                Variable(targets[0], requires_grad=False))

        loss.backward()

        for p in shared_model.parameters():
            p.grad.data = p.grad.data.clone()


        # Set initial optimizer state
        p_id = {id(p.data): id(p) for p in shared_model.parameters()}
        label_to_id = \
                {n: p_id[id(p)] for n, p in shared_model.state_dict().items()}


        shared_optimizer_state = repair_ids(shared_optimizer_state,
                                            original_id_to_label,
                                            label_to_id)
        my_state = my_optimizer.state_dict()
        my_state['state'] = from_shared_state(shared_optimizer_state)
        my_optimizer.load_state_dict(my_state)

        while True:
            with lock:
                if crt_step.value >= steps_no:
                    return
                if (crt_step.value % workers_no) != my_pid:
                    continue

                step = crt_step.value
                if verbose:
                    sys.stdout.write(
                        "Worker {:d} does step {:d}.\n".format(my_pid, step)
                    )

                update_from_shared_values(my_optimizer.state,
                                          shared_optimizer_state)

                if verbose:
                    sys.stdout.write(
                        "Optimizer state before step {:d}:\n".format(step)
                    )
                    show_state_summary(my_optimizer)
                    sys.stdout.write(
                        "Params before step {:d}:".format(step)
                    )
                    show_parameters_summary(shared_model)
                    sys.stdout.flush()


                my_optimizer.zero_grad()
                x = Variable(inputs[step])
                y = shared_model(x)
                t = Variable(targets[step], requires_grad=False)
                loss = F.smooth_l1_loss(y, t)
                loss.backward()
                my_optimizer.step()

                update_shared_values(my_optimizer.state, shared_optimizer_state)

                if verbose:
                    sys.stdout.write(
                        "Loss @ step {:d}: {:f}\n".format(step,loss.data[0])
                    )

                    sys.stdout.write(
                        "Optimizer state after step {:d}:\n".format(step)
                    )
                    show_state_summary(my_optimizer)

                    sys.stdout.write("Params after step {:d}:".format(step))
                    show_parameters_summary(shared_model)
                    sys.stdout.flush()

                crt_step.value += 1


def train_multi_process(initial_parameters, inputs, targets, args):

    # ---------------------------------------------------------------------
    # Start some threads that will train the a third model on the same data

    WORKERS_NO = args.workers_no
    crt_step = mp.Value('i', 1)
    lock = mp.Lock()

    IN_SIZE = args.in_size
    OUT_SIZE = args.out_size
    STEPS_NO = args.steps_no

    shared_model = SimpleModel(IN_SIZE, OUT_SIZE)
    if args.gpu:
        shared_model.cuda()

    main_optimizer = getattr(optim, args.algorithm)(shared_model.parameters(),
                                                    **eval(args.optim_args))

    shared_model.load_state_dict(initial_parameters)
    shared_model.share_memory()

    inputs.share_memory_()
    targets.share_memory_()

    main_optimizer.zero_grad()
    loss = F.smooth_l1_loss(shared_model(Variable(inputs[0])),
                            Variable(targets[0], requires_grad=False))
    loss.backward()
    main_optimizer.step()
    optimizer_state = to_shared_state(main_optimizer.state_dict()['state'])

    sys.stdout.write("\n --------- SHARED MODEL ---------\n\n")

    if args.verbose:
        sys.stdout.write(
            "Loss at step {:d}: {:.4f}\n".format(0, loss.data[0])
        )

        sys.stdout.write(
            "Optimizer state after step 0:\n".format(0)
        )
        show_state_summary(main_optimizer)

        sys.stdout.write("Params after step {:d}:".format(0))
        show_parameters_summary(shared_model)
        sys.stdout.flush()

    del main_optimizer

    p_id = {id(p.data): id(p) for p in shared_model.parameters()}
    original_id_to_label = \
                {p_id[id(p)]: n for n, p in shared_model.state_dict().items()}

    shared_stuff = {
        "shared_model": shared_model,
        "lock": lock,
        "crt_step": crt_step,
        "inputs": inputs,
        "targets": targets,
        "steps_no": STEPS_NO,
        "shared_optimizer_state": optimizer_state,
        "algorithm": args.algorithm,
        "optim_args": args.optim_args,
        "verbose": args.verbose,
        "original_id_to_label": original_id_to_label
    }

    workers = [Worker(i, WORKERS_NO, shared_stuff) for i in range(WORKERS_NO)]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    sys.stdout.write("Params after training:")
    show_parameters_summary(shared_model)
    sys.stdout.flush()

def main():
    parser = ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="Adam", dest="algorithm",
                        help="Optimizer to use.")
    parser.add_argument("-oa", "--optim-args", default="{}", dest="optim_args",
                        help="Optimizer to use.")
    parser.add_argument("-s", "--steps-no", default=1000, dest="steps_no",
                        type=int, help="Number of optimization steps")
    parser.add_argument("-i", "--in-size", default=256, dest="in_size",
                        type=int, help="Size of input vectors")
    parser.add_argument("-o", "--out-size", default=128, dest="out_size",
                        type=int, help="Size of output vectors")
    parser.add_argument("-b", "--batch-size", default=64, dest="batch_size",
                        type=int, help="Batch size")
    parser.add_argument("-w", "--workers-no", default=8, dest="workers_no",
                        type=int, help="Number of processes")
    parser.add_argument("-v", "--verbose", action='count',
                        help="Verbosity level")
    parser.add_argument("-g", "--gpu", action='store_true', help="Use GPU")


    args = parser.parse_args()

    STEPS_NO = args.steps_no
    BATCH_SIZE = args.batch_size
    IN_SIZE = args.in_size
    OUT_SIZE = args.out_size


    mp.set_start_method('spawn')

    # Create some data to work with

    torch.manual_seed(1)

    f = nn.Linear(IN_SIZE, OUT_SIZE)

    inputs = torch.randn(STEPS_NO, BATCH_SIZE, IN_SIZE)
    targets = F.tanh(f(Variable(inputs.view(-1, IN_SIZE), volatile=True)))
    targets = targets.data.view(STEPS_NO, BATCH_SIZE, OUT_SIZE)

    if args.gpu:
        inputs = inputs.cuda()
        targets = targets.cuda()

    my_model = SimpleModel(IN_SIZE, OUT_SIZE)
    if args.gpu:
        my_model.cuda()

    initial_parameters = my_model.state_dict()
    del my_model

    train_single_process(deepcopy(initial_parameters), inputs, targets, args)
    train_multi_process(deepcopy(initial_parameters), inputs, targets, args)

if __name__ == "__main__":
    main()
