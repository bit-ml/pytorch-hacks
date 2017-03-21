"""This script demonstrates multi-process optimization using Adam or RMSprop.

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


# Setup the problem

parser = ArgumentParser()
parser.add_argument("-a", "--algorithm", default="Adam", dest="algorithm",
                    choices=["RMSprop", "Adam"], help="Optimizer to use.")
parser.add_argument("-s", "--steps-no", default=1000, dest="steps_no",
                    type=int, help="Number of optimization steps")
parser.add_argument("-v", "--verbose", action='count', help="Verbosity level")
parser.add_argument("-g", "--gpu", action='store_true', help="Use GPU")


args = parser.parse_args()

STEPS_NO = args.steps_no
BATCH_SIZE = 64
IN_SIZE = 1
OUT_SIZE = 1

Optimizer = getattr(optim, args.algorithm)

# Create some model

class SimpleModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(SimpleModel, self).__init__()
        h_size = in_size + (out_size - in_size) // 2
        self.l1 = nn.Linear(in_size, h_size)
        self.l2 = nn.Linear(h_size, out_size)

    def forward(self, inputs):
        return F.tanh(self.l2(F.relu(self.l1(inputs))))


# Create some data to work with

torch.manual_seed(1)

f = nn.Linear(IN_SIZE, OUT_SIZE)

inputs = torch.randn(STEPS_NO, BATCH_SIZE, IN_SIZE)
targets = F.tanh(f(Variable(inputs.view(-1, IN_SIZE), volatile=True)))
targets = targets.data.view(STEPS_NO, BATCH_SIZE, OUT_SIZE)

def show_parameters_summary(model):
    for name, p in model.state_dict().items():
        sys.stdout.write(" | {:s} : {:f}".format(name, p.abs().sum()))
    sys.stdout.write("\n")
    sys.stdout.flush()

def show_state_summary(state):
    s = state["state"]
    for param, info in s.items():
        sys.stdout.write(" | ".join(["{:s}={:f}".format(
            name, float(x) if not torch.is_tensor(x) else x.abs().sum()
        ) for name, x in info.items()]))
        sys.stdout.write("\n")
    sys.stdout.write("\n")
# ------------------------------
# Initialize model and optimizer

torch.manual_seed(1)

my_model = SimpleModel(IN_SIZE, OUT_SIZE)
original_parameters = deepcopy(my_model.state_dict())
adam_optimizer = Optimizer(my_model.parameters())

sys.stdout.write("Original:")
show_parameters_summary(my_model)

sys.stdout.write("\n --------- MODEL 1 ---------\n\n")


for step in range(STEPS_NO):
    if args.verbose:
        sys.stdout.write(" ---> STEP {:d} <--- \n".format(step))

    adam_optimizer.zero_grad()
    x = Variable(inputs[step])
    y = my_model(x)
    t = Variable(targets[step], requires_grad=False)
    loss = F.smooth_l1_loss(y, t)
    loss.backward()
    adam_optimizer.step()

    if args.verbose:
        sys.stdout.write("Loss @ step {:d}: {:f}\n".format(step, loss.data[0]))

        sys.stdout.write("Optimizer state after step {:d}:\n".format(step))
        show_state_summary(adam_optimizer.state_dict())

        sys.stdout.write("Params after step {:d}:".format(step))
        show_parameters_summary(my_model)

sys.stdout.write("Params after training:")
show_parameters_summary(my_model)

# -----------------------------------------------------
# Train another model starting from the same parameters

my_model_2 = SimpleModel(IN_SIZE, OUT_SIZE)
my_model_2.load_state_dict(original_parameters)
adam_optimizer_2 = Optimizer(my_model_2.parameters())

sys.stdout.write("\n --------- MODEL 2 ---------\n\n")


if args.verbose:
    sys.stdout.write("Optimizer state before step {:d}:\n".format(0))
    show_state_summary(adam_optimizer_2.state_dict())



for step in range(STEPS_NO):
    if args.verbose:
        sys.stdout.write(" ---> STEP {:d} <--- \n".format(step))

    adam_optimizer_2.zero_grad()
    x = Variable(inputs[step])
    y = my_model_2(x)
    t = Variable(targets[step], requires_grad=False)
    loss = F.smooth_l1_loss(y, t)
    loss.backward()
    adam_optimizer_2.step()

    if args.verbose:
        sys.stdout.write("Loss @ step {:d}: {:f}\n".format(step, loss.data[0]))

        sys.stdout.write("Optimizer state after step {:d}:\n".format(step))
        show_state_summary(adam_optimizer_2.state_dict())

        sys.stdout.write("Params after step {:d}:".format(step))
        show_parameters_summary(my_model_2)

sys.stdout.write("Params after training:")
show_parameters_summary(my_model)



# ---------------------------------------------------------------------
# Start some threads that will train the a third model on the same data

def to_shared_state(state):
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

def from_shared_state(shared_state):
    new_state = {}
    for k, v in shared_state.items():
        if type(v) == dict:
            new_state[k] = from_shared_state(v)
        elif torch.is_tensor(v):
            new_state[k] = v
        elif type(v) == Synchronized:
            new_state[k] = v.value
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
        self.shared_model = shared_stuff["shared_model"]
        self.shared_optimizer_state = shared_stuff["shared_optimizer_state"]
        self.lock = shared_stuff["lock"]
        self.crt_step = shared_stuff["crt_step"]
        self.steps_no = shared_stuff["steps_no"]
        self.my_pid = pid
        self.workers_no = workers_no
        self.inputs = shared_stuff["inputs"]
        self.targets = shared_stuff["targets"]

    def run(self):

        steps_no, workers_no = self.steps_no, self.workers_no
        my_pid = self.my_pid
        lock = self.lock
        shared_model = self.shared_model
        shared_state = self.shared_optimizer_state

        inputs, targets = self.inputs, self.targets

        my_optimizer = Optimizer(shared_model.parameters())

        # Decouple gradients
        my_optimizer.zero_grad()
        loss = F.smooth_l1_loss(shared_model(Variable(inputs[0])),
                                Variable(targets[0], requires_grad=False))
        loss.backward()
        for p in shared_model.parameters():
            p.grad.data = p.grad.data.clone()

        # Set initial optimizer state
        shared_optimizer_state = self.shared_optimizer_state
        my_state = my_optimizer.state_dict()
        my_state['state'] = from_shared_state(shared_optimizer_state)
        my_optimizer.load_state_dict(my_state)

        while True:
            with lock:
                if crt_step.value >= steps_no:
                    return
                if (crt_step.value % workers_no) == my_pid:

                    step = crt_step.value
                    if args.verbose:
                        sys.stdout.write(
                            "Worker {:d} does step {:d}.\n".format(my_pid, step)
                        )
                    update_from_shared_values(my_optimizer.state,
                                              shared_optimizer_state)

                    if args.verbose:
                        sys.stdout.write(
                            "Optimizer state before step {:d}:\n".format(step)
                        )
                        show_state_summary(my_optimizer.state_dict())
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


                    update_shared_values(my_optimizer.state,
                                         shared_optimizer_state)

                    if args.verbose:
                        sys.stdout.write(
                            "Loss @ step {:d}: {:f}\n".format(step,loss.data[0])
                        )

                        sys.stdout.write(
                            "Optimizer state after step {:d}:\n".format(step)
                        )
                        show_state_summary(my_optimizer.state_dict())

                        sys.stdout.write("Params after step {:d}:".format(step))
                        show_parameters_summary(shared_model)
                        sys.stdout.flush()

                    crt_step.value += 1


workers_no = 8
crt_step = mp.Value('i', 1)
lock = mp.Lock()

shared_model = SimpleModel(IN_SIZE, OUT_SIZE)

main_optimizer = Optimizer(shared_model.parameters())

shared_model.load_state_dict(original_parameters)
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
    show_state_summary(main_optimizer.state_dict())

    sys.stdout.write("Params after step {:d}:".format(0))
    show_parameters_summary(shared_model)
    sys.stdout.flush()

del main_optimizer


shared_stuff = {
    "shared_model": shared_model,
    "lock": lock,
    "crt_step": crt_step,
    "inputs": inputs,
    "targets": targets,
    "steps_no": STEPS_NO,
    "shared_optimizer_state": optimizer_state
}

workers = [Worker(i, workers_no, shared_stuff) for i in range(workers_no)]

for worker in workers:
    worker.start()

for worker in workers:
    worker.join()

sys.stdout.write("Params after training:")
show_parameters_summary(shared_model)
sys.stdout.flush()
