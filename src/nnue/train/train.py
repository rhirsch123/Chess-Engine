import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

FEATURE_SIZE = 768
L1_SIZE = 1536
L2_SIZE = 16
L3_SIZE = 32

SCALE = 400

this_dir = os.path.dirname(os.path.abspath(__file__))
ext_path = os.path.join(this_dir, "data_loader.cc")

nnue_loader = load(
    name="nnue_batchloader",
    sources=[ext_path],
    extra_cflags=["-O3", "-std=c++20", "-march=native"],
    with_cuda=False,
    verbose=False,
)

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(FEATURE_SIZE, L1_SIZE)
        self.l2_weight_stm = nn.Parameter(torch.empty(L1_SIZE, L2_SIZE))
        self.l2_weight_opp = nn.Parameter(torch.empty(L1_SIZE, L2_SIZE))
        self.l2_bias = nn.Parameter(torch.empty(L2_SIZE))
        self.l3 = nn.Linear(L2_SIZE, L3_SIZE)
        self.out = nn.Linear(L3_SIZE, 1)

        nn.init.normal_(self.l1.weight, mean=0.0, std=(2.0 / FEATURE_SIZE) ** 0.5)
        nn.init.zeros_(self.l1.bias)

        nn.init.normal_(self.l2_weight_stm, mean=0.0, std=(2.0 / L1_SIZE) ** 0.5)
        nn.init.normal_(self.l2_weight_opp, mean=0.0, std=(2.0 / L1_SIZE) ** 0.5)
        nn.init.zeros_(self.l2_bias)
        
        nn.init.normal_(self.l3.weight, mean=0.0, std=(2.0 / L2_SIZE) ** 0.5)
        nn.init.zeros_(self.l3.bias)

        nn.init.normal_(self.out.weight, mean=0.0, std=(2.0 / L3_SIZE) ** 0.5)
        nn.init.zeros_(self.out.bias)


    def forward(self, ft_stm, ft_opp):
        x_stm = torch.clamp(self.l1(ft_stm), 0.0, 1.0)
        x_opp = torch.clamp(self.l1(ft_opp), 0.0, 1.0)

        x = self.l2_bias + (x_stm @ self.l2_weight_stm + x_opp @ self.l2_weight_opp)

        x = torch.clamp(x, 0.0, 1.0)
        x = self.l3(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.out(x).squeeze(-1)

        return torch.sigmoid(x)


def train(data_file, num_positions, epochs, batch_size):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    cpu_threads = 2
    torch.set_num_threads(cpu_threads)

    pin = (device.type == "cuda")
    loader = nnue_loader.NNUEBatchLoader(data_file, num_positions, pin_memory=pin)

    model = NNUE().to(device)

    adamw_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name == "l1.weight":
            adamw_params.append(param)
        else:
            adam_params.append(param)

    opt = torch.optim.AdamW(
        [
            {"params": adamw_params, "weight_decay": 0.01},
            {"params": adam_params, "weight_decay": 0.0}
        ]
    )

    loss_fn = nn.MSELoss()

    # lr schedule
    lr_steps = 100
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=lr_steps, eta_min=1e-6)
    total_batches = epochs * (num_positions // batch_size)
    lr_step_period = total_batches // lr_steps

    info_period = 100
    current_batch = 0
    
    for epoch in range(1, epochs + 1):
        print(f"epoch: {epoch}\n")

        model.train()
        running_loss = 0.0
        current_position = 0
        start_time = time.time()

        while current_position < num_positions:
            x_stm, x_opp, ev = loader.get_batch(current_position, batch_size)

            current_batch += 1
            current_position += batch_size

            x_stm = x_stm.to(device, non_blocking=True)
            x_opp = x_opp.to(device, non_blocking=True)
            ev = ev.to(device, non_blocking=True)

            opt.zero_grad()
            out = model(x_stm, x_opp)

            target = torch.sigmoid(ev.float() / SCALE).squeeze(-1)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()

            # decrease learning rate
            lr = sched.get_last_lr()[0]
            if current_batch % lr_step_period == 0 and lr > sched.eta_min:
                sched.step()
                print(f"step {sched._step_count - 1}: learning rate dropped to {lr:.7f}\n")

            # clip weights so quantized inference doesn't overflow
            clip = 1.98
            with torch.no_grad():
                model.l2_weight_stm.clamp_(-clip, clip)
                model.l2_weight_opp.clamp_(-clip, clip)

            running_loss += loss.item() * batch_size

            if current_batch % info_period == 0:
                print(f"position: {current_position}")
                print(f"loss: {(running_loss / current_position):.6f}")

                elapsed = time.time() - start_time

                pps = int(current_position / elapsed)
                print(f"positions per second: {pps}")

                positions_remaining = num_positions * epochs - (current_position + num_positions * (epoch - 1))
                hours_remaining = (elapsed * positions_remaining) / (60 * 60 * current_position)
                print(f"hours remaining: {hours_remaining:.2f}\n")

                torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    total_positions = 5000000000
    batch_size = 4096
    train_positions = batch_size * (total_positions // batch_size)
    train("data.bin", train_positions, 2, batch_size)