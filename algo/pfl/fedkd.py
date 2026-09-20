"""FedKD 算法实现。"""

import torch

from ..core import BaseClient, BaseServer, clone_state


def add_arguments(parser):
    parser.add_argument("--lr-g", type=float, default=0.01)
    parser.add_argument("--energy", type=float, default=0.9)


def compress(tensor, energy):
    if tensor.ndim not in (2, 4):
        return tensor.detach().cpu()
    shape = tensor.shape
    matrix = tensor.reshape(shape[0], -1) if tensor.ndim == 4 else tensor
    try:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        try:
            u, s, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
        except RuntimeError:
            return tensor.detach().cpu()
    total_energy = s.square().sum()
    if total_energy == 0:
        return tensor.detach().cpu()
    cumulative_energy = torch.cumsum(s.square(), dim=0)
    reached = cumulative_energy > energy * total_energy
    rank = (
        len(s)
        if not reached.any()
        else int(torch.searchsorted(reached.int(), 1).item() + 1)
    )
    return {
        "u": u[:, :rank].detach().cpu(),
        "s": s[:rank].detach().cpu(),
        "vh": vh[:rank].detach().cpu(),
        "original_shape": shape,
        "is_compressed": True,
    }


def decompress(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if not value.get("is_compressed"):
        raise ValueError("未知的 FedKD 压缩参数格式")
    matrix = value["u"].to(device) @ (
        torch.diag(value["s"].to(device)) @ value["vh"].to(device)
    )
    return matrix.view(value["original_shape"])


class FedKDClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.lr_g: float = args.lr_g
        self.energy: float = args.energy

    def train(self, task):
        global_model = self.copy_model(
            {
                k: decompress(v, self.device)
                for k, v in task.payload["compressed"].items()
            }
        ).requires_grad_(True)
        align = torch.nn.Linear(self.feature_dim, self.feature_dim, bias=False).to(
            self.device
        )
        if task.payload["align"] is not None:
            align.load_state_dict(task.payload["align"])
        local_opt = self.build_optimizer()
        global_opt = torch.optim.SGD(
            global_model.parameters(),
            lr=self.lr_g,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        align_opt = torch.optim.SGD(
            align.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        total = 0.0
        steps = 0
        for _ in range(self.num_epochs):
            for x, y in self.train_loader(task.client_id):
                x, y = x.to(self.device), y.to(self.device)
                f = self.model.extractor(x)
                fg = global_model.extractor(x)
                o = self.model.classifier(f)
                og = global_model.classifier(fg)
                scale = self.ce_loss(o, y).item() + self.ce_loss(og, y).item() + 1e-8
                loss = (
                    self.ce_loss(o, y)
                    + torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(o, 1),
                        torch.nn.functional.softmax(og.detach(), 1),
                        reduction="batchmean",
                    )
                    / scale
                    + torch.nn.functional.mse_loss(f, align(fg.detach())) / scale
                )
                lossg = (
                    self.ce_loss(og, y)
                    + torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(og, 1),
                        torch.nn.functional.softmax(o.detach(), 1),
                        reduction="batchmean",
                    )
                    / scale
                    + torch.nn.functional.mse_loss(f.detach(), align(fg)) / scale
                )
                local_opt.zero_grad()
                global_opt.zero_grad()
                align_opt.zero_grad()
                loss.backward(retain_graph=True)
                lossg.backward()
                local_opt.step()
                global_opt.step()
                align_opt.step()
                total += loss.item()
                steps += 1
        return self.result(
            task,
            total / steps,
            {
                "compressed": {
                    k: compress(v, self.energy)
                    for k, v in global_model.state_dict().items()
                },
                "align": clone_state(align.state_dict()),
            },
        )


class Server(BaseServer):
    client_class = FedKDClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.energy: float = args.energy
        self.compressed = {
            k: compress(v, self.energy) for k, v in self.model.state_dict().items()
        }
        self.align = {i: None for i in range(self.num_clients)}

    def training_payloads(self):
        return {
            i: {"compressed": self.compressed, "align": self.align[i]}
            for i in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        for i in self.selected:
            self.align[i] = results[i].payload["align"]
        total = sum(results[i].num_samples for i in self.selected)
        state = {}
        for name in self.model.state_dict():
            state[name] = (
                torch.stack(
                    [
                        decompress(results[i].payload["compressed"][name], self.device)
                        * results[i].num_samples
                        for i in self.selected
                    ]
                ).sum(dim=0)
                / total
            ).cpu()
        self.model.load_state_dict(state)
        self.compressed = {
            k: compress(v, self.energy) for k, v in self.model.state_dict().items()
        }

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "compressed": self.compressed,
            "align": self.align,
        }
