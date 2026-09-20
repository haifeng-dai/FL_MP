"""L2C：通过验证损失学习邻居协作权重的两阶段算法。"""

import torch
from torch.utils.data import Subset

from ..core import BaseClient, BaseServer, clone_state
from ..core.decentralized import adjacency


def add_arguments(parser):
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr-alpha", type=float, default=0.01)
    parser.add_argument("--prune-round", type=int, default=0)
    parser.add_argument("--prune-num", type=int, default=0)


class L2CClient(BaseClient):
    def train(self, task):
        if task.payload["phase"] == 1:
            return self.phase_one(task)
        return self.phase_two(task)

    def phase_one(self, task):
        state = clone_state(self.model.state_dict())
        size = len(self.train_sets[task.client_id])
        indices = torch.randperm(size).tolist()
        cut = int(size * task.payload["val_ratio"])
        train = Subset(self.train_sets[task.client_id], indices[cut:])
        optimizer = self.build_optimizer()
        total, batches = 0.0, 0
        self.model.train()
        for _ in range(self.num_epochs):
            for inputs, targets in torch.utils.data.DataLoader(
                train, self.batch_size, shuffle=True
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = self.ce_loss(
                    self.model(inputs.to(self.device)), targets.to(self.device)
                )
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
        updated = clone_state(self.model.state_dict())
        delta = {name: state[name] - updated[name] for name in state}
        return self.result(
            task,
            total / batches,
            {"state_t": state, "delta": delta, "val_indices": indices[:cut]},
        )

    def phase_two(self, task):
        payload = task.payload
        alpha = payload["alpha"].to(self.device).detach().requires_grad_(True)
        weights = torch.softmax(alpha, dim=0)
        state = {
            name: value.to(self.device).clone()
            for name, value in payload["state_t"].items()
        }
        for name in state:
            deltas = torch.stack(
                [delta[name].to(self.device) for delta in payload["deltas"]]
            )
            state[name] -= (deltas * weights.view(-1, *([1] * (deltas.ndim - 1)))).sum(
                0
            )
        validation = Subset(self.train_sets[task.client_id], payload["val_indices"])
        if len(validation):
            inputs, targets = next(
                iter(torch.utils.data.DataLoader(validation, self.batch_size))
            )
            output = torch.func.functional_call(
                self.model, state, (inputs.to(self.device),)
            )
            gradient = torch.autograd.grad(
                self.ce_loss(output, targets.to(self.device)), alpha
            )[0]
            alpha = (alpha - payload["lr_alpha"] * gradient).detach()
        self.model.load_state_dict(state)
        return self.result(
            task,
            0.0,
            {"alpha": alpha.cpu(), "weights": weights.detach().cpu().clone()},
        )


class Server(BaseServer):
    client_class = L2CClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.val_ratio, self.lr_alpha = args.val_ratio, args.lr_alpha
        self.prune_round, self.prune_num = args.prune_round, args.prune_num
        self.graph = adjacency(args)
        self.alphas = {
            i: torch.zeros(int(self.graph[i].sum())) for i in range(self.num_clients)
        }
        self.delta_cache = {}
        self.phase = 1
        self.phase_one = {}

    def training_payloads(self):
        if self.phase == 1:
            return {i: {"phase": 1, "val_ratio": self.val_ratio} for i in self.selected}
        payloads = {}
        for i in self.selected:
            neighbors = torch.where(self.graph[i] > 0)[0].tolist()
            fallback = {
                name: torch.zeros_like(value)
                for name, value in self.phase_one[i]["delta"].items()
            }
            payloads[i] = {
                "phase": 2,
                "state_t": self.phase_one[i]["state_t"],
                "deltas": [
                    self.phase_one[n]["delta"]
                    if n in self.phase_one
                    else self.delta_cache.get(n, fallback)
                    for n in neighbors
                ],
                "alpha": self.alphas[i],
                "val_indices": self.phase_one[i]["val_indices"],
                "lr_alpha": self.lr_alpha,
            }
        return payloads

    def run_round(self):
        self.select_clients()
        self.phase = 1
        first = self.run_clients()
        self.phase_one = {i: first[i].payload for i in self.selected}
        self.phase = 2
        second = self.run_clients()
        for i in self.selected:
            self.client_states[i] = second[i].state
            self.alphas[i] = second[i].payload["alpha"]
            self.delta_cache[i] = self.phase_one[i]["delta"]
        self._prune(second)
        loss = sum(result.loss for result in first.values()) / len(first)
        return loss, self.evaluate()

    def _prune(self, results):
        if not self.prune_num or len(self.accuracies) + 1 != self.prune_round:
            return
        for i in self.selected:
            neighbors = torch.where(self.graph[i] > 0)[0].tolist()
            candidates = [
                (results[i].payload["weights"][j], neighbor)
                for j, neighbor in enumerate(neighbors)
                if neighbor != i
            ]
            for _, neighbor in sorted(candidates)[: self.prune_num]:
                self.graph[i, neighbor] = 0

    def apply_results(self, results):
        raise RuntimeError("L2C 使用 run_round 的两阶段结果")

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "graph": self.graph,
            "alphas": self.alphas,
            "delta_cache": self.delta_cache,
        }
