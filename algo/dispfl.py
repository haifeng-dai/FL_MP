"""DisPFL：带动态稀疏掩码的去中心化个性化联邦学习。"""

import argparse
import math

import torch

from algo.core import BaseClient, BaseServer
from runtime import adjacency, clone_state


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dense-ratio", type=float, default=0.5)
    parser.add_argument("--anneal-factor", type=float, default=0.1)
    parser.add_argument("--erk-power-scale", type=float, default=1.0)


class DisPFLClient(BaseClient):
    def train(self, task):
        masks = {
            name: value.to(self.device) for name, value in task.payload["masks"].items()
        }
        optimizer = self.build_optimizer()
        self.model.train()
        total, batches = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                optimizer.zero_grad(set_to_none=True)
                loss = self.ce_loss(
                    self.model(inputs.to(self.device)), targets.to(self.device)
                )
                loss.backward()
                optimizer.step()
                for name, parameter in self.model.named_parameters():
                    if name in masks:
                        parameter.data.mul_(masks[name])
                total += loss.item()
                batches += 1
        inputs, targets = next(iter(self.train_loader(task.client_id)))
        self.model.zero_grad(set_to_none=True)
        loss = self.ce_loss(self.model(inputs.to(self.device)), targets.to(self.device))
        loss.backward()
        alpha = (
            task.payload["anneal_factor"]
            * 0.5
            * (
                1
                + math.cos(task.payload["round"] * math.pi / task.payload["num_rounds"])
            )
        )
        updated = {}
        for name, parameter in self.model.named_parameters():
            if name not in masks:
                continue
            mask = masks[name].clone()
            active = int(mask.sum().item())
            remove = min(math.ceil(alpha * active), active)
            if remove:
                active_weights = torch.where(
                    mask.bool(), parameter.data.abs(), torch.inf
                )
                removed = active_weights.flatten().topk(remove, largest=False).indices
                mask.flatten()[removed] = 0
                available = int((mask == 0).sum().item())
                regrow = min(remove, available)
                gradient = parameter.grad.abs().masked_fill(mask.bool(), -torch.inf)
                grown = gradient.flatten().topk(regrow).indices
                mask.flatten()[grown] = 1
            updated[name] = mask.cpu()
        return self.result(task, total / batches, {"masks": updated})


class Server(BaseServer):
    client_class = DisPFLClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        if not 0 < args.dense_ratio <= 1:
            raise ValueError("dense-ratio 必须在 (0, 1] 内")
        self.graph = adjacency(args).to(self.device)
        self.anneal_factor: float = args.anneal_factor
        self.masks = {
            client_id: self._initial_masks(args.dense_ratio, args.erk_power_scale)
            for client_id in range(self.num_clients)
        }
        self.previous_masks = {
            client_id: clone_state(mask) for client_id, mask in self.masks.items()
        }
        self._apply_masks(self.client_states, self.masks)

    def _initial_masks(self, density, power_scale):
        parameters = dict(self.model.named_parameters())
        raw = {
            name: (sum(parameter.shape) / parameter.numel()) ** power_scale
            for name, parameter in parameters.items()
        }
        total = sum(parameter.numel() for parameter in parameters.values())
        dense = set()
        while True:
            remaining = total * density - sum(
                parameters[name].numel() for name in dense
            )
            divisor = sum(
                raw[name] * parameters[name].numel()
                for name in raw
                if name not in dense
            )
            epsilon = remaining / divisor
            newly_dense = {
                name for name in raw if name not in dense and raw[name] * epsilon > 1
            }
            if not newly_dense:
                break
            dense.update(newly_dense)
        masks = {}
        for name, parameter in parameters.items():
            layer_density = 1 if name in dense else raw[name] * epsilon
            count = int(parameter.numel() * layer_density)
            mask = torch.zeros(parameter.numel())
            mask[torch.randperm(parameter.numel())[:count]] = 1
            masks[name] = mask.reshape_as(parameter).cpu()
        for name, value in self.model.state_dict().items():
            if name not in masks:
                masks[name] = torch.ones_like(value)
        return masks

    def _apply_masks(self, states, masks):
        for client_id, state in states.items():
            for name, mask in masks[client_id].items():
                state[name].mul_(mask)

    def training_payloads(self):
        return {
            client_id: {
                "masks": self.masks[client_id],
                "round": len(self.accuracies),
                "num_rounds": self.num_rounds,
                "anneal_factor": self.anneal_factor,
            }
            for client_id in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        for client_id in self.selected:
            self.masks[client_id] = results[client_id].payload["masks"]
        self.previous_masks = {
            client_id: clone_state(mask) for client_id, mask in self.masks.items()
        }

    def run_round(self):
        self._mix_sparse_states()
        return super().run_round()

    def _mix_sparse_states(self):
        old = self.client_states
        new = {client_id: {} for client_id in range(self.num_clients)}
        for name, value in old[0].items():
            if name not in self.previous_masks[0]:
                for client_id in range(self.num_clients):
                    new[client_id][name] = old[client_id][name].clone()
                continue
            states = torch.stack(
                [
                    old[i][name].to(self.device).flatten()
                    for i in range(self.num_clients)
                ]
            )
            masks = torch.stack(
                [
                    self.previous_masks[i][name].to(self.device).flatten()
                    for i in range(self.num_clients)
                ]
            )
            count = self.graph @ masks
            summed = self.graph @ (states * masks)
            mixed = summed / count.clamp_min(1)
            for client_id in range(self.num_clients):
                new[client_id][name] = (
                    (mixed[client_id] * masks[client_id]).reshape_as(value).cpu()
                )
        self.client_states = new

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "graph": self.graph.cpu(),
            "masks": self.masks,
        }
