"""DFedAvgM 算法实现。"""

from ..core import BaseClient, BaseServer
from ..core.decentralized import adjacency, metropolis_hastings, mix_states


class DFedAvgMClient(BaseClient):
    def train(self, task):
        optimizer = self.build_optimizer()
        if task.payload["optimizer_state"] is not None:
            optimizer.load_state_dict(task.payload["optimizer_state"])
        self.model.train()
        total, batches = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                optimizer.zero_grad(set_to_none=True)
                loss = self.ce_loss(
                    self.model(inputs.to(self.device, non_blocking=True)),
                    targets.to(self.device, non_blocking=True),
                )
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
        return self.result(task, total / batches, {"optimizer_state": None})


class Server(BaseServer):
    client_class = DFedAvgMClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.graph = adjacency(args)
        self.weights = metropolis_hastings(self.graph).to(self.device)
        self.optimizer_states = {
            client_id: None for client_id in range(self.num_clients)
        }

    def training_payloads(self):
        return {
            client_id: {"optimizer_state": self.optimizer_states[client_id]}
            for client_id in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        for client_id in self.selected:
            self.optimizer_states[client_id] = results[client_id].payload[
                "optimizer_state"
            ]
        self.client_states = {
            client_id: state
            for client_id, state in enumerate(
                mix_states(
                    [self.client_states[index] for index in range(self.num_clients)],
                    self.weights,
                    self.device,
                )
            )
        }

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "graph": self.graph.cpu(),
            "optimizer_states": self.optimizer_states,
        }
