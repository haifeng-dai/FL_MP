"""DFedPGP：Push-Sum 去偏的解耦去中心化训练。"""

import torch

from ..core import BaseClient, BaseServer, clone_state
from ..core.decentralized import adjacency, mix_states


def add_arguments(parser):
    parser.add_argument("--lr-v", type=float, default=0.01)
    parser.add_argument("--local-v-epochs", type=int, default=1)
    parser.add_argument("--momentum-v", type=float, default=0.0)
    parser.add_argument("--weight-decay-v", type=float, default=0.0)


class DFedPGPClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.lr_v: float = args.lr_v
        self.local_v_epochs: int = args.local_v_epochs
        self.momentum_v: float = args.momentum_v
        self.weight_decay_v: float = args.weight_decay_v

    def train(self, task):
        body: dict[str, torch.Tensor] = task.payload["body"]
        head: dict[str, torch.Tensor] = task.payload["head"]
        mu: float = task.payload["mu"]
        state = {**body, **head}
        self.model.load_state_dict(state)
        self.model.extractor.load_state_dict(
            {
                name.removeprefix("extractor."): value.to(self.device) / mu
                for name, value in body.items()
            }
        )
        head_optimizer = torch.optim.SGD(
            self.model.classifier.parameters(),
            lr=self.lr_v,
            momentum=self.momentum_v,
            weight_decay=self.weight_decay_v,
        )
        for parameter in self.model.extractor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(True)
        for _ in range(self.local_v_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                head_optimizer.zero_grad(set_to_none=True)
                loss = self.ce_loss(
                    self.model(inputs.to(self.device)), targets.to(self.device)
                )
                loss.backward()
                head_optimizer.step()
        self.model.extractor.load_state_dict(
            {
                name.removeprefix("extractor."): value.to(self.device)
                for name, value in body.items()
            }
        )
        for parameter in self.model.extractor.parameters():
            parameter.requires_grad_(True)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(False)
        body_optimizer = self.build_optimizer()
        total, batches = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                with torch.no_grad():
                    for parameter in self.model.extractor.parameters():
                        parameter.div_(mu)
                body_optimizer.zero_grad(set_to_none=True)
                loss = self.ce_loss(
                    self.model(inputs.to(self.device)), targets.to(self.device)
                )
                loss.backward()
                with torch.no_grad():
                    for parameter in self.model.extractor.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(mu)
                        parameter.mul_(mu)
                body_optimizer.step()
                total += loss.item()
                batches += 1
        state = clone_state(self.model.state_dict())
        return self.result(
            task,
            total / batches,
            {
                "body": {
                    name: value
                    for name, value in state.items()
                    if name.startswith("extractor.")
                },
                "head": {
                    name: value
                    for name, value in state.items()
                    if name.startswith("classifier.")
                },
            },
        )


class Server(BaseServer):
    client_class = DFedPGPClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        graph = adjacency(args)
        self.weights = (graph / graph.sum(dim=1, keepdim=True)).T
        state = clone_state(self.model.state_dict())
        self.bodies = {
            i: {k: v for k, v in state.items() if k.startswith("extractor.")}
            for i in range(self.num_clients)
        }
        self.heads = {
            i: {k: v for k, v in state.items() if k.startswith("classifier.")}
            for i in range(self.num_clients)
        }
        self.mus = torch.ones(self.num_clients)

    def training_payloads(self):
        return {
            i: {"body": self.bodies[i], "head": self.heads[i], "mu": self.mus[i]}
            for i in self.selected
        }

    def apply_results(self, results):
        for i in self.selected:
            self.bodies[i] = results[i].payload["body"]
            self.heads[i] = results[i].payload["head"]
        mixed = mix_states(
            [self.bodies[i] for i in range(self.num_clients)], self.weights, self.device
        )
        self.bodies = dict(enumerate(mixed))
        self.mus = (self.weights.to(self.device) @ self.mus.to(self.device)).cpu()
        self.client_states = {
            i: {
                **self.heads[i],
                **{k: v / self.mus[i] for k, v in self.bodies[i].items()},
            }
            for i in range(self.num_clients)
        }

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "bodies": self.bodies,
            "heads": self.heads,
            "mus": self.mus,
            "weights": self.weights,
        }
