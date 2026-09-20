"""ProxyFL：私有模型与邻居代理模型的相互蒸馏。"""

import torch

from .core import BaseClient, BaseServer, clone_state
from .core.decentralized import adjacency, mix_states
from .core.protocol import EvaluationTask


def add_arguments(parser):
    parser.add_argument("--mu", type=float, default=1.0)


class ProxyFLClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.mu: float = args.mu

    def train(self, task):
        proxy = self.copy_model(task.state)
        local = self.copy_model(task.payload["local_state"])
        proxy.train().requires_grad_(True)
        local.train().requires_grad_(True)
        proxy_optimizer = self.build_optimizer(proxy)
        local_optimizer = self.build_optimizer(local)
        total, proxy_total, batches = 0.0, 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                proxy_logits, local_logits = proxy(inputs), local(inputs)
                proxy_loss = self.ce_loss(proxy_logits, targets) + self.mu * (
                    torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(proxy_logits, dim=1),
                        torch.nn.functional.softmax(local_logits.detach(), dim=1),
                        reduction="batchmean",
                    )
                )
                local_loss = self.ce_loss(local_logits, targets) + self.mu * (
                    torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(local_logits, dim=1),
                        torch.nn.functional.softmax(proxy_logits.detach(), dim=1),
                        reduction="batchmean",
                    )
                )
                proxy_optimizer.zero_grad(set_to_none=True)
                proxy_loss.backward()
                proxy_optimizer.step()
                local_optimizer.zero_grad(set_to_none=True)
                local_loss.backward()
                local_optimizer.step()
                total += local_loss.item()
                proxy_total += proxy_loss.item()
                batches += 1
        self.model.load_state_dict(local.state_dict())
        return self.result(
            task,
            total / batches,
            {
                "proxy_state": clone_state(proxy.state_dict()),
                "proxy_loss": proxy_total / batches,
            },
        )


class Server(BaseServer):
    client_class = ProxyFLClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        graph = adjacency(args)
        self.weights = graph / graph.sum(dim=1, keepdim=True)
        initial = clone_state(self.model.state_dict())
        self.proxy_states = {
            client_id: clone_state(initial) for client_id in range(self.num_clients)
        }
        self.proxy_losses = []
        self.proxy_accuracies = []

    def training_states(self):
        mixed = mix_states(
            [self.proxy_states[client_id] for client_id in range(self.num_clients)],
            self.weights,
            self.device,
        )
        return {client_id: mixed[client_id] for client_id in self.selected}

    def training_payloads(self):
        return {
            client_id: {"local_state": self.client_states[client_id]}
            for client_id in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        for client_id in self.selected:
            self.proxy_states[client_id] = results[client_id].payload["proxy_state"]
        self.proxy_losses.append(
            sum(results[client_id].payload["proxy_loss"] for client_id in self.selected)
            / len(self.selected)
        )

    def evaluate_proxies(self):
        tasks = [
            EvaluationTask(client_id, clone_state(self.proxy_states[client_id]))
            for client_id, test_set in self.test_sets.items()
            if len(test_set)
        ]
        results = self.pool.evaluate(tasks)
        accuracies = [
            100 * result.correct / result.num_samples for result in results.values()
        ]
        if not accuracies:
            raise ValueError("代理模型评估没有可用的客户端测试样本")
        return sum(accuracies) / len(accuracies)

    def run_round(self):
        self.select_clients()
        results = self.run_clients()
        self.apply_results(results)
        loss = sum(results[client_id].loss for client_id in self.selected) / len(
            self.selected
        )
        accuracy = self.evaluate()
        self.proxy_accuracies.append(self.evaluate_proxies())
        return loss, accuracy

    def progress_fields(self):
        return {
            **super().progress_fields(),
            "proxy_loss": f"{self.proxy_losses[-1]:.4f}",
            "proxy_acc": f"{self.proxy_accuracies[-1]:.2f}%",
        }

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "proxy_states": self.proxy_states,
            "proxy_losses": self.proxy_losses,
            "proxy_accuracies": self.proxy_accuracies,
        }
