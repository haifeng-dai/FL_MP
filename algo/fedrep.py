"""FedRep 算法实现。"""

from .core import BaseClient, BaseServer
from .fedper import aggregate_states


def add_arguments(parser):
    """注册分类头本地训练轮数。"""
    parser.add_argument("--epochs-head", type=int, default=1)


class FedRepClient(BaseClient):
    """先训练私有分类头，再训练共享特征提取器。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.epochs_head: int = args.epochs_head

    def train(self, task):
        optimizer = self.build_optimizer()
        self.model.train()
        for parameter in self.model.extractor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(True)
        for _ in range(self.epochs_head):
            for inputs, targets in self.train_loader(task.client_id):
                optimizer.zero_grad(set_to_none=True)
                loss = self.ce_loss(
                    self.model(inputs.to(self.device, non_blocking=True)),
                    targets.to(self.device, non_blocking=True),
                )
                loss.backward()
                optimizer.step()

        for parameter in self.model.extractor.parameters():
            parameter.requires_grad_(True)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(False)
        loss_sum, batches = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                optimizer.zero_grad(set_to_none=True)
                loss = self.ce_loss(
                    self.model(inputs.to(self.device, non_blocking=True)),
                    targets.to(self.device, non_blocking=True),
                )
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                batches += 1
        return self.result(task, loss_sum / batches)


class Server(BaseServer):
    """仅聚合 body，并保持客户端私有 head。"""

    client_class = FedRepClient
    pfl = True

    def apply_results(self, results):
        self.update_client_states(results)
        total = sum(results[client_id].num_samples for client_id in self.selected)
        weights = [
            results[client_id].num_samples / total for client_id in self.selected
        ]
        bodies = [
            {
                name.removeprefix("extractor."): value
                for name, value in results[client_id].state.items()
                if name.startswith("extractor.")
            }
            for client_id in self.selected
        ]
        body = aggregate_states(bodies, weights)
        self.model.extractor.load_state_dict(body)
        for state in self.client_states.values():
            for name, value in body.items():
                state[f"extractor.{name}"] = value.clone()
