"""FedProx 算法实现。"""

import torch

from .core import BaseClient, BaseServer


def add_arguments(parser):
    """注册 FedProx 私有超参数。"""
    parser.add_argument("--mu", type=float, default=0.01, help="近端正则系数")


class FedProxClient(BaseClient):
    """在本地监督损失中加入近端正则的 FedProx 客户端。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.mu: float = args.mu

    def train(self, task):
        """训练本地模型，使其不偏离本轮全局模型。"""
        global_parameters = [
            parameter.detach().clone() for parameter in self.model.parameters()
        ]
        self.model.train()
        optimizer = self.build_optimizer()
        total_loss, batches = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs.to(self.device))
                ce_loss = self.ce_loss(logits, targets.to(self.device))
                proximal = sum(
                    torch.sum((parameter - global_parameter) ** 2)
                    for parameter, global_parameter in zip(
                        self.model.parameters(), global_parameters
                    )
                )
                loss = ce_loss + 0.5 * self.mu * proximal
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
        return self.result(task, total_loss / batches)


class Server(BaseServer):
    """FedProx 服务端：使用标准按样本数加权聚合。"""

    client_class = FedProxClient

    def apply_results(self, results):
        self.aggregate_weighted(results)
