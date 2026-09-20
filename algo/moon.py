"""MOON 算法实现。"""

from __future__ import annotations

import torch

from algo.core import BaseClient, BaseServer
from runtime import clone_state


def add_arguments(parser):
    """注册 MOON 私有超参数。"""
    parser.add_argument("--mu", type=float, default=1.0, help="模型对比损失权重")
    parser.add_argument("--tau", type=float, default=0.5, help="模型对比温度")


class MoonClient(BaseClient):
    """使用全局模型和历史本地模型进行对比学习的客户端。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.mu = args.mu
        self.tau = args.tau

    def train(self, task):
        previous = self.copy_model(task.payload["previous_state"])
        global_model = self.copy_model(task.state)
        optimizer = self.build_optimizer()
        similarity = torch.nn.CosineSimilarity(dim=1)
        self.model.train()
        loss_sum, steps = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                features = self.model.extractor(inputs)
                logits = self.model.classifier(features)
                with torch.no_grad():
                    global_features = global_model.extractor(inputs)
                    previous_features = previous.extractor(inputs)
                contrastive_logits = (
                    torch.stack(
                        (
                            similarity(features, global_features),
                            similarity(features, previous_features),
                        ),
                        dim=1,
                    )
                    / self.tau
                )
                labels = torch.zeros(len(targets), device=self.device, dtype=torch.long)
                loss = self.ce_loss(logits, targets) + self.mu * self.ce_loss(
                    contrastive_logits, labels
                )
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
        return self.result(task, loss_sum / steps)


class Server(BaseServer):
    """MOON 服务端，保存每个客户端上一轮的模型状态。"""

    client_class = MoonClient

    def __init__(self, args):
        super().__init__(args)
        initial = clone_state(self.model.state_dict())
        self.client_states = {
            client_id: clone_state(initial) for client_id in range(self.num_clients)
        }

    def training_payloads(self):
        return {
            client_id: {"previous_state": clone_state(self.client_states[client_id])}
            for client_id in self.selected
        }

    def apply_results(self, results):
        """保存参与客户端的历史模型并完成标准全局聚合。"""
        for client_id, result in results.items():
            self.client_states[client_id] = clone_state(result.state)
        self.aggregate_weighted(results)

    def checkpoint_state(self):
        return {"client_states": self.client_states}
