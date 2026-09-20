"""FedProc 算法实现。"""

import torch

from .core import BaseClient, BaseServer


def prototype_loss(
    features: torch.Tensor, prototypes: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """计算以类别原型为分类器的余弦交叉熵损失。"""
    normalized_features = torch.nn.functional.normalize(features, dim=1)
    normalized_prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    logits = normalized_features @ normalized_prototypes.T
    return torch.nn.functional.cross_entropy(logits, targets)


def extract_prototypes(model, loader, num_classes, feature_dim, device):
    """从本地数据提取按类别平均的表征与样本数。"""
    prototypes = torch.zeros(num_classes, feature_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            features = model.extractor(inputs.to(device, non_blocking=True))
            targets = targets.to(device, non_blocking=True)
            prototypes.index_add_(0, targets, features)
            counts.index_add_(0, targets, torch.ones_like(targets, dtype=counts.dtype))
    nonempty = counts > 0
    prototypes[nonempty] /= counts[nonempty].unsqueeze(1)
    return prototypes.cpu(), counts.cpu()


def aggregate_prototypes(payloads, old, by_count):
    """聚合非空类别原型；可选择按类别样本数加权。"""
    sums = torch.zeros_like(old)
    counts = torch.zeros(old.shape[0], dtype=old.dtype)
    for payload in payloads:
        prototypes = payload["prototypes"]
        local_counts = payload["counts"].to(dtype=old.dtype)
        weights = local_counts if by_count else (local_counts > 0).to(old.dtype)
        sums.add_(prototypes * weights.unsqueeze(1))
        counts.add_(weights)
    result = old.clone()
    nonempty = counts > 0
    result[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    return result


class FedProcClient(BaseClient):
    """带全局类别原型对比约束的客户端。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.feature_dim: int = args.feature_dim

    def train(self, task):
        prototypes = task.payload["global_prototypes"].to(self.device)
        alpha = task.payload["alpha"]
        optimizer = self.build_optimizer()
        self.model.train()
        loss_sum, steps = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                features = self.model.extractor(inputs)
                logits = self.model.classifier(features)
                loss = (1.0 - alpha) * self.ce_loss(
                    logits, targets
                ) + alpha * prototype_loss(features, prototypes, targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
        local_prototypes, counts = extract_prototypes(
            self.model,
            self.train_loader(task.client_id),
            self.num_classes,
            self.feature_dim,
            self.device,
        )
        return self.result(
            task,
            loss_sum / steps,
            {"prototypes": local_prototypes, "counts": counts},
        )


class Server(BaseServer):
    """FedProc 服务端，维护全局类别原型。"""

    client_class = FedProcClient

    def __init__(self, args) -> None:
        super().__init__(args)
        self.feature_dim: int = args.feature_dim
        self.global_prototypes = torch.zeros(self.num_classes, self.feature_dim)
        self.alpha = 1.0

    def training_payloads(self):
        self.alpha = 1.0 - len(self.accuracies) / self.num_rounds
        return {
            client_id: {
                "global_prototypes": self.global_prototypes.clone(),
                "alpha": self.alpha,
            }
            for client_id in self.selected
        }

    def apply_results(self, results):
        """按 FedAvg 权重聚合模型，并更新全局类别原型。"""
        self.aggregate_weighted(results)
        self.global_prototypes = aggregate_prototypes(
            [results[client_id].payload for client_id in self.selected],
            self.global_prototypes,
            by_count=False,
        )

    def checkpoint_state(self):
        return {"global_prototypes": self.global_prototypes.clone()}

    def progress_fields(self):
        return {"alpha": f"{self.alpha:.3f}"}
