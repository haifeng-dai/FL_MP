"""FedLSA 算法实现。"""

import torch

from .core import BaseClient, BaseServer, clone_state


def add_arguments(parser):
    """注册 FedLSA 私有超参数。"""
    parser.add_argument("--lambda-com", type=float, default=0.1)
    parser.add_argument("--alpha-sep", type=float, default=0.02)
    parser.add_argument("--server-epochs", type=int, default=500)
    parser.add_argument("--server-lr", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.1)


def prototype_loss(features, anchors, targets, tau=1.0):
    """计算样本表征与全局语义锚点的余弦对比损失。"""
    logits = (
        torch.nn.functional.normalize(features, dim=1)
        @ torch.nn.functional.normalize(anchors, dim=1).T
    )
    return torch.nn.functional.cross_entropy(logits / tau, targets)


def separation_loss(anchors, tau=1.0):
    """惩罚不同类别语义锚点之间的过高相似度。"""
    count = len(anchors)
    normalized = torch.nn.functional.normalize(anchors, dim=1)
    similarities = normalized @ normalized.T / tau
    similarities.fill_diagonal_(float("-inf"))
    return (
        torch.logsumexp(similarities, dim=1)
        - torch.log(torch.tensor(count - 1, device=anchors.device, dtype=anchors.dtype))
    ).mean()


class FedLSAModel(torch.nn.Module):
    """将 CNN 的表征归一化后用于 FedLSA 分类和锚点学习。"""

    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model.extractor
        self.head = base_model.classifier

    def extractor(self, inputs):
        return torch.nn.functional.normalize(self.backbone(inputs), dim=1)

    def classifier(self, features):
        return self.head(features)

    def forward(self, inputs):
        return self.classifier(self.extractor(inputs))


class AnchorMapping(torch.nn.Module):
    """将类别随机向量映射为 feature_dim 维语义锚点。"""

    def __init__(self, feature_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, vectors):
        return torch.nn.functional.normalize(self.net(vectors), dim=1)


class FedLSAClient(BaseClient):
    """使用全局语义锚点训练归一化表征模型的客户端。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.model = FedLSAModel(self.model)
        self.lambda_com: float = args.lambda_com
        self.tau: float = args.tau

    def train(self, task):
        anchors = task.payload["anchors"].to(self.device)
        optimizer = self.build_optimizer()
        self.model.train()
        loss_sum, batches = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                features = self.model.extractor(inputs)
                loss = self.ce_loss(self.model.classifier(features), targets)
                loss = loss + self.lambda_com * prototype_loss(
                    features, anchors, targets, self.tau
                )
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                batches += 1
        return self.result(task, loss_sum / batches)


class Server(BaseServer):
    """FedLSA 服务端，聚合客户端模型并优化全局语义锚点。"""

    client_class = FedLSAClient

    def __init__(self, args):
        super().__init__(args)
        self.model = FedLSAModel(self.model)
        self.lambda_com: float = args.lambda_com
        self.alpha_sep: float = args.alpha_sep
        self.server_epochs: int = args.server_epochs
        self.server_lr: float = args.server_lr
        self.tau: float = args.tau
        self.random_vectors = torch.randn(
            self.num_classes, args.feature_dim, device=self.device, requires_grad=True
        )
        self.anchor_mapping = AnchorMapping(args.feature_dim).to(self.device)
        self.labels = torch.arange(self.num_classes, device=self.device)
        self.last_server_loss = 0.0

    def training_payloads(self):
        """返回本轮下发给各客户端的语义锚点。"""
        anchors = self.anchors().detach().cpu()
        return {client_id: {"anchors": anchors} for client_id in self.selected}

    def apply_results(self, results) -> None:
        """聚合客户端模型后更新全局语义锚点。"""
        self.aggregate_weighted(results)
        self.optimize_anchors()

    def anchors(self) -> torch.Tensor:
        """生成当前全局语义锚点。"""
        return self.anchor_mapping(self.random_vectors)

    def optimize_anchors(self) -> None:
        """冻结全局模型，仅优化随机向量和锚点映射。"""
        self.model.to(self.device).eval()
        self.anchor_mapping.train()
        optimizer = torch.optim.SGD(
            [self.random_vectors, *self.anchor_mapping.parameters()], lr=self.server_lr
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        try:
            for _ in range(self.server_epochs):
                anchors = self.anchors()
                classification = torch.nn.functional.cross_entropy(
                    self.model.classifier(anchors), self.labels
                )
                loss = classification + self.alpha_sep * separation_loss(
                    anchors, self.tau
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.last_server_loss = loss.item()
        finally:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)

    def checkpoint_state(self):
        return {
            "random_vectors": self.random_vectors.detach().cpu().clone(),
            "anchor_mapping": clone_state(self.anchor_mapping.state_dict()),
        }

    def progress_fields(self):
        return {"lsa_loss": f"{self.last_server_loss:.4f}"}
