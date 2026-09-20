"""FedPLN 算法实现。"""

import torch

from dataset import make_loader

from .core import BaseClient, BaseServer, clone_state


def add_arguments(parser):
    """注册 FedPLN 私有超参数。"""
    parser.add_argument("--lambda", dest="lambda_", type=float, default=10.0)
    parser.add_argument("--epoch-pln", type=int, default=10)
    parser.add_argument("--lr-pln", type=float, default=0.01)
    parser.add_argument("--batch-size-pln", type=int, default=64)
    parser.add_argument("--depth-pln", type=int, default=1)
    parser.add_argument("--width-pln", type=int, default=512)
    parser.add_argument("--fixed-proto", type=int, choices=(0, 1), default=0)
    parser.add_argument("--init-emb", type=int, choices=range(8), default=0)
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--har", type=int, choices=(0, 1), default=0)


class PLN(torch.nn.Module):
    """从类别 ID 生成 feature_dim 维全局原型的网络。"""

    def __init__(
        self,
        num_classes: int,
        width: int,
        feature_dim: int,
        depth: int,
        fixed: bool,
        init_emb: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth_pln 必须至少为 1")
        self.embeddings = torch.nn.Embedding(num_classes, width)
        self.initialize_embeddings(init_emb)
        self.embeddings.weight.requires_grad_(not fixed)
        self.middle = torch.nn.Sequential(
            *[
                torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.ReLU())
                for _ in range(depth)
            ]
        )
        self.fc = torch.nn.Linear(width, feature_dim)

    def initialize_embeddings(self, init_emb: int) -> None:
        """应用与 Ray 版本一致的类别嵌入初始化策略。"""
        initializers = {
            0: lambda: None,
            1: lambda: torch.nn.init.uniform_(self.embeddings.weight, -0.1, 0.1),
            2: lambda: torch.nn.init.normal_(self.embeddings.weight, 0.0, 0.1),
            3: lambda: torch.nn.init.normal_(self.embeddings.weight, 0.0, 0.01),
            4: lambda: torch.nn.init.xavier_uniform_(self.embeddings.weight),
            5: lambda: torch.nn.init.xavier_normal_(self.embeddings.weight),
            6: lambda: torch.nn.init.kaiming_uniform_(
                self.embeddings.weight, nonlinearity="linear"
            ),
            7: lambda: torch.nn.init.orthogonal_(self.embeddings.weight),
        }
        initializers[init_emb]()

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.fc(self.middle(self.embeddings(class_ids)))


def dist_contrastive_loss(features, prototypes, targets, margin=0.0):
    """以欧氏距离负值作为分类 logits 的原型对比损失。"""
    distances = torch.cdist(features, prototypes.to(features.device), p=2.0)
    if margin:
        distances = (
            distances
            + torch.nn.functional.one_hot(targets, prototypes.shape[0]).to(distances)
            * margin
        )
    return torch.nn.functional.cross_entropy(-distances, targets)


def aggregate_pln_states(states, weights):
    """按客户端权重聚合 PLN 的 CPU 状态。"""
    aggregate = clone_state(states[0])
    for name, tensor in aggregate.items():
        if torch.is_floating_point(tensor):
            tensor.zero_()
            for state, weight in zip(states, weights, strict=True):
                tensor.add_(state[name], alpha=weight)
        else:
            tensor.copy_(states[0][name])
    return aggregate


class FedPLNClient(BaseClient):
    """依次训练全局模型与类别原型网络的客户端。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.lambda_: float = args.lambda_
        self.epoch_pln: int = args.epoch_pln
        self.lr_pln: float = args.lr_pln
        self.batch_size_pln: int = args.batch_size_pln
        self.pln = PLN(
            num_classes,
            args.width_pln,
            self.feature_dim,
            args.depth_pln,
            bool(args.fixed_proto),
            args.init_emb,
        ).to(self.device)
        self.all_classes = torch.arange(num_classes, device=self.device)

    def train(self, task):
        self.pln.load_state_dict(task.payload["pln_state"])
        self.model.train()
        self.pln.eval()
        with torch.no_grad():
            prototypes = self.pln(self.all_classes)
        optimizer = self.build_optimizer()
        model_loss, model_batches = 0.0, 0
        loader = self.train_loader(task.client_id)
        for _ in range(self.num_epochs):
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                features = self.model.extractor(inputs)
                logits = self.model.classifier(features)
                loss = self.ce_loss(
                    logits, targets
                ) + self.lambda_ * dist_contrastive_loss(features, prototypes, targets)
                loss.backward()
                optimizer.step()
                model_loss += loss.item()
                model_batches += 1

        self.model.eval()
        self.pln.train()
        pln_optimizer = torch.optim.SGD(
            self.pln.parameters(),
            lr=self.lr_pln,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        pln_loss, pln_batches = 0.0, 0
        pln_loader = (
            loader
            if self.batch_size_pln == self.batch_size
            else make_loader(
                self.train_sets[task.client_id], self.batch_size_pln, shuffle=True
            )
        )
        for _ in range(self.epoch_pln):
            for inputs, targets in pln_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.no_grad():
                    features = self.model.extractor(inputs)
                pln_optimizer.zero_grad(set_to_none=True)
                loss = dist_contrastive_loss(
                    features, self.pln(self.all_classes), targets
                )
                loss.backward()
                pln_optimizer.step()
                pln_loss += loss.item()
                pln_batches += 1
        return self.result(
            task,
            model_loss / model_batches,
            {
                "pln_state": clone_state(self.pln.state_dict()),
                "loss_proto": pln_loss / pln_batches,
            },
        )


class Server(BaseServer):
    """FedPLN 服务端，聚合全局分类模型和原型生成网络。"""

    client_class = FedPLNClient

    def __init__(self, args):
        super().__init__(args)
        self.pln = PLN(
            self.num_classes,
            args.width_pln,
            args.feature_dim,
            args.depth_pln,
            bool(args.fixed_proto),
            args.init_emb,
        )
        self.prototype_losses: list[float] = []
        self.prototype_accuracies: list[float] = []

    def training_payloads(self):
        return {
            client_id: {"pln_state": clone_state(self.pln.state_dict())}
            for client_id in self.selected
        }

    def apply_results(self, results):
        """按样本数同步聚合模型与 PLN 状态。"""
        self.aggregate_weighted(results)
        total = sum(results[client_id].num_samples for client_id in self.selected)
        weights = [
            results[client_id].num_samples / total for client_id in self.selected
        ]
        states = [
            results[client_id].payload["pln_state"] for client_id in self.selected
        ]
        self.pln.load_state_dict(aggregate_pln_states(states, weights))
        self.prototype_losses.append(
            sum(results[client_id].payload["loss_proto"] for client_id in self.selected)
            / len(self.selected)
        )

    @torch.no_grad()
    def evaluate_global(self):
        """同时计算分类头与 PLN 原型最近邻准确率。"""
        self.model.to(self.device).eval()
        self.pln.to(self.device).eval()
        prototypes = self.pln(torch.arange(self.num_classes, device=self.device))
        correct, prototype_correct, total = 0, 0, 0
        for inputs, targets in make_loader(
            self.test_set, self.batch_size, shuffle=False
        ):
            inputs = inputs.to(self.device, non_blocking=True)
            features = self.model.extractor(inputs)
            predictions = self.model.classifier(features).argmax(dim=1).cpu()
            prototype_predictions = (
                torch.cdist(features, prototypes).argmin(dim=1).cpu()
            )
            correct += (predictions == targets).sum().item()
            prototype_correct += (prototype_predictions == targets).sum().item()
            total += len(targets)
        self.prototype_accuracies.append(100 * prototype_correct / total)
        return 100 * correct / total

    def checkpoint_state(self):
        return {
            "pln": clone_state(self.pln.state_dict()),
            "prototype_losses": self.prototype_losses,
            "prototype_accuracies": self.prototype_accuracies,
        }

    def progress_fields(self):
        return {
            "loss_pln": f"{self.prototype_losses[-1]:.4f}",
            "acc_pln": f"{self.prototype_accuracies[-1]:.2f}%",
        }
