"""FedDyn 算法实现。"""

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..core import BaseClient, BaseServer


def add_arguments(parser):
    """注册 FedDyn 私有超参数。"""
    parser.add_argument("--alpha-coef", type=float, default=0.01, help="动态正则系数")


class FedDynClient(BaseClient):
    """带动态线性与二次正则项的客户端。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.alpha_coef: float = args.alpha_coef

    def train(self, task):
        previous_gradient = task.payload["local_gradient"].to(self.device)
        global_vector = task.payload["global_vector"].to(self.device)
        optimizer = self.build_optimizer()
        self.model.train()
        loss_sum, steps = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                loss_ce = self.ce_loss(self.model(inputs), targets)
                parameters = parameters_to_vector(self.model.parameters())
                loss_linear = -torch.dot(previous_gradient, parameters)
                loss_quadratic = (
                    self.alpha_coef / 2.0 * torch.sum((parameters - global_vector) ** 2)
                )
                loss = loss_ce + loss_linear + loss_quadratic
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
        return self.result(task, loss_sum / steps)


class Server(BaseServer):
    """FedDyn 服务端，维护全局和按客户端区分的梯度历史。"""

    client_class = FedDynClient

    def __init__(self, args):
        super().__init__(args)
        self.alpha_coef: float = args.alpha_coef
        vector = parameters_to_vector(self.model.parameters()).detach().cpu()
        self.h = torch.zeros_like(vector)
        self.local_gradients = {
            client_id: torch.zeros_like(vector) for client_id in range(self.num_clients)
        }

    def training_payloads(self):
        self.model.cpu()
        global_vector = parameters_to_vector(self.model.parameters()).detach().clone()
        return {
            client_id: {
                "local_gradient": self.local_gradients[client_id].clone(),
                "global_vector": global_vector.clone(),
            }
            for client_id in self.selected
        }

    def apply_results(self, results):
        """更新客户端梯度历史、全局历史梯度和全局模型。"""
        self.model.cpu()
        global_vector = parameters_to_vector(self.model.parameters()).detach().clone()
        client_vectors = []
        for client_id in self.selected:
            self.model.load_state_dict(results[client_id].state)
            vector = parameters_to_vector(self.model.parameters()).detach().clone()
            client_vectors.append(vector)
            self.local_gradients[client_id].sub_(
                vector - global_vector, alpha=self.alpha_coef
            )
        average = torch.stack(client_vectors).mean(dim=0)
        self.h.sub_(
            average - global_vector,
            alpha=self.alpha_coef * len(self.selected) / self.num_clients,
        )
        vector_to_parameters(
            average - self.h / self.alpha_coef, self.model.parameters()
        )

    def checkpoint_state(self):
        return {"h": self.h.clone(), "local_gradients": self.local_gradients}
