"""SCAFFOLD 算法实现。"""

import torch

from .core import BaseClient, BaseServer, clone_state


def add_arguments(parser):
    """注册 SCAFFOLD 私有超参数。"""
    parser.add_argument("--global-lr", type=float, default=1.0, help="服务端聚合学习率")


class ScaffoldClient(BaseClient):
    """带控制变量校正的本地 SGD 客户端。"""

    def train(self, task):
        payload: dict[str, dict[str, torch.Tensor]] = task.payload
        global_control = {
            name: value.to(self.device)
            for name, value in payload["global_control"].items()
        }
        local_control = {
            name: value.to(self.device)
            for name, value in payload["local_control"].items()
        }
        global_parameters = {
            name: value.to(self.device) for name, value in task.state.items()
        }
        optimizer = self.build_optimizer()
        self.model.train()
        loss_sum, steps = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs.to(self.device, non_blocking=True))
                loss = self.ce_loss(outputs, targets.to(self.device, non_blocking=True))
                loss.backward()
                with torch.no_grad():
                    for name, parameter in self.model.named_parameters():
                        parameter.add_(
                            local_control[name] - global_control[name], alpha=self.lr
                        )
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
        new_local: dict[str, torch.Tensor] = {}
        delta: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            scale = 1.0 / (steps * self.lr)
            for name, parameter in self.model.named_parameters():
                updated = (
                    local_control[name]
                    - global_control[name]
                    + (global_parameters[name] - parameter) * scale
                )
                new_local[name] = updated.detach().cpu().clone()
                delta[name] = (updated - local_control[name]).detach().cpu().clone()
        return self.result(
            task,
            loss_sum / steps,
            {"local_control": new_local, "control_delta": delta},
        )


class Server(BaseServer):
    """SCAFFOLD 服务端，维护全局及各客户端控制变量。"""

    client_class = ScaffoldClient

    def __init__(self, args):
        super().__init__(args)
        self.global_lr: float = args.global_lr
        self.global_control = {
            name: torch.zeros_like(parameter, device="cpu")
            for name, parameter in self.model.named_parameters()
        }
        self.local_controls = {
            client_id: clone_state(self.global_control)
            for client_id in range(self.num_clients)
        }

    def training_payloads(self):
        return {
            client_id: {
                "global_control": clone_state(self.global_control),
                "local_control": clone_state(self.local_controls[client_id]),
            }
            for client_id in self.selected
        }

    def apply_results(self, results):
        """聚合本轮模型，并同步更新全局与本地控制变量。"""
        previous = clone_state(self.model.state_dict())
        weights = {client_id: 1.0 / len(self.selected) for client_id in self.selected}
        self.aggregate_weighted(results, weights)
        if self.global_lr != 1.0:
            blended = clone_state(self.model.state_dict())
            for name, value in blended.items():
                if torch.is_floating_point(value):
                    value.mul_(self.global_lr).add_(
                        previous[name], alpha=1.0 - self.global_lr
                    )
            self.model.load_state_dict(blended)
        for client_id in self.selected:
            payload = results[client_id].payload
            self.local_controls[client_id] = payload["local_control"]
            for name, delta in payload["control_delta"].items():
                self.global_control[name].add_(delta, alpha=1.0 / self.num_clients)

    def checkpoint_state(self):
        return {
            "global_control": clone_state(self.global_control),
            "local_controls": {
                client_id: clone_state(state)
                for client_id, state in self.local_controls.items()
            },
        }
