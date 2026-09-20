"""FedALA 算法实现。"""

import torch
from torch.utils.data import Subset

from data import make_loader

from ..core import BaseClient, BaseServer


def add_arguments(parser):
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--rand-percent", type=int, default=80)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--ala-threshold", type=float, default=0.1)
    parser.add_argument("--num-pre-loss", type=int, default=10)


class FedALAClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.eta: float = args.eta
        self.rand_percent: int = args.rand_percent
        self.layer_idx: int = args.layer_idx
        self.ala_threshold: float = args.ala_threshold
        self.num_pre_loss: int = args.num_pre_loss

    def adaptive_aggregate(self, client_id, global_state, saved_weights):
        global_model = self.copy_model(global_state)
        global_model.requires_grad_(False)
        params_global = list(global_model.parameters())
        params_local = list(self.model.parameters())
        if torch.sum(params_global[0] - params_local[0]) == 0:
            return saved_weights
        if self.layer_idx:
            for local, global_ in zip(
                params_local[: -self.layer_idx], params_global[: -self.layer_idx]
            ):
                local.data.copy_(global_.data)

        train_set = self.train_sets[client_id]
        count = max(1, int(len(train_set) * self.rand_percent / 100))
        start = torch.randint(len(train_set) - count + 1, (1,)).item()
        loader = self.train_loader_for(
            Subset(train_set, range(int(start), int(start) + count))
        )
        temp = self.copy_model(self.model.state_dict())
        temp.requires_grad_(True).train()
        params_temp = list(temp.parameters())
        local = params_local[-self.layer_idx :] if self.layer_idx else params_local
        global_ = params_global[-self.layer_idx :] if self.layer_idx else params_global
        temp_params = params_temp[-self.layer_idx :] if self.layer_idx else params_temp
        weights = saved_weights or [torch.ones_like(param) for param in local]
        weights = [weight.to(self.device) for weight in weights]
        for temp_param, local_param, global_param, weight in zip(
            temp_params, local, global_, weights, strict=True
        ):
            temp_param.data.copy_(local_param + (global_param - local_param) * weight)

        losses = []
        while True:
            batch_losses = []
            for inputs, targets in loader:
                temp.zero_grad(set_to_none=True)
                loss = self.ce_loss(
                    temp(inputs.to(self.device, non_blocking=True)),
                    targets.to(self.device, non_blocking=True),
                )
                loss.backward()
                for temp_param, local_param, global_param, weight in zip(
                    temp_params, local, global_, weights, strict=True
                ):
                    assert temp_param.grad is not None
                    weight.sub_(
                        self.eta * temp_param.grad * (global_param - local_param)
                    )
                    weight.clamp_(0, 1)
                    temp_param.data.copy_(
                        local_param + (global_param - local_param) * weight
                    )
                batch_losses.append(loss.item())
            losses.append(sum(batch_losses) / len(batch_losses))
            if saved_weights is not None or (
                len(losses) > self.num_pre_loss
                and torch.tensor(losses[-self.num_pre_loss :]).std()
                < self.ala_threshold
            ):
                break
        for local_param, temp_param in zip(local, temp_params, strict=True):
            local_param.data.copy_(temp_param.data)
        return [weight.detach().cpu().clone() for weight in weights]

    def train_loader_for(self, dataset):
        return make_loader(dataset, self.batch_size, shuffle=False)

    def train(self, task):
        weights = self.adaptive_aggregate(
            task.client_id, task.payload["global_state"], task.payload["weights"]
        )
        loss = self.train_supervised(task.client_id)
        return self.result(task, loss, {"weights": weights})


class Server(BaseServer):
    client_class = FedALAClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.client_weights = {client_id: None for client_id in range(self.num_clients)}

    def training_payloads(self):
        global_state = {
            name: value.cpu().clone() for name, value in self.model.state_dict().items()
        }
        return {
            client_id: {
                "global_state": global_state,
                "weights": self.client_weights[client_id],
            }
            for client_id in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        for client_id in self.selected:
            self.client_weights[client_id] = results[client_id].payload["weights"]
        self.aggregate_weighted(results)

    def checkpoint_state(self):
        return {**super().checkpoint_state(), "client_weights": self.client_weights}
