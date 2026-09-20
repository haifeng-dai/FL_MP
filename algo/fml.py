"""FML 算法实现。"""

import torch

from algo.core import BaseClient, BaseServer
from runtime import clone_state


def add_arguments(parser):
    parser.add_argument("--alpha-fml", type=float, default=1.0)
    parser.add_argument("--beta-fml", type=float, default=1.0)


class FMLClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.alpha: float = args.alpha_fml
        self.beta: float = args.beta_fml

    def train(self, task):
        global_model = self.copy_model(task.payload["global_state"])
        global_model.requires_grad_(True).train()
        local_optimizer = self.build_optimizer()
        global_optimizer = torch.optim.SGD(
            global_model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        local_sum = global_sum = 0.0
        batches = 0
        self.model.train()
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                out_g, out_l = global_model(inputs), self.model(inputs)
                loss_g = self.ce_loss(
                    out_g, targets
                ) + self.beta * torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(out_g, dim=1),
                    torch.nn.functional.softmax(out_l.detach(), dim=1),
                    reduction="batchmean",
                )
                loss_l = self.ce_loss(
                    out_l, targets
                ) + self.alpha * torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(out_l, dim=1),
                    torch.nn.functional.softmax(out_g.detach(), dim=1),
                    reduction="batchmean",
                )
                global_optimizer.zero_grad()
                loss_g.backward()
                global_optimizer.step()
                local_optimizer.zero_grad()
                loss_l.backward()
                local_optimizer.step()
                global_sum += loss_g.item()
                local_sum += loss_l.item()
                batches += 1
        return self.result(
            task,
            local_sum / batches,
            {
                "global_state": clone_state(global_model.state_dict()),
                "loss_global": global_sum / batches,
            },
        )


class Server(BaseServer):
    client_class = FMLClient
    pfl = True

    def training_payloads(self):
        state = clone_state(self.model.state_dict())
        return {client_id: {"global_state": state} for client_id in self.selected}

    def apply_results(self, results):
        self.update_client_states(results)
        total = sum(results[c].num_samples for c in self.selected)
        aggregate = clone_state(self.model.state_dict())
        for name, value in aggregate.items():
            if torch.is_floating_point(value):
                value.zero_()
                for client_id in self.selected:
                    value.add_(
                        results[client_id].payload["global_state"][name],
                        alpha=results[client_id].num_samples / total,
                    )
        self.model.load_state_dict(aggregate)
