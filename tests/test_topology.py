"""去中心化运行时的纯 CPU 测试。"""

from types import SimpleNamespace

import torch

from runtime.topology import adjacency, metropolis_hastings, mix_states, sinkhorn


def test_ring_topology_and_mh_weights_are_row_stochastic() -> None:
    args = SimpleNamespace(
        num_clients=4,
        adj_type="ring",
        edge_p=0.2,
        k_small_world=2,
        m_scale_free=1,
    )
    graph = adjacency(args)
    weights = metropolis_hastings(graph)
    assert torch.equal(graph.diag(), torch.ones(4))
    assert torch.allclose(weights.sum(dim=1), torch.ones(4))


def test_state_mixing_produces_one_independent_state_per_client() -> None:
    states = [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([3.0])}]
    mixed = mix_states(states, torch.tensor([[0.75, 0.25], [0.5, 0.5]]), torch.device("cpu"))
    assert torch.equal(mixed[0]["weight"], torch.tensor([1.5]))
    assert torch.equal(mixed[1]["weight"], torch.tensor([2.0]))
    assert mixed[0]["weight"].data_ptr() != states[0]["weight"].data_ptr()


def test_sinkhorn_is_nearly_doubly_stochastic() -> None:
    weights = sinkhorn(torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-4)
    assert torch.allclose(weights.sum(dim=0), torch.ones(2), atol=1e-4)


def test_random_and_scale_free_topologies_are_connected() -> None:
    """随机重试和 BA 初始核不应产生孤立客户端。"""
    for kind in ("random", "scale_free", "small_world"):
        args = SimpleNamespace(
            num_clients=10,
            adj_type=kind,
            edge_p=0.2,
            k_small_world=4,
            m_scale_free=2,
        )
        graph = adjacency(args)
        reached = torch.zeros(10, dtype=torch.bool)
        reached[0] = True
        for _ in range(10):
            reached |= graph[reached].sum(dim=0).bool()
        assert reached.all()
