"""Data-parallel helpers, run as two real processes (gloo on CPU: the same collectives NCCL runs on the two T4s)."""
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from dental_agent.training import ddp


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, case, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    r, lr, w = ddp.init_process_group(backend="gloo", timeout_minutes=2)
    assert (r, lr, w) == (rank, rank, world) and ddp.is_active()
    out = None
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 2)
    for p in lin.parameters():
        p.data.fill_(0.5)
    if case == "sum":
        if rank == 0:
            (lin(torch.ones(1, 3)).sum()).backward()
        # rank 1 has NO gradient at all (nothing to train on): it must still join the collective
        n = ddp.sync_trainable_grads(lin.parameters(), contributed=(rank == 0))
        out = (n, [p.grad.clone().tolist() for p in lin.parameters()])
    elif case == "mean":
        (lin(torch.full((1, 3), float(rank + 1))).sum()).backward()
        n = ddp.sync_trainable_grads(lin.parameters(), contributed=True, average_over_contributors=True)
        out = (n, [p.grad.clone().tolist() for p in lin.parameters()])
    elif case == "token_weighted":
        # exactness: summed per-rank loss SUMS divided by the GLOBAL valid-token count == the one-process token mean
        x = torch.arange(1, 7, dtype=torch.float32).view(6, 1).repeat(1, 3) / 10.0
        y = torch.arange(6, dtype=torch.float32).view(6, 1).repeat(1, 2)
        mine = slice(0, 2) if rank == 0 else slice(2, 6)          # unequal token counts per rank
        loss_sum = ((lin(x[mine]) - y[mine]) ** 2).sum()
        loss_sum.backward()
        tokens = ddp.all_reduce_sum([float(len(x[mine]))], torch.device("cpu"))[0]
        ddp.sync_trainable_grads(lin.parameters())
        for p in lin.parameters():
            p.grad.div_(tokens)
        out = [p.grad.clone().tolist() for p in lin.parameters()]
    elif case == "broadcast":
        torch.manual_seed(100 + rank)                       # different random init on every rank, like a fresh LoRA
        lin2 = torch.nn.Linear(3, 2)
        lin2.bias.requires_grad_(False)                     # a frozen tensor must be left alone
        before_bias = lin2.bias.clone()
        n = ddp.broadcast_trainable(lin2.parameters())
        out = (n, lin2.weight.detach().clone().tolist(), torch.equal(before_bias, lin2.bias))
    elif case == "order":
        log = []
        ddp.in_rank_order(lambda: log.append(rank))
        out = log
    queue.put((rank, out))
    ddp.shutdown()


def _run(case, world=2):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, case, q)) for r in range(world)]
    for p in procs:
        p.start()
    results = dict(q.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    return results


def test_sum_with_a_rank_that_has_no_gradients():
    res = _run("sum")
    assert res[0][0] == 1 and res[1][0] == 1                       # one contributor
    assert res[0][1] == res[1][1]                                  # identical on both ranks
    assert res[0][1][0] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_average_over_contributors():
    res = _run("mean")
    assert res[0][0] == 2 and res[0][1] == res[1][1]
    assert res[0][1][0][0] == [1.5, 1.5, 1.5]                      # mean of inputs 1 and 2


def test_token_weighted_sum_equals_single_process_gradient():
    res = _run("token_weighted")
    assert res[0] == res[1]
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 2)
    for p in lin.parameters():
        p.data.fill_(0.5)
    x = torch.arange(1, 7, dtype=torch.float32).view(6, 1).repeat(1, 3) / 10.0
    y = torch.arange(6, dtype=torch.float32).view(6, 1).repeat(1, 2)
    (((lin(x) - y) ** 2).sum() / 6).backward()
    for got, p in zip(res[0], lin.parameters()):
        assert torch.allclose(torch.tensor(got), p.grad, atol=1e-6)


def test_broadcast_makes_the_replicas_identical():
    res = _run("broadcast")
    assert res[0][0] == res[1][0] == 1
    assert res[0][1] == res[1][1]
    assert res[0][2] and res[1][2]


def test_rank_order_is_serialized():
    res = _run("order")
    assert res[0] == [0] and res[1] == [1]


def _base(n):
    from dental_agent.training.sft import ResumableShuffleSampler
    return ResumableShuffleSampler(n, seed=7)


def test_strided_batches_are_disjoint_equal_and_resumable():
    n, bs, world = 23, 2, 2
    ranks = [ddp.RankStridedBatches(_base(n), bs, r, world) for r in range(world)]
    for s in ranks:
        s.set_epoch(3)
    got = [list(s) for s in ranks]
    assert len(got[0]) == len(got[1]) == ranks[0].batches_per_rank() * bs == 10      # 23 // (2*2) = 5 batches each, tail dropped
    assert not set(got[0]) & set(got[1])
    assert len(set(got[0]) | set(got[1])) == 20
    for s, full in zip(ranks, got):
        s.set_epoch(3)
        s.skip_next(2 * bs)                                           # two per-rank batches already trained on
        assert list(s) == full[2 * bs:]
    ranks[0].set_epoch(4)
    assert list(ranks[0]) != got[0]                                   # reshuffled every epoch


def test_flags_must_agree_with_the_launch():
    with pytest.raises(SystemExit):
        ddp.require_consistent(True, {"WORLD_SIZE": "1"})
    with pytest.raises(SystemExit):
        ddp.require_consistent(False, {"WORLD_SIZE": "2"})
    ddp.require_consistent(True, {"WORLD_SIZE": "2"})
    ddp.require_consistent(False, {})


def test_everything_is_inert_without_a_process_group():
    assert not ddp.is_active() and ddp.rank() == 0 and ddp.world_size() == 1 and ddp.is_main()
    assert ddp.device_map() == "auto"
    assert ddp.all_reduce_sum([1, 2], torch.device("cpu")) == [1.0, 2.0]
    lin = torch.nn.Linear(2, 2)
    lin(torch.ones(1, 2)).sum().backward()
    before = [p.grad.clone() for p in lin.parameters()]
    assert ddp.sync_trainable_grads(lin.parameters()) == 1
    assert all(torch.equal(a, p.grad) for a, p in zip(before, lin.parameters()))
