import torch

from datasets.samplers import BalancedRemainderBatchSampler


def test_remainder_batches_are_balanced_and_cover_every_index():
    sampler = torch.utils.data.SequentialSampler(range(105))
    batches = list(BalancedRemainderBatchSampler(sampler, batch_size=16))
    sizes = [len(batch) for batch in batches]
    assert len(batches) == 7
    assert max(sizes) - min(sizes) <= 1
    assert min(sizes) > 1
    assert sorted(index for batch in batches for index in batch) == list(range(105))
