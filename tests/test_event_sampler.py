from __future__ import annotations

from datasets.samplers import EventEpochSampler


def test_event_epoch_sampler_visits_every_index_exactly_once() -> None:
    event_ids = ["event_a", "event_b", "event_c", "event_d"]
    sampler = EventEpochSampler(event_ids, seed=17)
    first = list(iter(sampler))
    assert sorted(first) == list(range(len(event_ids)))
    assert len(first) == len(set(first))

    sampler.set_epoch(1)
    second = list(iter(sampler))
    assert sorted(second) == list(range(len(event_ids)))
    assert first != second


def test_event_epoch_sampler_spreads_repeated_events_without_omission() -> None:
    event_ids = ["event_a", "event_a", "event_a", "event_b", "event_c"]
    sampler = EventEpochSampler(event_ids, seed=3)
    order = list(iter(sampler))
    assert sorted(order) == list(range(len(event_ids)))
    first_round = [event_ids[index] for index in order[:3]]
    assert set(first_round) == {"event_a", "event_b", "event_c"}
