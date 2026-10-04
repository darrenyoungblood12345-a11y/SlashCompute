import mlx.core as mx
import pytest
from mlx_lm.utils import load_model

from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.pipeline.data import load_examples, make_batch
from slashcompute.pipeline.local import build_compute, run_local_pipeline
from slashcompute.pipeline.model_profile import profile_model
from slashcompute.pipeline.shard import load_shard
from slashcompute.pipeline.stage import merge_checkpoints


def _spec(model, data, **kw):
    base = dict(model=str(model), dataset_path=str(data), steps=4, batch_size=4, microbatches=2,
                learning_rate=1e-2, lora_rank=4, max_seq_len=32, seed=7)
    return LoraFinetuneSpec(**(base | kw))


def test_profile(tiny_model):
    p = profile_model(str(tiny_model))
    assert p.num_layers == 6 and p.tie_word_embeddings
    assert all(b > 0 for b in p.layer_bytes)
    assert p.total_weight_bytes == sum(p.layer_bytes) + p.embed_bytes + p.head_bytes


def test_shards_compose_to_full_model(tiny_model):
    full, _ = load_model(tiny_model)
    x = mx.array([[1, 2, 3, 4, 5, 6]])
    ref = full(x)
    a, _ = load_shard(tiny_model, 0, 2)
    b, _ = load_shard(tiny_model, 2, 5)
    c, _ = load_shard(tiny_model, 5, 6)
    assert a.embed_tokens is not None and c.embed_tokens is not None  # tied head
    assert b.embed_tokens is None and b.norm is None
    out = c(b(a(x)))
    assert mx.allclose(out, ref, atol=1e-5).item()


def test_batches_deterministic(tiny_model, tiny_dataset):
    ex = load_examples(tiny_dataset, tiny_model, 32)
    b1, b2 = make_batch(ex, 3, 4, 0), make_batch(ex, 3, 4, 0)
    assert mx.array_equal(b1.inputs, b2.inputs).item()
    other = [make_batch(ex, s, 4, 0).inputs for s in range(4, 8)]
    assert any(o.shape != b1.inputs.shape or not mx.array_equal(o, b1.inputs).item() for o in other)
    assert b1.ntoks == int(b1.mask.sum().item()) > 0


def test_examples_with_completion_truncated_away_are_dropped(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"tokens": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], "loss_start": 15}\n'
                    '{"tokens": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], "loss_start": 5}\n')
    ex = load_examples(path, None, 8)
    assert [e.loss_start for e in ex] == [5]
    assert make_batch(ex, 0, 1, 0).mask.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]
    path.write_text('{"tokens": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], "loss_start": 15}\n')
    with pytest.raises(ValueError, match="no usable examples"):
        load_examples(path, None, 8)


async def test_pipeline_matches_single_stage_reference(tiny_model, tiny_dataset, tmp_path):
    spec = _spec(tiny_model, tiny_dataset)
    ref = await run_local_pipeline(spec, [0, 6], tmp_path / "ref")
    two = await run_local_pipeline(spec, [0, 3, 6], tmp_path / "two")
    three = await run_local_pipeline(spec, [0, 1, 4, 6], tmp_path / "three")

    ref_losses = [s.loss for s in ref[0]]
    assert len(ref_losses) == 4
    assert ref_losses[-1] < ref_losses[0]  # it learns
    for got in (two, three):
        last = max(got)
        losses = [s.loss for s in got[last]]
        assert losses == pytest.approx(ref_losses, rel=1e-4, abs=1e-5)
        # chain check: each stage's output digest is the next stage's input digest
        for i in range(last):
            for a, b in zip(got[i], got[i + 1]):
                assert a.out_digest == b.in_digest


async def test_resume_from_merged_checkpoint_with_new_partition(tiny_model, tiny_dataset, tmp_path):
    spec = _spec(tiny_model, tiny_dataset, steps=4)
    full = await run_local_pipeline(spec, [0, 3, 6], tmp_path / "full", checkpoint_every=2)

    # Checkpoint at step 2 from a 2-stage run, merged, then resumed on 3 stages.
    merged = tmp_path / "merged.safetensors"
    merge_checkpoints([tmp_path / "full/stage0/step_000002.safetensors",
                       tmp_path / "full/stage1/step_000002.safetensors"], merged)
    resumed = await run_local_pipeline(spec, [0, 2, 4, 6], tmp_path / "resumed",
                                       start_step=2, resume_from=merged, checkpoint_every=2)
    assert [s.step for s in resumed[2]] == [3, 4]
    assert [s.loss for s in resumed[2]] == pytest.approx([s.loss for s in full[1][2:]], rel=1e-4)


def test_checkpoint_missing_layers_rejected(tiny_model, tiny_dataset, tmp_path):
    spec = _spec(tiny_model, tiny_dataset)
    a = build_compute(spec, 0, 3, 6)
    a.save_checkpoint(tmp_path / "a.safetensors", 1)
    b = build_compute(spec, 2, 6, 6)
    with pytest.raises(ValueError):
        b.load_checkpoint(tmp_path / "a.safetensors")
