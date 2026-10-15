# Compute accounting (Next Steps item 2) — started 2026-10-14

Goal: replace the gate-weighted cost proxy with real FLOPs and parameters of
the structure that is present. Report two numbers for each run:

- **realisable**: the cost if the present structure were physical tensors;
- **executed**: what the gated supernet really runs (always the full dense shape).

## Definitions

- Shape: d=768, head_dim=64, FFN group = 24 channels, SwiGLU (up is 2x), tied lm_head.
- Forward matmul FLOPs per token, per present layer:
  - attention: QKV 2*d*3*h*dh + scores 2*T*h*dh + mix 2*T*h*dh + out 2*h*dh*d
    (T = seq_len; the code computes the full T x T and masks it);
  - FFN: up 2*d*2*g*s + down 2*g*s*d;
  - lm_head 2*d*V once. Norms, softmax, skips, embeddings: not counted (elementwise).
- Training FLOPs = 3 x forward.
- Phase 2 "present": state PROTECTED or MATURE, or DEAD with its death ramp
  still running. A head or FFN group counts only if its layer is present.
- Phase 1 "present": gate >= threshold (`export_architecture`).

## Steps

- [x] `compute.py`: structure spec -> FLOPs/token and params; dense reference.
- [x] Replay: seed split (`Random(seed)`, unit order of `_register_units`) + events.json
      -> structure at each step. Check against final `state_counts` in growth_log.json.
- [x] Test: params formula == `CompactTransformer` / dense `SelfOrganizingTransformer` params;
      FLOPs formula == `torch.utils.flop_counter.FlopCounterMode` on a CompactTransformer.
- [ ] `train_phase2.py`: log realisable FLOPs/token and its fraction of dense at each growth check.
- [x] Quality on the same structure: val ppl of the supernet with every non-present unit forced off (not `build_from_sot`: it uses gate >= 0.5 and snaps head counts).
- [x] Report for attempts 5, 6, 7 and 8 (RESEARCH_LOG.md 2026-10-14): final inference FLOPs, cumulative training FLOPs vs dense.
- [ ] Latency (needs GPU): later.

## Review

- Replay matches `state_counts` at all 118 growth checks of attempts 5-7, and each unit's final gate band (alive >= 0.237, dormant 0.014-0.052, dead <= 0.0011).
- tests/compute_test.py 5/5: formulas equal FlopCounterMode and numel() exactly.
- Dense 768: 313.1 MFLOP/token forward, 152.1M params. lm_head is 77.2 MFLOP (25%), a fixed floor.
- Attempt 7 final: 0.473 dense FLOPs/token, 0.478 params; 0.300 of dense non-lm_head FLOPs. Training-FLOPs lower bound 0.454.
- Attempt 8 final: 0.466 dense FLOPs. Val ppl 26.67 as trained (48 batches), 132.42 with non-present units off. Attempt 7: 30.25 -> 244.29.
- Cause: 706 dormant FFN groups (attempt 7) carry most of it (-> 211.7). Dead logits drift after the ramp (dead layers ~3 ppl).
- Open: `train_phase2.py` logging (step above) waits on the dormant/dead decision, since what counts as "present" may change.

# Inactive units: exact zero effect, zero gradient — started 2026-10-15

Requirement (user): dormant and dead units have no gradient and no effect on the model. See tasks/lessons.md.

## Design
- `PlasticTransformer` buffers (persistent, default 1 so old checkpoints load unchanged): `layer_scale` [L], `head_scale` [L,H], `group_scale` [L,G].
- Forward gate = scale * sigmoid(logit) + probe_mask * probe. Scale 0 -> exact zero output -> exact zero grad on the unit's weights and logit.
- Probes: zero-valued Parameters for the birth signal (dL/dg of a dormant unit). Optimizer group lr=0, wd=0; not in grad clipping. architecture_cost/sparsity_loss use gates without probes, so the birth signal is LM loss only.
- Birth: logit -> birth_init_logit, scale ramps 0 -> 1 linearly. Death: scale ramps 1 -> 0 linearly, then exact 0. Logit trains freely in both ramps.
- Weight freeze: snapshot inactive slices before optimizer.step, restore after (weight decay and Adam momentum otherwise move them).
- Birth score is raw |dL/dg| (EMA). Smoke run: ~1e-6, vs 0.006-0.01 before, so runs use birth_score_floor 0. Within a check every dormant score is G x a constant (contribution is 0), so the ranking is by G alone.
- Skip edge i <- i-s: gate * layer_scale[i] * layer_scale[i-s]. Off edges are in the freeze set.
- Units in a DORMANT or DEAD layer: out of birth, death and medians. Layer birth: its live children restart as PROTECTED (age 0).
- Acceptance test: CompactTransformer built from the scales (gates folded in) gives the supernet's logits (~1e-5), with a dormant layer, a dead layer and skips on.
- The probe is a read-only gradient on dormant units; births rank random-init units (GradMax-style). Results are not comparable with attempts 1-8.

## Steps
- [x] model.py: `_forward_gates()` hook in forward (Phase 1 unchanged).
- [x] plastic_model.py: scales, probes, gates, inactive masks, snapshot/restore.
- [x] plasticity.py: controller drives scales; reads dormant signal from probes.
- [x] train_phase2.py: optimizer groups, clip list, zero_grad, freeze/restore.
- [x] Tests: forward invariance to inactive weights; zero grads; probe = dL/dg; weights bit-identical after a step; ramps; old checkpoint loads; structure_eval present == trained.
- [x] Update tests 1-39 for scale-based ramps.
- [x] Smoke run.

## Review (zero effect)
- zero_effect_test 9/9, controller 42/42 (40-42: dormant-layer rule; fail with the old rule), compute 5/5.
- Smoke run: supernet val loss 4.592691 = physical model 4.592691 (max batch diff 4.8e-7).
- Found during the smoke run: raw probe scores ~1e-6, so birth_score_floor must be 0 (was 0.005). Removed a sigmoid' rescale that pretended to keep the old calibration.
