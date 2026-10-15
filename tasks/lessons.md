# Lessons

## 2026-10-15 — Dormant and dead units must have zero effect
- Mistake: Phase 2 ran 8 attempts with dormant (gate ~0.04) and dead (gate ~0.001) units still in the forward pass and still trained. Forcing them off took attempt 8 from 26.67 to 132 val ppl. Every Phase 2 quality number measured the supernet, not the structure.
- The premise needs it: only the active structure may affect the loss. Dormant/dead units get no gradient and add nothing to the output.
- Rule: before reporting a quality number for a sparse or gated model, check that the units counted as "off" really are off. Force them to exactly zero and compare. Do this before the first full-scale run, not after.
- Rule: a gate value that is small is not off. Off means an exact zero in the forward pass.
