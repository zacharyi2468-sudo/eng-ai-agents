### 1. Keep or remove batch normalization

Decide whether to keep batch normalization in the FPN and box head. Answer from your plots. Compare the difference between `no_bn` and `bn` at seed 0 with the difference between the two `bn` seeds. Make this comparison for AP and for the smoothness measurements.

**Answer:** **Keep it, but the case is weak.** At this learning rate and schedule, BN brings no accuracy benefit I can measure. It only makes optimization slightly better behaved, mostly early in training.

*AP: no detectable effect.* After 4 epochs, `bn-seed0` reaches mAP 0.1966 and `bn-seed1` reaches 0.2034, so retraining alone moves mAP by 0.0069. Removing BN at seed 0 gives 0.1961, a change of −0.0005, which is about 0.07× the seed gap. mAP@50 is 0.348 (`bn-seed0`) vs 0.335 (`no-bn-seed0`), a gap of 0.012 against a seed gap of 0.006 (2×). With one `no_bn` seed, a 2× ratio on a single metric is not evidence. `no_bn` did not diverge, and the training loss curves are nearly identical (post-warmup median total loss 0.649 / 0.662 for the BN seeds vs 0.665 without BN).

*Smoothness: small, consistent effect on the gradient, none on loss spread.* Post-warmup medians (steps ≥ 500), BN seed 0 / BN seed 1 / no BN:
- Loss spread ΔL: 0.0737 / 0.0723 / 0.0752. The arm gap (0.0015) equals the seed gap (0.0014), and `no_bn` is above `bn-seed0` at only 51% of the paired measurement steps. No difference.
- Gradient change `grad_pred`: 1.050 / 1.005 / 1.154. The arm gap (0.104) is 2.3× the seed gap, and `no_bn` is higher at 81% of paired steps (vs 49% for `bn-seed1`).
- Gradient norm: 2.14 / 2.16 / 2.32 (arm gap 7.7× the seed gap).
- Relative change `grad_pred/grad_norm`: 0.477 / 0.465 / 0.523. That is 10% higher without BN, 3.6× the seed gap, and higher at 74% of paired steps.

The difference is largest during warmup (steps 0–500). There `no_bn` has median ΔL 0.102 vs 0.060, `grad_pred` 2.16 vs 1.22, and relative change 0.60 vs 0.47. The `grad_pred` plot shows the `no_bn` gradient changing by up to ≈ 4 per step around steps 100–200, while both BN runs stay at or below ≈ 2.

*Decision.* The only differences that exceed the seed-to-seed gap favor BN, and keeping BN costs nothing in AP, so I keep it. Its measurable value here is a somewhat better-conditioned gradient, especially early, which should leave more margin for a larger step size. These runs do not show that BN improves the detector. A higher learning rate (the optional extension) would test whether that margin matters. Limits: one `no_bn` seed with no estimate of its own variance, two BN seeds that give only a rough scale, 4 epochs, and a frozen pretrained backbone, so only the FPN and box head differ.

### 2. Covariate shift or smoothness

Which explanation do your measurements support: reduced internal covariate shift, or a smoother loss landscape? For each measurement you rely on, say what it cannot show.

**Answer:** The data argue **against internal covariate shift (ICS)** as what BN buys here. They give **weak support to the smoothness account**. With neither arm failing, neither mechanism produced an accuracy difference.

**Covariate shift is large without BN, and it does not hurt.** Without BN, the block outputs that the next layers receive drift a lot. At `fpn.layer3` the output std goes 0.51 → 0.94 → 1.76 (start → step 500 → end), at `fpn.inner3` 0.52 → 1.45, and at `fpn.layer0` 2.65 → 3.83. In the box head, `conv3` moves the other way (1.46 → 0.60). In `drift_table()`, the median step-to-step change in log-std of the block output is 0.02–0.04 per measurement without BN, against ≈ 0.0002 (FPN) and 0.002–0.008 (box head) with BN, about 10–100× more. The mean shifts by 0.002–0.02 of a std per measurement vs ≈ 0 with BN. So BN does remove most of this shift. But the `no_bn` detector reaches the same training loss (0.665 vs 0.649 / 0.662) and the same mAP (0.196 vs 0.197 / 0.203) despite it. If reducing ICS were what made training work, the arm with 10–100× more shift should have trained worse, and it did not.

**BN does not stop its own input from shifting.** Before normalization (`act_std`), the BN arm's conv outputs move as much as or more than the `no_bn` ones. At `fpn.inner3`, the BN arm goes 0.52 → 2.75 (5×) vs 0.52 → 1.45 without BN. BN makes the downstream layers insensitive to that drift rather than preventing it, and BN-arm training is unaffected.

**Smoothness: a small effect in the predicted direction.** With BN the gradient is more predictive over one step: `grad_pred/grad_norm` is 0.48 vs 0.52 post-warmup (3.6× the seed gap, 74% of paired steps) and 0.47 vs 0.60 during warmup. That is the direction Santurkar et al. report. Loss spread, however, is indistinguishable after warmup (0.074 vs 0.075) and differs only during warmup (0.060 vs 0.102). So BN's smoothing is measurable mainly in the gradient and mainly early. At `lr = 0.01` the unnormalized problem is already smooth enough that it doesn't show up in loss or AP.

**What each measurement cannot show.**
- *`act_out_mean` / `act_out_std` (block output):* only the first two moments, pooled over all channels, images, positions and, in the box head, RoIs. Per-channel shifts that cancel, and changes in shape (skew, sparsity after ReLU), are invisible. In the BN arm the near-constant output std (≈ 1.0 in the FPN, ≈ 0.58 after the box-head ReLU) is largely guaranteed by the operation itself, so it is not independent evidence. Stable moments do not prove the absence of distribution shift, and moving moments do not by themselves show that shift harms optimization.
- *`drift_table` shift numbers:* measured every 25 steps, so they capture slow drift, not step-to-step changes within that window. They summarize moments, not distributions.
- *`act_std` (pre-normalization):* shows that the BN layer's input scale drifts, but because BN is invariant to that scale, it cannot say whether the drift would matter. It also does not separate weight-norm growth (which BN tolerates and weight decay counters) from a real change in the input distribution.
- *Loss spread ΔL:* one batch, one direction (the current gradient), six step lengths. It mixes slope and curvature, since a larger gradient alone makes it larger, so it is not a curvature or Lipschitz estimate, and it says nothing about directions other than −g. RPN proposals change when weights are perturbed, which adds noise.
- *`grad_pred` and `grad_pred/grad_norm`:* one step on one batch, with the same proposal-noise caveat. The raw value scales with the gradient (the `no_bn` gradient norm is also 8% larger), which is why I lean on the ratio. Neither gives a global smoothness constant.
- *The comparison as a whole:* a single paired seed, a frozen pretrained backbone, and a learning rate where both arms train fine. Nothing here manipulates covariate shift independently of BN (as Santurkar et al. did by injecting noise after BN), so the evidence is correlational. Still, a network that tolerates large measured shift with no loss in accuracy is a direct counterexample to "BN helps because it reduces ICS" in this setting.
