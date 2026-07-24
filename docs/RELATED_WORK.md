# Related Work & Positioning

Running bibliography. Every entry notes *what it does* and *how EDC differs*. arXiv IDs are the
canonical handle; verify each in `paper/references.bib` before submission.

## Energy-based reasoning (the base architecture — reused, not claimed)

- **IREM** `2206.15448` — Learning Iterative Reasoning through Energy Minimization. Reason by
  gradient descent on `E(x, y)`; adapts steps to difficulty; OOD size generalisation.
- **IRED** `2406.11179` — Learning Iterative Reasoning through Energy Diffusion. Annealed
  landscapes (coarse→sharp) + score+energy supervision; Sudoku/matrices/pathfinding.
- **Compositional Energy Minimization** `2510.20607` — composing energy constraints.
- *EDC difference:* we do not claim the reasoner. We claim the **geometry of its inference-time
  descent** as a calibrated reliability signal.

## Energy-Based Transformers (the nearest capability threat)

- **EBT** `2507.02092` — Energy-Based Transformers are Scalable Learners and Thinkers. Scalar
  energy as confidence; "think longer" via more descent steps; best-of-N by energy.
- *EDC difference:* EBT uses **no restarts, no basin/curvature geometry, and no calibration
  guarantee**. Our signal is the basin *shape and cross-restart agreement*, not the scalar
  energy — and EBT itself notes best-of-N-by-energy can pick a worse answer, which restart
  geometry is designed to catch. This is the baseline our falsification test must beat.

## EBMs for uncertainty / OOD / verification (threats to distinguish)

- **Energy-based OOD detection** `2010.03759` — the founding energy-score detector (scalar).
- **Revisiting EBM for OOD** `2412.03058`; **Bounded/uniform energy OOD for graphs** `2504.13429`
  (bounds, but graph-only, static).
- **Distributional EBMs for structured LLM reasoning** `2605.18871` — ensemble std over LoRA
  adapters, hard-threshold abstention. Closest "EBM + UQ + abstention."
- **Energy landscapes enable reliable abstention** `2509.04482` — static learned energy
  threshold (RAG/healthcare).
- *EDC difference:* all use a **scalar** (energy value or ensemble std) with no distribution-free
  selective guarantee and no inference-landscape geometry.

## Curvature ↔ calibration (protects our novelty)

- **Too Sharp, Too Sure** `2604.20614` — curvature/sharpness predicts calibration, but in
  **weight space** (loss over parameters; a generalisation story).
- **SAM** `2010.01412`, **PyHessian** `1912.07145` — sharpness/curvature tooling over weights.
- *EDC difference:* our curvature is of the **energy over the latent `z`, per input, at fixed
  weights** — a distinct mathematical object never used as an uncertainty signal.

## Conformal prediction & risk control (our guarantee machinery)

- **A Gentle Intro to Conformal Prediction** `2107.07511`.
- **Conformal Risk Control** `2208.02814` — monotone bounded risk → used for **halting**.
- **Learn-then-Test** `2110.01052` — FWER-controlled selection for **non-monotone** risks →
  used for **selective error / abstention**.
- **Selective Conformal Risk Control** `2512.12844` — modern selective-risk variant.
- **Conformal Thinking** `2602.03814` — adaptive halting for token-budget LLMs via risk control
  (no EBM, no landscape). *EDC difference:* we halt an **energy descent** on a **geometry**
  certificate, not a token budget.
- **Conformal Abstention for LLMs** `2405.01563`.

## Selective prediction & adaptive computation

- **SelectiveNet** `1901.09192`; **Selective Classification for DNNs** `1705.08500`.
- **ACT** `1603.08983`; **PonderNet** `2107.05407` — adaptive computation, but no guarantee and
  no energy landscape.

## UQ baselines (to beat, all conformalized identically)

- **Deep Ensembles** `1612.01474`; **MC-Dropout** `1506.02142`; **Temperature Scaling**
  `1706.04599`.

## One-sentence positioning

> No prior work turns the **per-input geometry of an EBM reasoner's inference-time energy
> descent** into a **distribution-free selective-prediction and adaptive-halting certificate** —
> and shows that geometry beats the scalar energy EBT already uses.
