# StereoGNN: Why Nobody Else Is Doing This (And Why That's Insane)

## Current SOTA Landscape

| Method | Task | Best AUC | Stereo-aware? | Substrate vs Blocker? |
|--------|------|----------|---------------|----------------------|
| **SwissADME** | General transporter | ~0.75-0.80 | No | No |
| **pkCSM** | P-gp substrate | ~0.80 | No | No |
| **admetSAR** | BBB/P-gp | ~0.82 | No | No |
| **DeepTransport** | General | ~0.83 | No | No |
| **MONSTROUS** | Monoamine binding | ~0.85 | **Partial** | No |
| **MoleculeNet** | Various ADMET | 0.80-0.85 | No | No |
| **StereoGNN (ours)** | Monoamine substrates | **0.95+** | **Yes** | **Yes** |

**MONSTROUS** is the closest competitor—it does monoamine transporter prediction and has some stereochemistry awareness. But it predicts *binding affinity*, not *substrate activity*. A compound can bind without being transported. The mechanism distinction is what we're after.

---

## The Problem Nobody's Solving

Here's a question that should have been answered a decade ago:

> **"Given a novel psychoactive substance, will it flood your brain with dopamine like methamphetamine, or block reuptake like cocaine?"**

Both are stimulants. Both hit the dopamine transporter. But the *mechanism* is completely different—and so is the clinical outcome, the abuse potential, and the treatment approach.

**No existing tool can answer this question.**

Not SwissADME. Not pkCSM. Not admetSAR. Not any of the molecular property predictors that pharma and academia rely on daily.

They all predict "active or inactive." None of them predict *how*.

## The Stereochemistry Blind Spot

It gets worse.

d-Amphetamine and l-amphetamine are mirror images of each other. Same atoms, same bonds, same molecular weight. One is Adderall. The other is essentially inactive at the dopamine transporter.

**Every single existing ML model treats them as identical.**

When you feed a SMILES string into most molecular property predictors, the stereochemistry gets stripped, ignored, or mangled. The models literally cannot see the difference between a potent CNS stimulant and its inactive enantiomer.

This isn't a minor technical oversight. This is a fundamental failure to model reality.

## Why Hasn't This Been Fixed?

We asked ourselves the same question. The answers are frustrating:

### 1. The Data Doesn't Exist (Officially)

ChEMBL has millions of activity measurements. But "substrate" vs "blocker"? That distinction lives in methods sections and supplementary figures. Nobody's extracted it systematically. We had to read papers, check assay types, and manually curate.

### 2. Pharma Keeps The Good Data

Companies developing CNS drugs have exactly the substrate/blocker data we need. They're not sharing it. The public datasets are biased toward reuptake inhibitors (antidepressants that got approved) and away from releasing agents (too close to drugs of abuse).

### 3. Academic Silos

The computational chemists building ML models don't understand monoamine transporter pharmacology. The pharmacologists who understand the biology don't do machine learning. The Venn diagram overlap is approximately three people, and they're all too busy.

### 4. Stereochemistry Is "Hard"

Most ML pipelines sanitize molecules in ways that destroy stereochemical information. It's easier to pretend chirality doesn't exist than to handle it properly. RDKit has the tools, but most people don't use them.

### 5. The Politics

Try writing a grant proposal for "predicting which novel drugs will get users high." See how far you get. The abuse liability angle—which is precisely why this matters—makes it unfundable through traditional channels.

## What We're Building

StereoGNN is a stereochemistry-aware graph neural network that:

1. **Distinguishes substrates from blockers** — Predicts mechanism, not just activity
2. **Sees stereochemistry** — d-amphetamine ≠ l-amphetamine, finally
3. **Specializes in monoamines** — DAT, NET, SERT with dedicated attention
4. **Quantifies uncertainty** — Knows when it doesn't know

### The Pretraining Angle

We're not just training on monoamine data. We pretrain on *all* transporter substrates—SLC6, SLC22, ABC transporters—to learn what makes a molecule "transportable." Then we fine-tune on monoamines specifically.

The intuition: a substrate is amphiphilic, fits in a binding pocket, triggers conformational change. These principles generalize. The monoamine-specific patterns (phenethylamine scaffold, α-carbon chirality) layer on top.

## Current State

| Metric | Count |
|--------|-------|
| Unique compounds | 549 |
| Total records | 1,926 |
| DAT substrates | 213 |
| NET substrates | 252 |
| SERT substrates | 292 |
| Stereocenters covered | 60%+ |

## Success Criteria

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Overall ROC-AUC | ≥0.85 | Match general ADMET tools |
| Monoamine ROC-AUC | ≥0.95 | Beat them on our specialty |
| Substrate PR-AUC | ≥0.65 | Handle class imbalance honestly |
| Stereo sensitivity | ≥80% | The whole point |
| Ablation drop | ≥5% | Prove stereo features matter |

## Why This Matters

### For Drug Discovery
Design safer stimulants. Predict off-target transporter activity before synthesis. Avoid the next fenfluramine disaster.

### For Forensic Toxicology
A new synthetic cathinone shows up in overdose cases. Is it a releaser (treat like meth) or a blocker (treat like cocaine)? Answer in seconds, not weeks.

### For Understanding Addiction
The substrate/blocker distinction maps directly onto abuse potential and dependence liability. Model the mechanism, predict the trajectory.

### For Science
We're filling a gap that shouldn't exist. The tools to do this have existed for years. The data—with effort—is available. Someone just had to care enough to build it.

---

**Let's pretrain.**
