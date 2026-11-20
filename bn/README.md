# bn: Bayesian network for the joint model

Code for the joint Bayesian network of the merged paper (joint network + transfer + audit).
Simulated data only in this folder; nothing here reads CIS or ISSP microdata.

| File | What it does |
|---|---|
| `core.py` | Discrete Bayesian network: Dirichlet marginal likelihood, penalising modular prior `exp(-gamma*|pa|^power)` (Martins et al. 2024), tier constraints (`allowed[p, c]`), structure MCMC, conjugate parameter draws, forward sampling. |
| `missing.py` | Missing values inside the model by data augmentation: network, parameters and missing cells sampled jointly. |
| `martins_replication.py` | Replication of the simulation study of Martins, Gonçalves and Galletti (2024, arXiv:2402.17915). |
| `martins_figures.py` | Figures in the style of their Figures 2–4. |
| `check_missing.py` | Check of `missing.py` against the truth under missing-at-random data. |
| `diag_d7.py` | Mixing diagnostic: chains started from the empty and from the true network. |

## Structure sampler

Two moves are mixed:

- **Pair move.** An unordered pair {i, j} is chosen and its state (no edge, i→j, j→i) is changed. The move is symmetric, so acceptance is the posterior ratio.
- **Block Gibbs step (probability 0.3).** Two rows of the adjacency matrix, that is the parent sets of two nodes, are drawn jointly from their full conditional. This is Goudie and Mukherjee (2016) with m = 2; acyclicity is checked exactly with reachability bitmasks. When the joint table would exceed 250,000 entries, the step falls back to a single row.

The target is the same posterior as the sampler used by Martins et al.

Checks:

- On a 3-node problem, the chain matches exact enumeration of all 25 DAGs (total variation distance 0.012–0.015, which is Monte Carlo noise).
- With pair moves only, the d = 7 network mixed badly under strong penalties. Chains from the empty graph never reached a v-structure that scored 3 to 33 log units higher. After the block step, chains from the empty and from the true network agree in every case (`diag_d7.py`).

## Replication of Martins et al. (2024)

Run: `python -m bn.martins_replication --reps 10 --out bn_output` (about 25 minutes on two cores), then `python -m bn.martins_figures`. The log is in `bn_output/martins_replication.log`.

Setup:

- **Networks and sample sizes:** as in their Table 2.
- **True probabilities:** U(0.2, 0.8), redrawn until every parent changes the child's probability by at least 0.15 on average.
- **Priors:** Beta(1,1) on the parameters. P1 is the penalising prior with γ calibrated as the smallest value whose mean posterior probability of the true network exceeds 0.85; P2 is the uniform prior.
- **Network posterior:** summarised by Markov equivalence class.

Methods compared:

- **S1:** posterior predictive (their Algorithm 1), 500 draws.
- **S2:** highest-posterior network, MLE parameters, 5 datasets.
- **S3:** sequential CART synthesis, as in synthpop.

| scenario | prior | γ | reps with true class on top | mean P(true) on those | combined OIC, S1 | combined OIC, S2 |
|---|---|---|---|---|---|---|
| d=3, n=500 | P1 | 3 | 9 | 0.91 | 0.934 | 0.765 |
| d=3, n=500 | P2 | 0 | 10 | 0.60 | 0.971 | 0.841 |
| d=3, n=1000 | P1 | 3 | 9 | 0.96 | 0.965 | 0.839 |
| d=3, n=1000 | P2 | 0 | 9 | 0.63 | 0.975 | 0.763 |
| d=3, n=5000 | P1 | 1 | 10 | 0.92 | 0.970 | 0.854 |
| d=3, n=5000 | P2 | 0 | 10 | 0.81 | 0.971 | 0.843 |
| d=4, n=1000 | P1 | 2 | 10 | 0.89 | 0.968 | 0.882 |
| d=4, n=1000 | P2 | 0 | 10 | 0.52 | 0.967 | 0.820 |
| d=4, n=5000 | P1 | 2 | 10 | 0.96 | 0.971 | 0.818 |
| d=4, n=5000 | P2 | 0 | 10 | 0.74 | 0.976 | 0.897 |
| d=7, n=2000 | P1 | 14 | 10 | 0.86 | 0.959 | 0.903 |
| d=7, n=2000 | P2 | 0 | 8 | 0.17 | 0.971 | 0.810 |
| d=7, n=5000 | P1 | 8 | 9 | 0.92 | 0.960 | 0.866 |
| d=7, n=5000 | P2 | 0 | 5 | 0.25 | 0.972 | 0.853 |

What agrees with Martins et al.:

- The penalising prior concentrates the posterior on the true network, much more than the uniform prior, and the gap grows with d.
- The calibrated γ grows with d. Their Figure 1 places d = 7 at 16–30.
- S1 has the smallest error for the MLE of P(X2=1|X1=0), with mean absolute error 0.0005–0.006 against 0.002–0.011 for S2 and 0.006–0.027 for S3. Its interval contains the original value in every replication.
- The S2 intervals for the p-value of the X1–X2 test miss the original value in 10–30% of replications. This is the near-degenerate behaviour of S2 they report.

What needs a remark:

- **Per-dataset OIC.** The OIC of a single synthetic dataset is about 0.71 for S1 and about 0.80 for S2, in every scenario. This is what theory predicts. A posterior predictive dataset carries parameter and sampling variability, so its estimate differs from the original one by about √2 standard errors. A dataset drawn from the fitted model differs by about one. Applied to the original 95% interval, these give 0.71 and 0.80. Martins et al. report S1 values near 0.9, which must come from a different construction of the synthetic interval; the paper does not say which.
- **Combined OIC.** We therefore also report the OIC of the interval that a user would build from all synthetic datasets, using the combining rule for fully synthetic data, T = (1 + 1/M) b − ū (Raghunathan, Reiter and Rubin 2003). On that interval S1 reaches 0.93–0.98 and beats S2 (0.76–0.90) in every scenario, which is their qualitative conclusion.
- **Weak dependence at d = 7.** At d = 7 and n = 2000 the uniform prior sometimes prefers a reversed v-structure (node 3 with parents {4, 5} instead of node 4 with parents {3, 5}). The two networks have the same number of parameters and the data separate them weakly. This is posterior uncertainty, not a sampler failure; chains from both starts agree.

## Missing values inside the model (`check_missing.py`)

Setup:

- **Network:** 8 nodes with 2–4 categories, n = 3000.
- **Missingness:** two attitude-like variables lose values with a probability that depends on an always-observed education-like variable (missing at random).

Results under strong MAR (60% of respondents with education level 0 and 5% of the rest missing, `bn_output/check_missing_mar60.txt`):

| statistic | truth | full data | data augmentation | complete cases |
|---|---|---|---|---|
| P(worry = 2) | 0.362 | 0.368 [0.349, 0.387] | 0.371 [0.347, 0.394] | 0.304 [0.282, 0.328] |
| P(accepts) | 0.545 | 0.529 [0.509, 0.548] | 0.536 [0.515, 0.563] | 0.505 [0.478, 0.536] |

Complete-case analysis drops 59% of respondents and misses P(worry = 2) by 6 points, with the truth outside its 90% interval. Data augmentation recovers it.

Under mild MAR (`check_missing_mar30.txt`), all three agree, and complete cases give wider intervals.

Speed: about 0.4–0.65 s per sweep with 15 variables, n = 4000 and 3% missing cells, so 3000 sweeps take about half an hour on a laptop.

## Joint model on CIS 3391 and validation on ISSP (M2–M5)

| File | What it does | Data |
|---|---|---|
| `cis_joint.py` | Network over 22 CIS variables, with missing answers inside the model. Tables use the weighted pseudo-posterior (PESOFIN). Two chains, plus a complete-case comparison. | CIS microdata (local) |
| `transfer_node.py` | Adds the outcome node to the network. Its table comes from the transfer model with a hierarchical transport layer, and the projection is broken down by any network variable. | posterior of `cis_joint.py` |
| `synthetic.py` | Fully synthetic datasets with the outcome, and the combining rule for fully synthetic data, checked against the posterior predictive. | posterior of `cis_joint.py` |
| `validate_issp.py` | Retrospective validation of the joint model on ISSP 2020: leave-region-out hyperparameters estimated on same-item and on cross-item transfers, no inflation factor, and local updates by importance resampling. | ISSP microdata (local) |

Order of runs (outputs of the first and third stay local; the others are aggregate):

    python -m bn.cis_joint                     # ~20-40 min
    python -m bn.validate_issp                 # ~30-40 min; writes transport_output/hierarchical_transport.json
    python -m bn.transfer_node --hier transport_output/hierarchical_transport.json
    python -m bn.transfer_node                 # Finland-Spain transport layer, for comparison
    python -m bn.synthetic --hier transport_output/hierarchical_transport.json
