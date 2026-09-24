# PNDS Mathematical Specification v0.2

**Date:** 2026-09-24  
**Status:** Formal research specification; empirical claims remain bounded by the result ledgers.

## 0. Scope and notation

This specification uses one symbol for one role.

| Symbol | Meaning |
|---|---|
| q | current query/task |
| h | raw accumulated history |
| n | number of persistent candidate objects |
| k | number of candidates selected for execution |
| s | persistent state |
| x_i | persistent object i |
| r_i | router score |
| p_i | routing probability |
| I | selected candidate index set |
| g | executable graph/gating structure |
| d | typed decision |
| a | action/computation |
| o | observed outcome |
| v | verification result |
| e | experimental state |
| p | registered persistent state |
| Q | quarantine operator/status |
| Reg | registration operator |
| τ | router relaxation temperature |
| η | verifier threshold |
| κ | cost |
| ℓ | accumulated-history size |
| H_P | predictive entropy |

The letters R, C, A, V, G, E and H are not overloaded with multiple incompatible meanings. Types and operators use roman/calligraphic notation where needed.

## 1. State object

Persistent state is one typed object:

s_t = (X_t, Γ_t, M_t)

where:

- X_t = {x_1,...,x_n} is the persistent object set;
- Γ_t = (V_t,E_t) is the executable dependency graph;
- M_t contains provenance, validity, evidence, and lifecycle metadata.

Each object is:

x_i = (c_i, type_i, prov_i, conf_i, dep_i, val_i, ev_i, status_i)

with:

status_i ∈ {E,P,R,Q,T}

E = experimental, P = provisional, R = registered, Q = quarantined, T = retired.

The semantic state decomposition is:

X_t = {identity, goals, constraints, evidence, procedures, routing, history}

This is one decomposition of X_t, not a second incompatible definition of s_t.

## 2. Persistent lifecycle theorem

Let Gate(x) be the conjunction of the four registration gates:

Gate(x) = G_U(x) ∧ G_S(x) ∧ G_I(x) ∧ G_K(x)

where U = utility, S = stability, I = integrity, K = cost.

The registration transition is guarded:

P -> R  iff  Gate(x)=1.

No other transition enters R.

### Proposition — no unqualified registration

If an object has status R at time t, then all four registration gates have passed for that object.

### Proof

Base case: newly created objects have status E and therefore are not R.

Inductive step: the only transition into R is P -> R, whose transition predicate explicitly requires Gate(x)=1. All other transitions have target status E, P, Q, or T and therefore cannot create an R object.

Thus, by induction over all lifecycle transitions, status R implies Gate(x)=1.

This proposition is checkable without assumptions about model quality. A property test should enumerate state transitions and assert that an R transition with any failed gate is rejected.

The proposition does not prove that the gates themselves are good predictors of future utility; that remains empirical.

## 3. PNDS decision loop

The registered-state loop is:

p_t -> Router -> Graph -> Decision -> Action -> Outcome -> Verifier -> ExperimentalState -> Registration -> p_{t+1}

Formally:

g_t = Router_θ(p_t,q_t)

d_t = Compute_θ(g_t,p_t,q_t)

a_t ~ Policy_θ(d_t,p_t,q_t)

o_t ~ P_E(o_t | p_t,q_t,a_t)

v_t = Verify_φ(p_t,q_t,d_t,a_t,o_t)

e_{t+1} = Update_ψ(p_t,q_t,d_t,a_t,o_t,v_t)

p_{t+1} = Reg(e_{t+1}, p_t, history_of_evidence)

The important separation is:

e_{t+1} is proposed/experimental state; p_{t+1} is registered state.

## 4. Query-conditioned candidate generation

A router that scores every stored object cannot establish sublinear scaling.

Therefore PNDS uses two stages:

J_t = Index_kappa(h_t,q_t,s_t)

I_t = Route_θ(J_t,s_t,q_t)

where J_t is a candidate set of size m and I_t has size k ≤ m.

Total routing cost is:

κ_route(n,k,m) = κ_index(n,m) + κ_score(m) + κ_select(m,k)

not merely κ_score(k).

The architecture must therefore specify the index/candidate-generation mechanism used in a scaling experiment.

Candidate generation may be:

- exact index;
- approximate-nearest-neighbor index;
- learned coarse router;
- hierarchical structural index.

The scaling hypothesis is empirical:

κ_index(n,m) + κ_score(m) + κ_exec(k) + κ_verify(k)

should grow materially more slowly than full-context processing as irrelevant history grows, while m and k remain controlled by the relevant structure.

A router that evaluates all n objects is O(n) at the routing stage and fails the proposed sublinear-routing requirement, regardless of later TopK execution.

## 5. Softmax margin bound

Suppose r relevant candidates receive score at least z+m and n-r distractors receive score z.

For softmax temperature τ, the worst-case total distractor probability is:

ε_leak ≤ (n-r) / (r exp(m/τ) + n-r).

To require ε_leak ≤ ε:

m ≥ τ log(((n-r)(1-ε))/(r ε)).

For n >> r and small ε this behaves approximately as:

m ≳ τ log(n/(r ε)).

Therefore a fixed score margin cannot guarantee fixed leakage as n grows. Either the margin must grow, the temperature must decrease, or candidate generation must prevent all n distractors from entering the competition.

This bound is a routing constraint, not evidence that a particular router achieves it.

## 6. Differentiable Top-k training

Hard TopK is used only at inference or evaluation.

During training define a differentiable K-subset relaxation:

w^(τ) = SoftTopK_τ(r,g;k)

where g_i are i.i.d. Gumbel(0,1) variables and:

0 ≤ w_i^(τ) ≤ 1,

Σ_i w_i^(τ) = k,

lim_{τ→0} w^(τ) = one_hot(TopK(r+g,k))

in probability under the chosen relaxation.

The forward inference gate is:

g_i^hard = 1[i∈TopK(r,k)].

A straight-through estimator may use:

g_i^ST = stopgrad(g_i^hard - w_i^(τ)) + w_i^(τ).

Temperature is an explicit experimental variable:

τ_0 > τ_1 > ... > τ_min > 0.

The schedule, relaxation, and straight-through rule must be frozen before a confirmatory routing experiment.

A hard TopK result without a specified training relaxation is not considered a complete trainable formulation.

## 7. CASM executable graph

Let Γ=(V,E). For edge e=(j,i), define a gate b_e.

The executable weight is:

W_e = W_e^0 b_e.

During training:

b_e ∈ [0,1].

During discrete execution:

b_e ∈ {0,1}.

Node computation:

z_i = f_i({z_j : (j,i)∈E}, W).

The router selects b based on permitted state/query features, not runtime answer values or gold graph metadata.

## 8. Structural causal model

An intervention requires an explicit SCM:

Z = f_Z(Pa_Z,U_Z)

A = f_A(Z,Q,U_A)

O = f_O(A,Z,U_O)

Y = f_Y(O,Z,U_Y).

An intervention do(b_e=1) replaces the structural equation governing b_e while leaving exogenous variables and all non-intervened structural equations unchanged.

For an alternative on-manifold graph g', define:

Δ_exec = E[Y | do(G=g)] - E[Y | do(G=g')].

A causal execution claim requires a predefined non-zero practical effect threshold δ_exec:

|Δ_exec| > δ_exec.

The threshold is fixed before confirmatory evaluation.

## 9. On-manifold state controls

A state shuffle is valid only if the control preserves relevant nuisance distributions sufficiently for the intended comparison.

Let T be a state transformation satisfying, where applicable:

P(T(s)|q) ≈ P(s|q)

for measured nuisance features.

Use:

- reset control;
- matched cross-episode shuffle;
- wrong-context replacement sampled from the same state distribution;
- targeted corruption preserving surface statistics;
- stale-state replacement.

The state effect is then:

Δ_state(T)=E[Y|s,q]-E[Y|T(s),q].

A large effect after an obviously out-of-distribution corruption is not sufficient evidence for causal state use; the matched controls are required.

## 10. State-to-computation causal coupling

Distinguish:

Predictive:
I(s;Y|q)>0.

Functional:
E[Y|do(s=s_1),q] ≠ E[Y|do(s=s_2),q].

Path-specific:
E[Y|do(s=s_1), do(R=r),q] ≠ E[Y|do(s=s_2), do(R=r),q].

The path-specific test holds routing fixed while changing state.

The reciprocal test holds state fixed while intervening on routing.

This separates state causality from route causality.

## 11. Verification

Let a path contain transitions z_1,...,z_L.

The verifier outputs conditional validity estimates:

v_j = P(Z_j valid | z_{<j}, evidence_j).

The probability that every transition is correct is:

P_all = Π_{j=1}^L v_j.

Because P_all is length-sensitive, use a length-normalized path score when comparing trajectories of different L:

c_path = exp((1/L) Σ_j log(max(v_j, ε_num))).

Final-output verification is:

c_final = V_final(y,evidence).

Path verification and final verification are distinct quantities.

A commit threshold must therefore be defined on a declared scale. If all-step probability is used, the threshold must be length-aware. If normalized log-confidence is used, its threshold is defined directly on that normalized scale.

The expression V_P=Πv_j must not be interpreted as an unconditional product of marginal probabilities.

## 12. Verification and repair

Repair is a bounded search over candidate revisions:

d^(0) -> d^(1) -> ... -> d^(L).

To prevent Goodharting against the verifier, the search objective cannot be the learned verifier score alone.

Require an independent or environment-grounded terminal criterion:

J_repair = outcome_loss + λ_c cost + λ_s risk.

The verifier may prune candidates, but final acceptance is evaluated using an independent held-out/environment outcome when available.

The repair budget L_max and candidate-search budget are preregistered.

## 13. Experimental update

Use an edit script rather than a vector subtraction:

Δ_s = Edit(s_p, e).

Edit operations may include:

add object, remove object, modify object, add edge, remove edge, change type, change provenance, change confidence, change procedure.

The update complexity is:

K(Δ_s)=Σ_j cost(edit_j).

Therefore:

e = Apply(p, Δ_s)

is a well-defined state transformation.

## 14. Registration gates

Use separate discovery, validation, and locked-test datasets.

D = discovery cohort  
V_1,...,V_m = validation cohorts  
T = locked test evaluated only after the registration decision.

Define:

G_U = [LCB_α(ΔU_V) > δ_U]

G_S = [stability statistic across V_1,...,V_m passes threshold]

G_I = [false_commit rate ≤ ε_false ∧ contamination fraction ≤ ε_contam]

G_K = [Δκ ≤ κ_allowance OR risk-adjusted utility compensates for Δκ].

Then:

Gate(x)=G_U∧G_S∧G_I∧G_K.

The locked set T must not be used repeatedly to choose thresholds, candidates, or registration decisions.

## 15. Contamination metrics

Define false admission:

FAR = P(status=R | object invalid).

Define contamination fraction:

CF = P(object invalid | status=R).

These are different metrics and should not be conflated.

## 16. Adaptation regularization

For an adaptation proposal Δ_s:

ΔU = U(e)-U(p)

Δκ = κ(e)-κ(p)

ΔK = K(Δ_s)

ΔRisk = Risk(e)-Risk(p).

Use the empirical utility-adjusted score:

J_adapt = ΔU - λ_KΔK - λ_κΔκ - λ_RΔRisk.

Evolution trajectory regularization can add:

Ω_edit = Σ_t K(Δ_s,t)

Ω_repeat = Σ_t 1[proposal_t repeats a previously falsified mechanism]

Ω_cost = Σ_t max(0,Δκ_t).

The empirical research question is whether regularization improves held-out utility/integrity per unit persistent change and cost.

No specific λ values are theoretically required.

## 17. Context-scaling proof

Construct:

h_ℓ = r ∪ i_ℓ

where r is fixed relevant information and |i_ℓ| increases.

Compare:

FC: C_F(ℓ)

PS: C_P(s)

PR: C_PR(R(s,q))

PNDS: C_N(g(R(s,q))).

Total PNDS cost is:

C_N = κ_index(n,m) + κ_route(m,k) + κ_exec(k) + κ_verify(k) + κ_update(Δ_s).

The target is not merely:

C_N << C_F.

The stronger empirical condition is:

∂C_N/∂ℓ << ∂C_F/∂ℓ

while:

Acc_N(ℓ) ≥ Acc_F(ℓ) - δ_acc

over preregistered history levels.

A claimed constant-time or O(N/20) regime requires empirical scaling evidence; it is not implied by TopK execution alone.

## 18. Evidence status

The following are empirical hypotheses, not current theorems:

- sublinear context scaling;
- relevance routing that remains effective under increasing distractors;
- cheap router equivalence to CDL teacher;
- causal execution;
- verifier superiority;
- registration superiority;
- adaptation regularization superiority;
- cross-domain universality.

The lifecycle invariant in §2 is a formal property of the transition system once its gate predicates are defined.

## 19. Current evidence mapping

### Supported/controlled
- TAC content-addressed memory: bounded controlled support.
- TAC-SCM REAL004/REAL005/REAL006: bounded structure-to-behavior/transfer evidence with explicit controls; REAL005 reports 10-seed and full-sweep configurations, and REAL006 reports 10-seed and full-sweep configurations.
- CDL Stage A relevance: bounded controlled support.
- CDL candidate-count scaling: bounded controlled support.

### Inconclusive or blocked
- cheap CDL-distilled router: inconclusive.
- CASM Phase 1.5A L2 held-out routing: inconclusive; 0/18 held-out routing cases in the recorded reconstruction.
- full causal PNDS loop: not yet run.
- registration experiment: not yet run.
- adaptation regularization experiment: not yet run.
- context-scaling proof: not yet run.
- path-verification comparison: not yet run.
- R13/R14 repair evidence: provenance not located in the current TAC Transformer repository identifiers; do not promote until the source experiment is located and audited.

## 20. Replication statistics

Three seeds are a replication gate, not a conventional p<0.05 guarantee.

For seed-level replication, preregister:

- minimum practical effect δ;
- required sign consistency;
- maximum allowed seed variance;
- number of evaluation episodes;
- confidence interval method.

Statistical inference over episode-level observations must not be confused with independent replication across seeds.

## 21. Commercial adapter mapping

Code Repair:
s=repository state; Γ=dependency/test graph; a=patch; o=tests/build; v=regression/security verification.

Enterprise Memory:
s=facts/procedures/relations; Γ=knowledge/workflow graph; a=retrieval/reasoning/execution; o=task result; v=provenance/consistency.

World Model:
s=environment state; Γ=entity/spatial/causal graph; a=intervention; o=transition; v=prediction/intervention consistency.

Decision Graph:
s=decision/evidence state; Γ=rules/dependencies; a=decision; o=downstream observation; v=rule/evidence verification.

These are domain instantiations of the same operators, not evidence that one substrate is already universal.

## 22. Complete formal loop

The current mathematical representation is:

p_t
→ CandidateIndex(q_t,p_t)
→ RelevanceRoute(q_t,p_t)
→ ExecutableGraph(g_t)
→ TypedDecision(d_t)
→ Action(a_t)
→ Environment(o_t)
→ Path/OutcomeVerification(v_t)
→ ExperimentalUpdate(e_{t+1})
→ Quarantine/Registration
→ p_{t+1}.

The principal research claim is:

A persistent computational substrate can reuse verified structure across time while restricting each decision to a relevant executable subset, provided routing is computationally efficient, structural selection is causally effective, verification is meaningful, and persistence is admitted only through explicit evidence gates.
