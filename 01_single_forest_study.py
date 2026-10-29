"""Stand-selection experiment: weighted MIS QUBO on a toy forest.
Exact brute force vs classical simulated annealing vs QAOA (Qiskit statevector)."""
import json, itertools, time
import numpy as np, networkx as nx
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('DATA_DIR', _HERE) + '/'
FIGS = os.environ.get('OUT_DIR', _HERE) + '/'
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
OUT = FIGS
rng = np.random.default_rng(7)

# ---------- 1. Toy forest: unit-disk graph of 14 stands ----------
N = 14
while True:
    pos = rng.random((N, 2))
    G = nx.random_geometric_graph(N, 0.38, pos={i: pos[i] for i in range(N)})
    if nx.is_connected(G) and 3.0 <= 2 * G.number_of_edges() / N <= 4.5:
        break
vol = np.round(rng.uniform(50, 300, N))          # harvestable volume, m3
w = vol / vol.max()                                # normalised value
E = list(G.edges())
meandeg = 2 * len(E) / N

def qubo_energy(x, P):
    return -w @ x + P * sum(x[i] * x[j] for i, j in E)
def feasible(x): return all(not (x[i] and x[j]) for i, j in E)

# ---------- 2. Exact optimum ----------
best_val, best_x = -1, None
for bits in itertools.product([0, 1], repeat=N):
    x = np.array(bits)
    if feasible(x):
        v = w @ x
        if v > best_val: best_val, best_x = v, x
opt = best_val
print('exact optimum', opt, best_x, 'edges', len(E), 'meandeg', meandeg)

# ---------- 3. Simulated annealing (classical baseline) ----------
def sa(P, sweeps=300, T0=2.0, T1=0.01):
    x = rng.integers(0, 2, N)
    nb = {i: list(G.neighbors(i)) for i in range(N)}
    for k in range(sweeps):
        T = T0 * (T1 / T0) ** (k / sweeps)
        for i in rng.permutation(N):
            d = (1 - 2 * x[i]) * (-w[i] + P * sum(x[j] for j in nb[i]))
            if d < 0 or rng.random() < np.exp(-d / T): x[i] ^= 1
    return x
P_main = 1.5
sa_runs = [sa(P_main) for _ in range(100)]
sa_feas = np.mean([feasible(x) for x in sa_runs])
sa_opt = np.mean([feasible(x) and abs(w @ x - opt) < 1e-9 for x in sa_runs])
sa_ratio = np.mean([(w @ x) / opt if feasible(x) else 0 for x in sa_runs])
print('SA feasible', sa_feas, 'optimal', sa_opt, 'ratio', sa_ratio)

# ---------- 4. QUBO -> Ising Hamiltonian ----------
def hamiltonian(P):
    # x_i = (1 - z_i)/2
    terms = {}
    def add(label, c): terms[label] = terms.get(label, 0) + c
    for i in range(N):
        add(('Z', i), w[i] / 2)               # -w_i x_i -> -w_i/2 + w_i/2 Z_i
    for i, j in E:                            # P x_i x_j = P/4 (1 - Z_i - Z_j + Z_i Z_j)
        add(('Z', i), -P / 4); add(('Z', j), -P / 4); add(('ZZ', i, j), P / 4)
    ops = []
    for k, c in terms.items():
        s = ['I'] * N
        for q in k[1:]: s[N - 1 - q] = 'Z'
        ops.append((''.join(s), c))
    return SparsePauliOp.from_list(ops).simplify()

def bit_probs(sv):
    p = np.abs(sv.data) ** 2
    return p
idx_bits = np.array([[(k >> q) & 1 for q in range(N)] for k in range(2 ** N)])
vals = idx_bits @ w
feas_mask = np.array([feasible(b) for b in idx_bits])
opt_mask = feas_mask & (np.abs(vals - opt) < 1e-9)

def cost_diag(P):
    adjpen = np.zeros(2 ** N)
    for i, j in E: adjpen += idx_bits[:, i] * idx_bits[:, j]
    return -vals + P * adjpen

def qaoa_state(th, cd, p):
    psi = np.full(2 ** N, 1 / np.sqrt(2 ** N), dtype=complex)
    for l in range(p):
        g, b = th[l], th[p + l]
        psi = psi * np.exp(-1j * g * cd)
        c, sn = np.cos(b), -1j * np.sin(b)
        psi = psi.reshape([2] * N)
        for q in range(N):
            a0 = np.take(psi, 0, axis=q); a1 = np.take(psi, 1, axis=q)
            psi = np.stack([c * a0 + sn * a1, sn * a0 + c * a1], axis=q)
        psi = psi.reshape(-1)
    return psi

def check_against_qiskit(P, n=8):
    """Fidelity between NumPy QAOA state and Qiskit QAOAAnsatz state on an n-stand sub-forest (p=1)."""
    sub = [(i, j) for i, j in E if i < n and j < n]
    ops = [(''.join('Z' if k == n - 1 - i else 'I' for k in range(n)), w[i] / 2 - P / 4 * sum(i in e for e in sub)) for i in range(n)]
    ops += [(''.join('Z' if k in (n - 1 - i, n - 1 - j) else 'I' for k in range(n)), P / 4) for i, j in sub]
    H = SparsePauliOp.from_list(ops).simplify(); ans = QAOAAnsatz(H, reps=1)
    g, bta = 0.37, 0.81
    sv = Statevector(ans.assign_parameters({q: (bta if 'β' in q.name else g) for q in ans.parameters})).data
    bits = np.array([[(k >> q) & 1 for q in range(n)] for k in range(2 ** n)])
    cd = -bits @ w[:n] + P * sum(bits[:, i] * bits[:, j] for i, j in sub)
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex) * np.exp(-1j * g * cd)
    psi = psi.reshape([2] * n)
    for q in range(n):
        a0 = np.take(psi, 0, axis=q); a1 = np.take(psi, 1, axis=q)
        psi = np.stack([np.cos(bta) * a0 - 1j * np.sin(bta) * a1, -1j * np.sin(bta) * a0 + np.cos(bta) * a1], axis=q)
    psi = psi.reshape(-1)
    dec = ans.decompose(reps=4)
    return float(abs(np.vdot(sv, psi)) ** 2), dec.depth(), dec.count_ops()

def run_qaoa(P, p, restarts=16, warm=None):
    cd = cost_diag(P)
    def cost(th):
        psi = qaoa_state(th, cd, p)
        return float((np.abs(psi) ** 2) @ cd)
    best = None
    starts = [np.concatenate([rng.uniform(0, 1.0, p), rng.uniform(0, np.pi / 2, p)]) for _ in range(restarts)]
    if warm is not None:   # INTERP-style warm start from depth p-1
        g0, b0 = warm[:p - 1], warm[p - 1:]
        starts.insert(0, np.concatenate([np.append(g0, g0[-1]), np.append(b0, b0[-1])]))
    for x0 in starts:
        res = minimize(cost, x0, method='COBYLA', options={'maxiter': 600})
        if best is None or res.fun < best.fun: best = res
    pr = np.abs(qaoa_state(best.x, cd, p)) ** 2
    samples = rng.choice(2 ** N, size=1024, p=pr / pr.sum())
    fs = [s for s in samples if feas_mask[s]]
    best_sample = max(vals[s] for s in fs) / opt if fs else 0
    exp_ratio = float((pr * np.where(feas_mask, vals, 0)).sum() / opt)
    run_qaoa.last = best.x
    return dict(p=p, P=P, energy=float(best.fun), p_feasible=float(pr[feas_mask].sum()), p_optimal=float(pr[opt_mask].sum()),
                exp_ratio=exp_ratio, best_of_1024=float(best_sample), n_params=2 * p)

# random-guess baseline
rand_feas = feas_mask.mean(); rand_opt = opt_mask.mean()

t0 = time.time()
fid, qdepth, qops = check_against_qiskit(P_main); print('qiskit fidelity', fid, 'depth p=1', qdepth, dict(qops))
qaoa_depth = []
for p in (1, 2, 3, 4):
    qaoa_depth.append(run_qaoa(P_main, p, warm=run_qaoa.last if p > 1 else None))
for r in qaoa_depth: print(r)
penalty = []
for lam in (0.5, 1.0, 1.5, 2.5, 4.0):
    r = run_qaoa(lam, 3, restarts=12); penalty.append(r); print('pen', r)
print('qaoa time', time.time() - t0)

# ---------- 5. Coupling counts: sparse adjacency vs dense even-flow ----------
T = 5
scaling = []
for n in (50, 100, 200, 500, 1000, 3000):
    r = np.sqrt(5 / (np.pi * n))     # radius for mean degree ~5
    g = nx.random_geometric_graph(n, r, seed=1)
    adj = g.number_of_edges() * T
    once = n * T * (T - 1) // 2
    flow = T * n * (n - 1) // 2
    scaling.append(dict(n=n, adjacency=adj, one_harvest=once, even_flow=flow, ratio=flow / (adj + once),
                        meandeg=2 * g.number_of_edges() / n))
for s in scaling: print(s)

# QUBO density for the toy forest over 3 periods, with and without even flow
T3 = 3; nv = N * T3; tot_pairs = nv * (nv - 1) // 2
sparse_nz = len(E) * T3 + N * T3 * (T3 - 1) // 2
with_flow_nz = len({tuple(sorted(((i, t), (j, t)))) for t in range(T3) for i in range(N) for j in range(N) if i < j}
                   | {tuple(sorted(((i, t), (j, t)))) for (i, j) in E for t in range(T3)}
                   | {((i, t), (i, u)) for i in range(N) for t in range(T3) for u in range(t + 1, T3)})
density = dict(variables=nv, possible_pairs=tot_pairs, sparse_nonzero=sparse_nz,
               sparse_density=sparse_nz / tot_pairs, flow_nonzero=with_flow_nz, flow_density=with_flow_nz / tot_pairs)
print(density)

# ---------- 6. Figures ----------
C1, C2, C3 = '#2F5530', '#C27C2C', '#6B8FB5'
fig, ax = plt.subplots(figsize=(5.4, 4.4))
P = {i: pos[i].copy() for i in range(N)}
P[7] = pos[7] + np.array([0.06, -0.07]); P[10] = pos[10] + np.array([-0.05, 0.04])   # separate two overlapping stands (drawing only)
nx.draw_networkx_edges(G, P, ax=ax, edge_color='#999')
nx.draw_networkx_nodes(G, P, ax=ax, node_color=[C1 if best_x[i] else '#D8E4D6' for i in range(N)],
                       node_size=[200 + vol[i] * 2 for i in range(N)], edgecolors='#333')
nx.draw_networkx_labels(G, P, ax=ax, labels={i: str(i) for i in range(N)}, font_color='black', font_size=8)
nx.draw_networkx_labels(G, P, ax=ax, labels={i: str(i) for i in range(N) if best_x[i]}, font_color='white', font_size=8)
ax.set_title('Toy forest: 14 stands (size = volume), optimal harvest set in dark green', fontsize=9); ax.axis('off')
plt.tight_layout(); plt.savefig(OUT + 'fig_forest.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.8))
x = np.arange(5); lbl = ['Random\nguess', 'QAOA p=1', 'QAOA p=2', 'QAOA p=3', 'QAOA p=4']
pf = [rand_feas] + [r['p_feasible'] for r in qaoa_depth]
po = [rand_opt] + [r['p_optimal'] for r in qaoa_depth]
ax.bar(x - 0.18, pf, 0.36, color=C3, label='P(feasible)')
ax.bar(x + 0.18, po, 0.36, color=C1, label='P(optimal)')
ax.set_ylim(0, 1.05)
for i in range(5):
    ax.text(x[i] - 0.18, pf[i] + 0.01, f'{pf[i]:.2f}', ha='center', fontsize=8)
    ax.text(x[i] + 0.18, po[i] + 0.01, f'{po[i]:.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_ylabel('Probability'); ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('QAOA output quality vs circuit depth (penalty P = 1.5)', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'fig_qaoa_depth.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.6))
lams = [r['P'] for r in penalty]
ax.plot(lams, [r['p_feasible'] for r in penalty], 'o-', color=C3, label='P(feasible)')
ax.plot(lams, [r['p_optimal'] for r in penalty], 's-', color=C1, label='P(optimal)')
ax.axvline(1.0, color='#999', ls=':', lw=1); ax.text(1.03, 0.9, 'P = max value', fontsize=8, color='#666',
        transform=ax.get_xaxis_transform())
ax.set_xlabel('Penalty weight P'); ax.set_ylabel('Probability'); ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Effect of penalty weight (QAOA p = 3)', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'fig_penalty.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.8))
ns = [s['n'] for s in scaling]
ax.loglog(ns, [s['adjacency'] + s['one_harvest'] for s in scaling], 'o-', color=C1, label='Adjacency + one-harvest (sparse)')
ax.loglog(ns, [s['even_flow'] for s in scaling], 's-', color=C2, label='Even flow (dense)')
ax.set_xlabel('Number of stands n'); ax.set_ylabel('QUBO couplings (T = 5)'); ax.legend(frameon=False)
ax.grid(True, which='major', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Coupling growth: sparse constraints vs even flow', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'fig_scaling.png', dpi=200); plt.close()

json.dump(dict(qiskit_fidelity=fid, circuit_depth_p1=qdepth, circuit_ops_p1={k:int(v) for k,v in dict(qops).items()}, N=N, edges=len(E), meandeg=meandeg, vol=vol.tolist(), opt=opt, opt_set=[int(i) for i in np.where(best_x)[0]],
               opt_volume=float(vol @ best_x), total_volume=float(vol.sum()),
               rand_feas=rand_feas, rand_opt=rand_opt, sa=dict(feasible=sa_feas, optimal=sa_opt, ratio=sa_ratio),
               qaoa_depth=qaoa_depth, penalty=penalty, scaling=scaling, density=density),
          open(DATA + 'results.json', 'w'), indent=1, default=float)
print('done')
