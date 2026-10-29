import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
DATA = os.environ.get('DATA_DIR', os.path.dirname(os.path.abspath(__file__))) + '/'
OUT = os.environ.get('OUT_DIR', os.path.dirname(os.path.abspath(__file__))) + '/'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
BLUE, ORANGE, AQUA, YELLOW, MAG, GREEN, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d2'
def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)

A = json.load(open(DATA + 'exp2_A.json'))['A']; B = json.load(open(DATA + 'exp2_B.json'))['B']
C = json.load(open(DATA + 'exp2_C.json'))['C']; D = json.load(open(DATA + 'exp2_D.json'))['D']
S = {'A': A, 'B': B}

# ---- C summary ----
def tts(p):  # samples/runs needed for 99% chance of seeing the optimum
    if p <= 0: return None
    return 1.0 if p >= 0.99 else float(np.log(0.01) / np.log(1 - p))
keys = [('Random guess', None), ('QAOA p=1', 'qaoa1'), ('QAOA p=2', 'qaoa2'), ('QAOA p=3', 'qaoa3'), ('QAOA p=4', 'qaoa4'),
        ('Rydberg sweep T=5', 'ryd5'), ('Rydberg sweep T=20', 'ryd20'), ('Rydberg sweep T=60', 'ryd60'), ('Greedy', 'greedy'), ('Simulated annealing', 'sa')]
Cs = {}
for name, k in keys:
    if k is None:
        f = np.array([r['rand_feas'] for r in C]); o = np.array([r['rand_opt'] for r in C]); ra = None
    else:
        f = np.array([r[k]['feas'] for r in C]); o = np.array([r[k]['opt'] for r in C]); ra = np.array([r[k]['ratio'] for r in C])
    Cs[name] = dict(feas=float(f.mean()), opt=float(o.mean()), opt_sd=float(o.std()), opt_min=float(o.min()), opt_max=float(o.max()),
                    ratio=None if ra is None else float(ra.mean()), tts_median=float(np.median([tts(x) or np.inf for x in o])))
S['C'] = dict(n_inst=len(C), N=12, mean_edges=float(np.mean([r['edges'] for r in C])), methods=Cs)
for k, v in Cs.items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})

# ---- D summary ----
Ds = {'Random guess': dict(feas=float(np.mean([r['rand_feas'] for r in D])), opt=float(np.mean([r['rand_opt'] for r in D])))}
for p in (1, 2, 3):
    Ds[f'QAOA p={p}'] = dict(feas=float(np.mean([r[f'qaoa{p}']['feas'] for r in D])), opt=float(np.mean([r[f'qaoa{p}']['opt'] for r in D])))
S['D'] = dict(n_inst=len(D), stands=6, periods=2, qubits=12, methods=Ds)
print('D', Ds)

# ---- Figure: classical matrix ----
ns = sorted({r['n'] for r in A}); meths = ['MILP (exact)', 'Greedy', 'Tabu search', 'Simulated annealing']; cols = [INK, BLUE, AQUA, ORANGE]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for m, c in zip(meths, cols):
    rr = [next(r for r in A if r['n'] == n and r['method'] == m) for n in ns]
    axs[0].plot(ns, [100 * r['ratio'] for r in rr], 'o-', color=c, lw=2, ms=5, label=m)
    axs[1].plot(ns, [r['time'] for r in rr], 'o-', color=c, lw=2, ms=5, label=m)
axs[0].set_xscale('log'); axs[0].set_ylabel('Solution quality (% of optimum)'); axs[0].set_xlabel('Stands'); axs[0].set_ylim(90, 100.8)
axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_ylabel('Time per solve (s)'); axs[1].set_xlabel('Stands')
axs[0].set_title('Quality — single-period stand selection', fontsize=9); axs[1].set_title('Runtime', fontsize=9)
for ax in axs: clean(ax)
axs[0].legend(frameon=False, fontsize=7.5, loc='lower left')
plt.tight_layout(); plt.savefig(OUT + 'f_classical.png', dpi=220); plt.close()

# ---- Figure: complexity (B) ----
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for model, c, mk in (('adjacency only', BLUE, 'o'), ('adjacency + even flow ±10%', ORANGE, 's'), ('adjacency + even flow ±5%', VIOLET, '^')):
    rr = [r for r in B if r['model'] == model and r['n'] >= 20]
    ax.plot([r['n'] for r in rr], [r['time'] for r in rr], marker=mk, color=c, lw=2, ms=6, label=model)
ax.axhline(30, color=INK2, ls=':', lw=1); ax.text(52, 12, '30 s time limit reached:\noptimum not proven', fontsize=7.5, color=INK2)
ax.set_xscale('log'); ax.set_yscale('log')
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
ax.xaxis.set_major_locator(FixedLocator([20, 25, 50, 100, 200])); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.set_xlabel('Stands (5 periods)'); ax.set_ylabel('Exact solver time (s)')
ax.set_title('HiGHS exact solver, 30 s limit: with even flow, optimality could not be proven from 25 stands', fontsize=9); clean(ax)
ax.legend(frameon=False, fontsize=8, loc='center right')
plt.tight_layout(); plt.savefig(OUT + 'f_complexity.png', dpi=220); plt.close()

# ---- Figure: quantum statistics (C) ----
names = ['Random guess', 'QAOA p=1', 'QAOA p=2', 'QAOA p=3', 'QAOA p=4', 'Rydberg sweep T=5', 'Rydberg sweep T=20', 'Rydberg sweep T=60', 'Greedy', 'Simulated annealing']
short = ['Random', 'QAOA\np=1', 'QAOA\np=2', 'QAOA\np=3', 'QAOA\np=4', 'Rydberg\nT=5', 'Rydberg\nT=20', 'Rydberg\nT=60', 'Greedy*', 'Sim.\nanneal.']
colc = [INK2] + [BLUE] * 4 + [VIOLET] * 3 + [AQUA, ORANGE]
fig, axs = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
x = np.arange(len(names))
f = [Cs[n]['feas'] for n in names]; o = [Cs[n]['opt'] for n in names]; sd = [Cs[n]['opt_sd'] for n in names]
axs[0].bar(x, [100 * v for v in f], color=colc, width=0.6)
for i, v in enumerate(f): axs[0].text(i, 100 * v + 2, f'{100 * v:.0f}%', ha='center', fontsize=7.5)
axs[0].set_ylabel('Valid plans (%)'); axs[0].set_ylim(0, 115)
axs[1].bar(x, [100 * v for v in o], color=colc, width=0.6, yerr=[100 * v for v in sd], error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
for i, v in enumerate(o): axs[1].text(i + 0.33, 100 * v + 1.5, f'{100 * v:.1f}%' if v < 0.1 else f'{100 * v:.0f}%', ha='left', fontsize=7.2)
axs[1].set_ylabel('Exact optimum found (%)'); axs[1].set_ylim(0, 112)
axs[1].set_xticks(x); axs[1].set_xticklabels(short, fontsize=7.5)
for ax in axs: clean(ax); ax.grid(axis='x', visible=False)
axs[0].set_title(f'Proof-of-concept: success rates over {len(C)} random 12-stand forests, noiseless simulation (mean; error bars = 1 s.d.)', fontsize=8.5)
fig.text(0.01, 0.005, '*Greedy is deterministic: its bar is the share of forests on which its single plan was optimal, not a sampling probability.', fontsize=7, color=INK2)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUT + 'f_quantum.png', dpi=220); plt.close()

# ---- Figure: single vs multi-period QAOA (D) ----
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
ps = [1, 2, 3]; w = 0.36
for ax, key, ttl in ((axs[0], 'feas', 'Valid plans (%)'), (axs[1], 'opt', 'Exact optimum found (%)')):
    single = [Cs[f'QAOA p={p}'][key] for p in ps]; multi = [Ds[f'QAOA p={p}'][key] for p in ps]
    ax.bar(np.array(ps) - w / 2, [100 * v for v in single], w, color=BLUE, label='Single period, adjacency only (12 stands)')
    ax.bar(np.array(ps) + w / 2, [100 * v for v in multi], w, color=ORANGE, label='Two periods + even flow (6 stands × 2)')
    top = 100 * max(single + multi)
    for p, a_, b_ in zip(ps, single, multi):
        ax.text(p - w / 2, 100 * a_ + top * 0.02, f'{100 * a_:.1f}', ha='center', fontsize=7.5); ax.text(p + w / 2, 100 * b_ + top * 0.02, f'{100 * b_:.1f}', ha='center', fontsize=7.5)
    ax.set_xticks(ps); ax.set_xticklabels([f'p={p}' for p in ps]); ax.set_title(ttl, fontsize=9); ax.set_ylim(0, top * 1.15)
    clean(ax); ax.grid(axis='x', visible=False)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc='lower center', ncol=2)
fig.suptitle(f'QAOA on 12 qubits (different instance sets: {len(C)} single-period vs {len(D)} two-period forests)', fontsize=9.5)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(OUT + 'f_multiperiod.png', dpi=220); plt.close()

# ---- Figure: coupling growth (penalty-QUBO) ----
Rr = json.load(open(DATA + 'results.json'))
fig, ax = plt.subplots(figsize=(6, 3.8)); sc = Rr['scaling']; ns2 = [x['n'] for x in sc]
ax.loglog(ns2, [x['adjacency'] + x['one_harvest'] for x in sc], 'o-', color=GREEN, label='Adjacency + one-harvest (sparse)')
ax.loglog(ns2, [x['even_flow'] for x in sc], 's-', color=ORANGE, label='Even-flow penalty term (dense)')
ax.set_xlabel('Number of stands n'); ax.set_ylabel('QUBO couplings (T = 5)'); ax.legend(frameon=False, fontsize=8); clean(ax)
ax.set_title('Coupling growth in the penalty-QUBO formulation used here', fontsize=9)
plt.tight_layout(); plt.savefig(OUT + 'fig_scaling.png', dpi=200); plt.close()

json.dump(S, open(DATA + 'summary.json', 'w'), indent=1)
print('saved')
