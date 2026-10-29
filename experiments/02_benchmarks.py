"""Expanded experiments for survey v4.
A: classical results matrix (exact MILP, greedy, simulated annealing, tabu) on forests of 50-1000 stands.
B: complexity - multi-period harvest scheduling with even flow, exact MILP time vs size.
C: quantum statistics on 20 small forests - QAOA p=1..4 and simulated Rydberg (blockade) adiabatic sweep; TTS99.
D: complexity on quantum side - QAOA on small multi-period instances with even flow.
"""
import json, time, sys
import numpy as np, networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('DATA_DIR', os.path.join(_HERE, '..', 'results', 'data')) + '/'
FIGS = os.environ.get('OUT_DIR', os.path.join(_HERE, '..', 'results', 'figures')) + '/'
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
OUT = DATA
RES = {}

def forest(n, seed, deg=5.0):
    r = np.sqrt(deg / (np.pi * n))
    g = nx.random_geometric_graph(n, r, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    vol = np.round(rng.uniform(50, 300, n))
    return g, vol

# ---------------- exact single-period MWIS via MILP ----------------
def mwis_milp(g, w, tl=300):
    n = g.number_of_nodes(); E = list(g.edges())
    A = lil_matrix((max(len(E), 1), n))
    for k, (i, j) in enumerate(E): A[k, i] = 1; A[k, j] = 1
    t = time.perf_counter()
    r = milp(-w, constraints=[LinearConstraint(csr_matrix(A), -np.inf, 1)], integrality=np.ones(n), bounds=Bounds(0, 1),
             options={'time_limit': tl})
    return np.round(r.x).astype(int), time.perf_counter() - t

def greedy(g, w):
    t = time.perf_counter(); x = np.zeros(len(w), int); blocked = set()
    for i in sorted(range(len(w)), key=lambda i: -w[i] / (g.degree(i) + 1)):
        if i not in blocked: x[i] = 1; blocked |= set(g.neighbors(i)) | {i}
    return x, time.perf_counter() - t

def sa(g, w, P, sweeps, rng):
    n = len(w); nb = [np.array(list(g.neighbors(i)), int) for i in range(n)]
    t = time.perf_counter(); x = rng.integers(0, 2, n); T0, T1 = 2.0, 0.01
    for k in range(sweeps):
        T = T0 * (T1 / T0) ** (k / sweeps)
        for i in rng.permutation(n):
            d = (1 - 2 * x[i]) * (-w[i] + P * x[nb[i]].sum())
            if d < 0 or rng.random() < np.exp(-d / T): x[i] ^= 1
    return x, time.perf_counter() - t

def tabu(g, w, P, iters, rng, tenure=None):
    n = len(w); nb = [np.array(list(g.neighbors(i)), int) for i in range(n)]
    tenure = tenure or max(5, n // 10)
    t = time.perf_counter(); x = np.zeros(n, int)
    delta = np.array([(1 - 2 * x[i]) * (-w[i] + P * x[nb[i]].sum()) for i in range(n)], float)
    E = 0.0; best, bestE = x.copy(), 0.0; tl = np.zeros(n, int)
    for it in range(iters):
        cand = np.where((tl <= it) | (E + delta < bestE - 1e-12))[0]
        i = cand[np.argmin(delta[cand])]
        E += delta[i]; x[i] ^= 1; tl[i] = it + tenure
        delta[i] = -delta[i]
        s = 1 if x[i] else -1
        for j in nb[i]: delta[j] += (1 - 2 * x[j]) * P * s
        if E < bestE - 1e-12: bestE, best = E, x.copy()
    return best, time.perf_counter() - t

def feasible(g, x): return all(not (x[i] and x[j]) for i, j in g.edges())

# ---------------- A: classical matrix ----------------
def run_A():
    rows = []
    for n in (50, 100, 200, 500, 1000):
        stats = {k: dict(feas=[], succ=[], ratio=[], time=[]) for k in ('MILP (exact)', 'Greedy', 'Simulated annealing', 'Tabu search')}
        for s in range(10):
            g, vol = forest(n, seed=100 * n + s); w = vol / vol.max(); rng = np.random.default_rng(s)
            xo, to = mwis_milp(g, w); opt = w @ xo
            for name, (x, tt) in {'MILP (exact)': (xo, to), 'Greedy': greedy(g, w),
                                   'Simulated annealing': sa(g, w, 1.5, 200, rng),
                                   'Tabu search': tabu(g, w, 1.5, 20 * n, rng)}.items():
                f = feasible(g, x); v = w @ x if f else 0
                st = stats[name]; st['feas'].append(f); st['succ'].append(f and abs(v - opt) < 1e-9)
                st['ratio'].append(v / opt); st['time'].append(tt)
        for name, st in stats.items():
            rows.append(dict(n=n, method=name, feasible=float(np.mean(st['feas'])), success=float(np.mean(st['succ'])),
                             ratio=float(np.mean(st['ratio'])), time=float(np.mean(st['time']))))
            print('A', rows[-1], flush=True)
    RES['A'] = rows

# ---------------- B: multi-period with even flow, exact MILP ----------------
def run_B():
    rows = []; T = 5
    for n in (10, 15, 20, 25, 50, 100, 200):
        for band, label in ((None, 'adjacency only'), (0.10, 'adjacency + even flow ±10%'), (0.05, 'adjacency + even flow ±5%')):
            times, gaps, solved = [], [], []
            for s in range(3):
                g, vol = forest(n, seed=7000 + 10 * n + s)
                growth = np.array([(1.02) ** (5 * t) for t in range(T)]); disc = np.array([1.03 ** (-5 * t) for t in range(T)])
                v = np.outer(vol, growth)                     # volume if cut in period t
                c = -(v * disc).ravel()                       # maximise discounted volume
                idx = lambda i, t: i * T + t
                nv = n * T; rows_c = []; lb = []; ub = []
                A = lil_matrix((n + len(g.edges()) * T + (2 * T if band else 0), nv)); k = 0
                for i in range(n):
                    for t in range(T): A[k, idx(i, t)] = 1
                    lb.append(-np.inf); ub.append(1); k += 1
                for (i, j) in g.edges():
                    for t in range(T): A[k, idx(i, t)] = 1; A[k, idx(j, t)] = 1; lb.append(-np.inf); ub.append(1); k += 1
                if band:
                    H = 0.55 * v.sum() / T / growth.mean() * growth / growth.mean() * growth.mean() / T * T / T  # target ~ sustainable share
                    H = 0.6 * vol.sum() / T * growth / growth.mean()
                    for t in range(T):
                        for i in range(n): A[k, idx(i, t)] = v[i, t]
                        lb.append((1 - band) * H[t]); ub.append((1 + band) * H[t]); k += 1
                t0 = time.perf_counter()
                r = milp(c, constraints=[LinearConstraint(csr_matrix(A[:k]), lb, ub)], integrality=np.ones(nv), bounds=Bounds(0, 1),
                         options={'time_limit': 30, 'mip_rel_gap': 1e-6})
                tt = time.perf_counter() - t0
                gap = getattr(r, 'mip_gap', np.nan); ok = r.status == 0
                gap = np.nan if r.x is None else gap
                times.append(tt); gaps.append(gap if gap is not None else np.nan); solved.append(ok)
            rows.append(dict(n=n, model=label, time=float(np.mean(times)), max_time=float(np.max(times)),
                             solved=float(np.mean(solved)), gap=float(np.nanmax(gaps)) if not all(np.isnan(gaps)) else None))
            print('B', rows[-1], flush=True)
    RES['B'] = rows

# ---------------- quantum helpers (exact statevector) ----------------
def bits_of(N): return np.array([[(k >> q) & 1 for q in range(N)] for k in range(2 ** N)])

def qaoa_state(th, cd, p, N):
    psi = np.full(2 ** N, 2 ** (-N / 2), dtype=complex)
    for l in range(p):
        psi = psi * np.exp(-1j * th[l] * cd); c, s = np.cos(th[p + l]), -1j * np.sin(th[p + l])
        psi = psi.reshape([2] * N)
        for q in range(N):
            a0 = np.take(psi, 0, axis=q); a1 = np.take(psi, 1, axis=q)
            psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=q)
        psi = psi.reshape(-1)
    return psi

def run_qaoa(cd, N, p, rng, warm=None, restarts=8):
    cost = lambda th: float(np.abs(qaoa_state(th, cd, p, N)) ** 2 @ cd)
    starts = [np.concatenate([rng.uniform(0, 1.0, p), rng.uniform(0, np.pi / 2, p)]) for _ in range(restarts)]
    if warm is not None:
        starts.insert(0, np.concatenate([np.append(warm[:p - 1], warm[p - 2]), np.append(warm[p - 1:], warm[-1])]))
    best = min((minimize(cost, x0, method='COBYLA', options={'maxiter': 500}) for x0 in starts), key=lambda r: r.fun)
    return np.abs(qaoa_state(best.x, cd, p, N)) ** 2, best.x

def tts99(p, t_unit):
    if p <= 0: return np.inf
    if p >= 0.99: return t_unit
    return t_unit * np.log(0.01) / np.log(1 - p)

# Rydberg: evolve inside the independent-set subspace (perfect blockade)
def rydberg_sweep(g, w, Ttot, steps=200, Omega=1.0, Dmax=3.0):
    N = len(w); E = list(g.edges())
    states = [k for k in range(2 ** N) if all(not ((k >> i) & 1 and (k >> j) & 1) for i, j in E)]
    pos = {s: a for a, s in enumerate(states)}; M = len(states)
    X = lil_matrix((M, M))
    for a, s in enumerate(states):
        for i in range(N):
            t = s ^ (1 << i)
            if t in pos: X[a, pos[t]] = 0.5
    X = csr_matrix(X)
    nw = np.array([sum(w[i] for i in range(N) if (s >> i) & 1) for s in states])
    psi = np.zeros(M, complex); psi[pos[0]] = 1; dt = Ttot / steps
    for k in range(steps):
        t = (k + 0.5) / steps
        Om = Omega * np.sin(np.pi * t) ** 2          # smooth Rabi pulse
        De = Dmax * (2 * t - 1)                     # linear detuning sweep
        H = Om * X - De * csr_matrix((nw, (range(M), range(M))), shape=(M, M))
        psi = expm_multiply(-1j * dt * H, psi)
    pr = np.abs(psi) ** 2
    return states, nw, pr

def run_C():
    out = []; N = 12; B = bits_of(N)
    for s in range(20):
        rng = np.random.default_rng(500 + s)
        while True:
            pos = rng.random((N, 2)); g = nx.random_geometric_graph(N, 0.42, pos={i: pos[i] for i in range(N)})
            if nx.is_connected(g) and 3.0 <= 2 * g.number_of_edges() / N <= 5.0: break
        vol = np.round(rng.uniform(50, 300, N)); w = vol / vol.max()
        vals = B @ w; adj = sum(B[:, i] * B[:, j] for i, j in g.edges())
        feas = adj == 0; opt = vals[feas].max(); optm = feas & (np.abs(vals - opt) < 1e-9)
        rec = dict(inst=s, edges=g.number_of_edges(), rand_feas=float(feas.mean()), rand_opt=float(optm.mean()))
        cd = -vals + 1.0 * adj; warm = None
        for p in (1, 2, 3, 4):
            pr, warm = run_qaoa(cd, N, p, rng, warm)
            rec[f'qaoa{p}'] = dict(feas=float(pr[feas].sum()), opt=float(pr[optm].sum()),
                                   ratio=float((pr * np.where(feas, vals, 0)).sum() / opt))
        for Tt in (5, 20, 60):
            st, nw, pr = rydberg_sweep(g, w, Tt)
            rec[f'ryd{Tt}'] = dict(feas=1.0, opt=float(pr[np.abs(nw - opt) < 1e-9].sum()), ratio=float(pr @ nw / opt))
        sruns = [sa(g, w, 1.0, 100, rng)[0] for _ in range(50)]
        rec['sa'] = dict(feas=float(np.mean([feasible(g, x) for x in sruns])),
                         opt=float(np.mean([feasible(g, x) and abs(w @ x - opt) < 1e-9 for x in sruns])),
                         ratio=float(np.mean([(w @ x) / opt if feasible(g, x) else 0 for x in sruns])))
        gx, _ = greedy(g, w); rec['greedy'] = dict(feas=1.0, opt=float(abs(w @ gx - opt) < 1e-9), ratio=float(w @ gx / opt))
        out.append(rec); print('C', s, {k: (v if not isinstance(v, dict) else round(v['opt'], 3)) for k, v in rec.items()}, flush=True)
    RES['C'] = out

# ---------------- D: multi-period QAOA with even flow ----------------
def run_D():
    out = []; S, T = 6, 2; N = S * T; B = bits_of(N)
    for s in range(10):
        rng = np.random.default_rng(900 + s)
        while True:
            pos = rng.random((S, 2)); g = nx.random_geometric_graph(S, 0.55, pos={i: pos[i] for i in range(S)})
            if nx.is_connected(g): break
        vol = np.round(rng.uniform(50, 300, S)); v = vol / vol.max()
        X = B.reshape(-1, S, T)
        val = (X * v[None, :, None]).sum((1, 2))
        once = (X.sum(2) * (X.sum(2) - 1) / 2).sum(1)
        adj = sum(X[:, i, t] * X[:, j, t] for i, j in g.edges() for t in range(T))
        Vt = (X * v[None, :, None]).sum(1)
        H = 0.35 * v.sum()
        flowdev = ((Vt - H) ** 2).sum(1)
        okflow = (np.abs(Vt - H) <= 0.15 * H).all(1)
        feas = (once == 0) & (adj == 0) & okflow
        if feas.sum() == 0: continue
        opt = val[feas].max(); optm = feas & (np.abs(val - opt) < 1e-9)
        cd = -val + 1.0 * once + 1.0 * adj + 2.0 * flowdev / H
        rec = dict(inst=s, rand_feas=float(feas.mean()), rand_opt=float(optm.mean())); warm = None
        for p in (1, 2, 3):
            pr, warm = run_qaoa(cd, N, p, rng, warm, restarts=6)
            rec[f'qaoa{p}'] = dict(feas=float(pr[feas].sum()), opt=float(pr[optm].sum()))
        out.append(rec); print('D', rec, flush=True)
        if len(out) >= 8: break
    RES['D'] = out

if __name__ == '__main__':
    part = sys.argv[1]
    {'A': run_A, 'B': run_B, 'C': run_C, 'D': run_D}[part]()
    json.dump(RES, open(OUT + f'exp2_{part}.json', 'w'), indent=1, default=float)
    print('done', part)
