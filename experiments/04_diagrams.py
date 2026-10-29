import numpy as np, networkx as nx, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import os
OUT = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')) + '/'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
BLUE, ORANGE, AQUA, YELLOW, MAG, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d2'
FOREST = '#2F5530'

def box(ax, x, y, w, h, title, body, fc, ec, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.13, title, ha='center', va='top', fontsize=10, weight='bold', color=tc)
    ax.text(x + w / 2, y + h - 0.42, body, ha='center', va='top', fontsize=8, color=INK2, linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, c=INK2, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14, color=c, lw=1.5, linestyle=ls))

# ---------- D1 pipeline ----------
fig, ax = plt.subplots(figsize=(9, 3.6)); ax.set_xlim(0, 9); ax.set_ylim(0, 3.6); ax.axis('off')
stages = [('1. Inventory', 'Satellite, radar and laser data\n→ species, volume, condition\nper stand', 'Quantum ML\n(kernels, feature extraction)', 'Research stage'),
          ('2. Harvest scheduling', 'Which stands to cut\nin which period\n(adjacency + even flow)', 'QUBO: annealing, QAOA,\nRydberg MIS', 'Main focus'),
          ('3. Timber transport', 'Route log trucks\nfrom landings to mills\n(vehicle routing)', 'Hybrid quantum\nroute optimisation', 'Large literature')]
for k, (t, b, q, tag) in enumerate(stages):
    x = 0.25 + k * 3.0
    box(ax, x, 1.55, 2.5, 1.75, t, b, '#eef3ec' if k != 1 else '#dbe8d8', FOREST)
    ax.add_patch(FancyBboxPatch((x + 0.15, 0.25), 2.2, 0.95, boxstyle='round,pad=0.02,rounding_size=0.05', fc='#eaf1fb', ec=BLUE, lw=1.2))
    ax.text(x + 1.25, 0.95, q, ha='center', va='top', fontsize=8, color=INK)
    ax.text(x + 1.25, 0.37, tag, ha='center', fontsize=7.5, color=BLUE, style='italic')
    arrow(ax, x + 1.25, 1.53, x + 1.25, 1.22, BLUE)
    if k < 2: arrow(ax, x + 2.53, 2.42, x + 2.97, 2.42, FOREST)
ax.text(0.25, 3.5, 'Classical decision stage', fontsize=8, color=FOREST, weight='bold')
ax.text(0.25, 0.02, 'Candidate quantum method', fontsize=8, color=BLUE, weight='bold')
plt.tight_layout(); plt.savefig(OUT + 'd_pipeline.png', dpi=220); plt.close()

# ---------- D2 hybrid workflow ----------
fig, ax = plt.subplots(figsize=(9, 4.2)); ax.set_xlim(0, 9); ax.set_ylim(0, 4.2); ax.axis('off')
ax.add_patch(FancyBboxPatch((0.1, 2.3), 8.8, 1.8, boxstyle='round,pad=0.02', fc='#f6f6f2', ec=GRID))
ax.add_patch(FancyBboxPatch((0.1, 0.1), 8.8, 1.9, boxstyle='round,pad=0.02', fc='#f1f6fd', ec=GRID))
ax.text(0.25, 3.95, 'CLASSICAL LAYER (CPU / HPC)', fontsize=8, weight='bold', color=FOREST)
ax.text(0.25, 1.85, 'QUANTUM LAYER (QPU)', fontsize=8, weight='bold', color=BLUE)
box(ax, 0.3, 2.45, 1.9, 1.3, 'Inventory data', 'stands, volumes,\nadjacency map', 'white', FOREST)
box(ax, 2.5, 2.45, 2.0, 1.3, 'Set volume targets', 'classical relaxation\nenforces even flow', 'white', FOREST)
box(ax, 4.8, 2.45, 1.9, 1.3, 'Split into blocks', 'per period,\n~100–300 stands', 'white', FOREST)
box(ax, 7.0, 2.45, 1.8, 1.3, 'Combine & check', 'volumes vs targets;\nadjust and repeat', 'white', FOREST)
box(ax, 4.6, 0.3, 2.4, 1.35, 'Solve each block', 'weighted max independent set,\n~100–300 stands (subject to graph\ngeometry & embedding) → samples', 'white', BLUE)
arrow(ax, 2.2, 3.1, 2.48, 3.1, FOREST); arrow(ax, 4.5, 3.1, 4.78, 3.1, FOREST)
arrow(ax, 5.6, 2.43, 5.6, 1.68, BLUE); arrow(ax, 7.0, 0.95, 7.85, 2.43, BLUE)
ax.annotate('', xy=(3.5, 2.43), xytext=(7.9, 2.43), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.2, linestyle='--', connectionstyle='arc3,rad=-0.12'))
ax.text(7.3, 2.1, 'iterate until stable', fontsize=7.5, color=INK2, style='italic')
ax.text(0.3, 0.9, 'Even flow is handled classically\nas linear constraints, so no dense\npenalty term is built. Only sparse\nadjacency goes to the quantum device.', fontsize=8, color=INK2, va='center')
plt.tight_layout(); plt.savefig(OUT + 'd_hybrid.png', dpi=220); plt.close()

# ---------- D3 QAOA circuit ----------
fig, ax = plt.subplots(figsize=(9, 3.0)); ax.set_xlim(0, 10); ax.set_ylim(-0.6, 4.4); ax.axis('off')
nq = 4
for q in range(nq):
    y = 3.6 - q * 1.0; ax.plot([0.6, 9.6], [y, y], color=INK2, lw=1); ax.text(0.15, y, f'q{q}', va='center', fontsize=9)
def gate(x, y, lab, c, w=0.55):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.27), w, 0.54, boxstyle='round,pad=0.01,rounding_size=0.05', fc=c, ec='none'))
    ax.text(x, y, lab, ha='center', va='center', fontsize=7.5, color='white', weight='bold')
for q in range(nq): gate(1.1, 3.6 - q, 'H', INK2, 0.45)
ax.add_patch(FancyBboxPatch((1.6, -0.25), 5.3, 4.2, boxstyle='round,pad=0.02', fc='none', ec=GRID, ls='--'))
ax.text(4.25, 4.15, 'one QAOA layer (repeated p times)', ha='center', fontsize=8, color=INK2)
for q in range(nq): gate(2.1, 3.6 - q, 'RZ', ORANGE, 0.5)
pairs = [(0, 1, 3.0), (1, 2, 3.8), (0, 2, 4.6), (2, 3, 5.4)]
for a, b_, x in pairs:
    ya, yb = 3.6 - a, 3.6 - b_; ax.plot([x, x], [ya, yb], color=ORANGE, lw=2)
    for yy in (ya, yb): ax.add_patch(Circle((x, yy), 0.12, color=ORANGE))
    ax.text(x + 0.12, (ya + yb) / 2, 'ZZ', fontsize=7, color=ORANGE)
for q in range(nq): gate(6.4, 3.6 - q, 'RX', BLUE, 0.5)
for q in range(nq):
    ax.add_patch(FancyBboxPatch((8.7, 3.6 - q - 0.27), 0.6, 0.54, boxstyle='round,pad=0.01', fc='white', ec=INK))
    ax.text(9.0, 3.6 - q, 'M', ha='center', va='center', fontsize=8)
ax.text(1.1, -0.5, 'equal\nsuperposition', ha='center', fontsize=7.5, color=INK2)
ax.text(3.6, -0.5, 'cost step γ: stand values (RZ) and\none ZZ coupling per neighbour pair', ha='center', fontsize=7.5, color=ORANGE)
ax.text(6.4, -0.5, 'mixer step β', ha='center', fontsize=7.5, color=BLUE)
ax.text(9.0, -0.5, 'measure →\nharvest plan', ha='center', fontsize=7.5, color=INK2)
ax.text(7.6, 3.6, '· · ·', fontsize=14, color=INK2, va='center')
plt.tight_layout(); plt.savefig(OUT + 'd_qaoa.png', dpi=220); plt.close()

# ---------- D4 Rydberg blockade ----------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axs[0]; ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-0.4, 4.4); ax.set_ylim(-0.2, 3.4)
rng = np.random.default_rng(3); pts = np.array([[0.4, 0.6], [1.5, 0.4], [2.6, 0.8], [3.7, 0.5], [0.9, 1.8], [2.0, 1.6], [3.2, 1.9], [1.4, 2.9], [2.6, 2.8]])
R = 1.35
g = nx.Graph(); g.add_nodes_from(range(len(pts)))
for i in range(len(pts)):
    for j in range(i + 1, len(pts)):
        if np.linalg.norm(pts[i] - pts[j]) < R: g.add_edge(i, j)
mis = [0, 2, 7, 6]
mis = list(nx.maximal_independent_set(g, nodes=[0, 3, 7], seed=1))
for i, j in g.edges(): ax.plot(*zip(pts[i], pts[j]), color=GRID, lw=1.2, zorder=1)
for i in mis: ax.add_patch(Circle(pts[i], R / 2, fc=ORANGE, alpha=0.10, ec=ORANGE, lw=0.8, ls='--'))
for i, p in enumerate(pts):
    ax.add_patch(Circle(p, 0.17, fc=ORANGE if i in mis else 'white', ec=INK, lw=1.2, zorder=3))
ax.set_title('Atoms in optical tweezers', fontsize=9.5, color=INK)
ax.text(2.0, -0.15, 'Excited atom (orange) blocks every atom inside its radius:\nneighbours can never both be excited', ha='center', fontsize=7.5, color=INK2)
ax = axs[1]; ax.axis('off'); ax.set_xlim(0, 4.6); ax.set_ylim(0, 3.4)
rows = [('Forest stand', '→', 'one atom'), ('Adjacent stands', '→', 'atoms within blockade radius\n(may need a mapping step)'), ('Stand value', '→', 'local detuning Δᵢ (laser)'),
        ('“Not both neighbours”', '→', 'enforced by physics, no penalty'), ('Best harvest set', '→', 'ground state = max weight\nindependent set')]
for k, (a, m, b_) in enumerate(rows):
    y = 3.0 - k * 0.62
    ax.text(0.05, y, a, fontsize=8.5, weight='bold', color=FOREST, va='center')
    ax.text(1.75, y, m, fontsize=10, color=INK2, va='center')
    ax.text(2.05, y, b_, fontsize=8.5, color=INK, va='center')
ax.set_title('Mapping the forest problem', fontsize=9.5)
plt.tight_layout(); plt.savefig(OUT + 'd_rydberg.png', dpi=220); plt.close()

# ---------- D5 QUBO structure ----------
def forest_g(n, seed):
    r = np.sqrt(5 / (np.pi * n)); return nx.random_geometric_graph(n, r, seed=seed)
n, T = 40, 3; g = forest_g(n, 11)
def Q(even):
    M = np.zeros((n * T, n * T))
    for t in range(T):
        for i, j in g.edges(): M[i * T + t, j * T + t] = M[j * T + t, i * T + t] = 1
    for i in range(n):
        for t in range(T):
            for u in range(t + 1, T): M[i * T + t, i * T + u] = M[i * T + u, i * T + t] = 1
    if even:
        for t in range(T):
            for i in range(n):
                for j in range(i + 1, n): M[i * T + t, j * T + t] = M[j * T + t, i * T + t] = 1
    return M
fig, axs = plt.subplots(1, 2, figsize=(8, 3.9))
for ax, ev, ttl in ((axs[0], False, 'Adjacency + one-harvest only'), (axs[1], True, 'With even-flow constraint')):
    M = Q(ev); ax.spy(M, markersize=0.9, color=BLUE if not ev else ORANGE)
    ax.set_title(f'{ttl}\n{int(M.sum() / 2):,} couplings ({100 * M.sum() / (M.size - len(M)):.0f}% filled)', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'Penalty-QUBO coupling matrix (formulation of Section 5.2), {n} stands × {T} periods = {n * T} variables', fontsize=9.5, color=INK2)
plt.tight_layout(); plt.savefig(OUT + 'd_qubo.png', dpi=220); plt.close()

# ---------- D6 roadmap with stage-gates ----------
fig, ax = plt.subplots(figsize=(9, 4.0)); ax.set_xlim(2025.5, 2036); ax.set_ylim(-0.25, 3.9); ax.axis('off')
ax.plot([2026, 2035.6], [1.6, 1.6], color=INK2, lw=2)
for yr in range(2026, 2036, 1):
    ax.plot([yr, yr], [1.55, 1.65], color=INK2); ax.text(yr, 1.35, str(yr), ha='center', fontsize=7.5, color=INK2)
phases = [(2026, 2027.5, 'Phase 1 · Pilot', 'Benchmark one LVM\nsubdistrict; formulations,\ndata, skills', GREEN),
          (2027.5, 2030, 'Phase 2 · Hybrid prototype', 'Hybrid solver inside a\nplanning workflow; re-planning\nafter disturbances', BLUE),
          (2030, 2035.5, 'Phase 3 · Scale (only if gates passed)', 'Larger hybrid applications;\nintegrated harvest + transport\nunder uncertainty', VIOLET)]
for a_, b_, t, d, c in phases:
    ax.add_patch(FancyBboxPatch((a_ + 0.05, 1.85), b_ - a_ - 0.1, 0.32, boxstyle='round,pad=0.01', fc=c, ec='none', alpha=0.9))
    ax.text((a_ + b_) / 2, 2.01, t, ha='center', va='center', fontsize=7.8, color='white', weight='bold')
    ax.text((a_ + b_) / 2, 2.85, d, ha='center', va='center', fontsize=7.5, color=INK2)
for x, lab in ((2027.5, 'Gate 1'), (2030, 'Gate 2')):
    ax.add_patch(FancyBboxPatch((x - 0.32, 3.45), 0.64, 0.3, boxstyle='round,pad=0.01', fc='white', ec=RED, lw=1.3))
    ax.text(x, 3.6, lab, ha='center', va='center', fontsize=7.5, color=RED, weight='bold')
    ax.plot([x, x], [3.45, 2.2], color=RED, lw=1, ls=':')
ev = [(2026.4, 'EuroHPC pilot access\n(opened 2026)'), (2027.2, 'EU Quantum Act\n(proposal expected)'), (2029.5, 'Vendor targets for fault-\ntolerant systems (target)'), (2035, 'Market USD 43–71 bn\n(McKinsey forecast)')]
for k, (x, t) in enumerate(ev):
    yb = 0.95 if k % 2 == 0 else 0.45
    ax.plot([x, x], [1.6, yb], color=GRID, lw=1); ax.add_patch(Circle((x, 1.6), 0.06, color=ORANGE, zorder=3))
    ax.text(x, yb - 0.08, t, ha='center', va='top', fontsize=7, color=INK)
ax.text(2025.6, -0.2, 'Orange: external milestones — targets and forecasts, not guarantees.  Red: stage-gates with predefined KPIs (Section 12.8).', fontsize=7, color=INK2)
plt.tight_layout(); plt.savefig(OUT + 'd_roadmap.png', dpi=220); plt.close()

# ---------- D7 industry evidence ----------
fig, ax = plt.subplots(figsize=(8, 3.6))
cases = [('Ford Otosan (2025)\nvendor-reported', 30 * 60, 5 * 60, 'production; vs previous system'), ('BASF (2025)\nvendor-reported', 10 * 3600, 5, 'pilot; vs projected time'),
         ('Kim et al. (2025)\nacademic benchmark', 561, 0.0855, 'solver time only; excl. queue/network')]
y = np.arange(len(cases))[::-1]
for yi, (lab, before, after, st) in zip(y, cases):
    ax.barh(yi + 0.18, before, height=0.32, color='#b9b8b0', label='Classical / previous' if yi == y[0] else None)
    ax.barh(yi - 0.18, after, height=0.32, color=BLUE, label='Hybrid quantum' if yi == y[0] else None)
    fmt = lambda s: f'{s / 3600:.0f} h' if s >= 3600 else (f'{s / 60:.0f} min' if s >= 60 else f'{s:g} s')
    ax.text(before * 1.25, yi + 0.18, fmt(before), va='center', fontsize=7.5, color=INK2)
    ax.text(after * 1.25, yi - 0.18, f'{fmt(after)}  ({st})', va='center', fontsize=7.5, color=INK)
ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels([c[0] for c in cases], fontsize=8)
ax.set_xlabel('Reported time to schedule / solve (log scale)'); ax.set_xlim(0.03, 4e5)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc='lower right'); ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('Reported time reductions — different problems, baselines and timing definitions; not directly comparable', fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT + 'd_industry.png', dpi=220); plt.close()

# ---------- D8 arithmetic illustration ----------
fig, ax = plt.subplots(figsize=(7.5, 3.3))
rev = 586.1; imp = [0.5, 1, 2, 3]
vals = [rev * i / 100 for i in imp]
bars = ax.bar([f'{i}%' for i in imp], vals, color='#9db89a', width=0.55, hatch='//', edgecolor='white')
for b_, v in zip(bars, vals): ax.text(b_.get_x() + b_.get_width() / 2, v + 0.3, f'€{v:.1f} m', ha='center', fontsize=8.5)
ax.set_ylabel('EUR million'); ax.set_xlabel('Share of LVM net revenue (€586.1 m, 2024)')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('Scale illustration only: x% of LVM revenue in euros.\nNot an estimate of savings from a 1% better optimization objective.', fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT + 'd_value.png', dpi=220); plt.close()
print('ok')
