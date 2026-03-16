---
theme: default
title: "Greedy Multi-Solver Routing for PDE Linear Systems"
info: |
  Adaptive solver selection for the linear subproblems arising in incompressible CFD.
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
---

<div class="h-full flex flex-col items-center justify-center">
<div class="text-5xl font-bold tracking-tight leading-tight">
Greedy Multi-Solver Routing<br>for PDE Linear Systems
</div>

<div class="mt-4 text-lg opacity-50 tracking-widest uppercase">
Adaptive solver selection for incompressible CFD
</div>

<img src="./images/cfd_airflow.png" class="mt-6 w-160 h-32 object-cover rounded-lg opacity-60" />

<div class="abs-b mb-10 text-sm tracking-widest uppercase opacity-30">
SemiAnalysis x Fluidstack Hackathon
</div>
</div>

---

# Motivation: The Poisson Equation Is Everywhere

<div class="grid grid-cols-2 gap-10 mt-3 text-sm">
<div>

<div class="text-xs tracking-widest uppercase opacity-40 mb-2">Ubiquity of Poisson Solves</div>

The Poisson equation $\nabla^2 u = f$ appears as a **core computational kernel** across science and engineering:

<div class="mt-2 space-y-1 text-[0.82rem] opacity-80">

- **CFD** — pressure projection in incompressible Navier–Stokes
- **Electrostatics** — electric potential from charge distributions
- **Gravitational physics** — potential fields in astrophysics, geodesy
- **Structural mechanics** — stress analysis, plate bending
- **Heat transfer** — steady-state temperature distributions
- **Image processing** — Poisson blending, inpainting, surface reconstruction

</div>

</div>
<div>

<div class="text-xs tracking-widest uppercase opacity-40 mb-2">The Neural Operator Dilemma</div>

Neural operators can approximate PDE solutions **orders of magnitude faster** than classical solvers.

But they offer **no convergence guarantees** — predictions may look plausible while being quantitatively wrong.

<div class="mt-3 pl-4 border-l-2 border-cyan-400 opacity-80">

**Our approach:** Route between classical iterative methods (with guarantees) and a neural operator (with speed). Use ML when it helps, fall back to classical when it doesn't, and **always converge**.

</div>

<div class="mt-3 pl-4 border-l-2 border-amber-400 opacity-80">

Accelerating the Poisson solve has **outsized practical impact** — it is the dominant bottleneck in projection-based CFD solvers and appears in virtually every branch of computational physics.

</div>

</div>
</div>

---

# Focus: The Poisson Equation

<div class="mt-4">

<div class="text-xs tracking-widest uppercase opacity-40 mb-3">Our Test Problem</div>

<div class="rounded-lg p-5 bg-white/5 border border-cyan-400/30">

**2D Poisson with periodic boundary conditions**

$$-\nabla^2 u = f \quad \text{on } [0,1]^2$$

<div class="mt-3 space-y-1 text-sm opacity-80">

- Grid size $N = 31$ ($961$ unknowns)
- Forcing $f$ drawn from hierarchical Gaussian random fields
- Solutions unique up to a constant (mean-zero constraint)

</div>

</div>

<div class="mt-5 text-sm opacity-70">

This isolates the core challenge: **can adaptive solver routing accelerate convergence of the Poisson linear system?**

</div>

</div>

---

# The Routing Loop

<div class="flex justify-center">
<div class="relative" style="width: 780px; height: 310px;">

<svg viewBox="0 0 780 310" style="position:absolute;inset:0;width:100%;height:100%">
  <defs>
    <marker id="arrowC" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#22d3ee" opacity="0.4"/>
    </marker>
    <marker id="arrowW" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#fff" opacity="0.5"/>
    </marker>
    <marker id="arrowG" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#4ade80" opacity="0.7"/>
    </marker>
  </defs>
  <line x1="118" y1="105" x2="222" y2="105" stroke="#fff" stroke-width="1.5" opacity="0.3" marker-end="url(#arrowW)"/>
  <line x1="348" y1="105" x2="452" y2="105" stroke="#fff" stroke-width="1.5" opacity="0.3" marker-end="url(#arrowW)"/>
  <line x1="578" y1="105" x2="662" y2="105" stroke="#fff" stroke-width="1.5" opacity="0.15" stroke-dasharray="6,4" marker-end="url(#arrowW)"/>
  <path d="M 580,125 Q 620,250 390,265 Q 160,280 100,135" fill="none" stroke="#22d3ee" stroke-width="1.5" opacity="0.25" stroke-dasharray="6,4" marker-end="url(#arrowC)"/>
  <path d="M 390,220 L 390,170" stroke="#4ade80" stroke-width="1.5" opacity="0.5" marker-end="url(#arrowG)"/>
</svg>

<div class="absolute flex flex-col items-center" style="left: 40px; top: 76px;">
  <div class="px-4 py-2 rounded-lg bg-white/8 border border-white/15 text-center">
    <div class="font-mono text-base text-white">u<sup>(t)</sup></div>
    <div class="text-[10px] opacity-40 mt-1">current solution</div>
  </div>
</div>

<div class="absolute flex flex-col items-center" style="left: 215px; top: 50px;">
  <div class="text-[10px] tracking-widest uppercase opacity-30 mb-1">router picks k*</div>
  <div class="relative">
    <div class="w-28 h-24 rounded-xl bg-white/5 border border-cyan-400/30 flex items-center justify-center">
      <div class="text-center">
        <div class="text-[10px] tracking-widest uppercase text-cyan-400 opacity-70">action</div>
        <div class="text-xs mt-1 font-mono opacity-80">argmin<sub>k</sub> err</div>
      </div>
    </div>
    <div class="absolute text-[9px] font-mono" style="right: -80px; top: -4px;">
      <div class="px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-400/20">SOR(1.0)</div>
    </div>
    <div class="absolute text-[9px] font-mono" style="right: -80px; top: 20px;">
      <div class="px-2 py-0.5 rounded bg-orange-500/15 text-orange-300 border border-orange-400/20">SOR(1.3)</div>
    </div>
    <div class="absolute text-[9px] font-mono" style="right: -80px; top: 44px;">
      <div class="px-2 py-0.5 rounded bg-red-500/15 text-red-300 border border-red-400/20">SOR(1.6)</div>
    </div>
    <div class="absolute text-[9px] font-mono" style="right: -80px; top: 68px;">
      <div class="px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-400/30">FNO ✦</div>
    </div>
  </div>
</div>

<div class="absolute flex flex-col items-center" style="left: 460px; top: 76px;">
  <div class="px-4 py-2 rounded-lg bg-white/8 border border-white/15 text-center">
    <div class="font-mono text-base text-white">u<sup>(t+1)</sup></div>
    <div class="text-[10px] opacity-40 mt-1">updated solution</div>
  </div>
</div>

<div class="absolute font-mono text-xl tracking-[0.4em] opacity-20" style="left: 620px; top: 93px;">···</div>

<div class="absolute" style="left: 670px; top: 78px;">
  <div class="px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-400/20 text-center">
    <div class="font-mono text-sm text-emerald-400">u*</div>
    <div class="text-[10px] opacity-40 mt-0.5">converged</div>
  </div>
</div>

<div class="absolute" style="left: 250px; top: 225px;">
  <div class="px-5 py-2 rounded-lg bg-emerald-500/8 border border-emerald-400/20">
    <div class="text-[10px] tracking-widest uppercase text-emerald-400 opacity-70 mb-1">Unrolled training</div>
    <div class="text-xs opacity-70">FNO sees <span class="text-cyan-300 font-mono">real in-loop residuals</span></div>
    <div class="text-xs opacity-50 mt-0.5">∇<sub>θ</sub> Σ<sub>t</sub> ‖FNO(r<sup>(t)</sup>) − correction*‖²</div>
  </div>
</div>

<div class="absolute text-[10px] text-cyan-400 opacity-40 italic" style="left: 120px; top: 270px;">repeat until ‖r‖ &lt; ε</div>

</div>
</div>

<div class="grid grid-cols-2 gap-8 text-xs opacity-70">
<div>

**Inference:** At each iteration the router evaluates all K+1 candidates and picks the one minimising immediate error.

</div>
<div>

**Training:** The FNO is fine-tuned through unrolled trajectories so it learns to correct the residuals that *actually arise* mid-solve — not random i.i.d. residuals.

</div>
</div>

---

# Oracle Greedy vs. Learned Router

<div class="grid grid-cols-2 gap-10 mt-6">
<div>

<div class="text-xs tracking-widest uppercase opacity-40 mb-2">Oracle Greedy (upper bound)</div>

<div class="rounded-lg p-5 bg-white/5 border border-emerald-400/30 space-y-3 text-sm opacity-85">

At every iteration, the oracle **tries every solver** on the current state, measures which one actually reduces the error the most, and picks that one.

This is **cheating** — it requires knowing the true solution to measure error. We can't do this in practice, but it tells us the **best possible** routing strategy.

<div class="pl-3 border-l-2 border-emerald-400/50 text-xs opacity-70 mt-3">
Think of it as a chess player who can see every possible move's outcome one step ahead and always picks the best.
</div>

</div>

</div>
<div>

<div class="text-xs tracking-widest uppercase opacity-40 mb-2">Learned LSTM Router (practical)</div>

<div class="rounded-lg p-5 bg-white/5 border border-cyan-400/30 space-y-3 text-sm opacity-85">

A small recurrent neural network that **observes the solver state** (current residual, iteration count) and **predicts** which solver will be most effective — without access to the true solution.

Trained to **imitate the oracle's decisions** on a training set, then deployed on new problems it has never seen.

<div class="pl-3 border-l-2 border-cyan-400/50 text-xs opacity-70 mt-3">
The gap between the oracle and the router measures how much room remains for improving the learned policy.
</div>

</div>

</div>
</div>

<div class="mt-5 text-center text-xs opacity-40">
The oracle establishes what is achievable · the router shows what we can do without privileged information
</div>

---
class: text-center
---

<div class="h-full flex flex-col items-center justify-center">
<div class="text-xs tracking-widest uppercase opacity-30 mb-4">Results</div>
<div class="text-4xl font-bold tracking-tight">2D Poisson Equation</div>
<div class="mt-4 text-lg opacity-50">Oracle greedy routing — SOR(1.0) + SOR(1.3) + SOR(1.6) + FNO</div>
</div>

---

# 2D Poisson: Convergence

<img src="./images/poisson_fno_convergence.png" class="w-full max-h-72 object-contain rounded-lg border-0" />

<div class="grid grid-cols-3 gap-4 mt-3 text-sm">
<div class="text-center p-3 rounded-lg bg-white/5 border border-white/10">

**Best Classical (SOR 1.3)**
<br><span class="font-mono text-xs">Final L2: 3.41 × 10⁻⁶</span>
<br><span class="font-mono text-xs">AUC: 0.371</span>

</div>
<div class="text-center p-3 rounded-lg bg-white/5 border border-red-400/40">

<span class="text-red-400">**LSTM Router (learned)**</span>
<br><span class="font-mono text-xs">Final L2: 1.13 × 10⁻⁶</span>
<br><span class="font-mono text-xs text-red-400">AUC: 0.024 — 15.5× lower</span>

</div>
<div class="text-center p-3 rounded-lg bg-white/5 border border-emerald-400/40">

<span class="text-emerald-400">**Oracle Greedy**</span>
<br><span class="font-mono text-xs">Final L2: 5.22 × 10⁻⁸</span>
<br><span class="font-mono text-xs text-emerald-400">AUC: 8.77 × 10⁻⁴ — 423× lower</span>

</div>
</div>

---

# 2D Poisson: Routing Patterns

<img src="./images/poisson_routing_comparison.png" class="w-full max-h-72 object-contain rounded-lg border-0" />

<div class="grid grid-cols-2 gap-6 mt-3 text-sm">
<div class="opacity-80">

**Oracle Greedy** — the upper bound:
- Highly adaptive per-sample routing
- **SOR(1.6)** dominates (~64%), **SOR(1.3)** ~27%, **SOR(1.0)** ~8%
- FNO used sparingly (~0.5%) for targeted corrections
- AUC: **8.77 × 10⁻⁴** — 423× better than best classical

</div>
<div class="opacity-80">

**Learned LSTM Router** — trained to imitate:
- Captures SOR(1.6) dominance (~53%) and per-sample adaptation
- Uses FNO at similar rate (0.6%) to the oracle
- AUC: **0.024** — 15.5× better than best classical, still 27× gap to oracle

</div>
</div>

---
class: text-center
---

<div class="h-full flex flex-col items-center justify-center">
<div class="text-xs tracking-widest uppercase opacity-30 mb-4">Results</div>
<div class="text-4xl font-bold tracking-tight">2D Convection-Diffusion</div>
<div class="mt-4 text-lg opacity-50">Oracle greedy routing — SOR(1.0) + SOR(1.3) + SOR(1.6) + FNO</div>
</div>

---

# 2D ConvDiff: Convergence

<img src="./images/convdiff_fno_convergence.png" class="w-full max-h-72 object-contain rounded-lg border-0" />

<div class="grid grid-cols-3 gap-4 mt-3 text-sm">
<div class="text-center p-3 rounded-lg bg-white/5 border border-white/10">

**Best Classical (SOR 1.6)**
<br><span class="font-mono text-xs">Final L2: 1.41 × 10⁻⁸</span>
<br><span class="font-mono text-xs">AUC: 0.058</span>

</div>
<div class="text-center p-3 rounded-lg bg-white/5 border border-red-400/40">

<span class="text-red-400">**LSTM Router (learned)**</span>
<br><span class="font-mono text-xs">Final L2: 1.41 × 10⁻⁸</span>
<br><span class="font-mono text-xs text-red-400">AUC: 0.020 — 2.9× lower</span>

</div>
<div class="text-center p-3 rounded-lg bg-white/5 border border-emerald-400/40">

<span class="text-emerald-400">**Oracle Greedy**</span>
<br><span class="font-mono text-xs">Final L2: 1.41 × 10⁻⁸</span>
<br><span class="font-mono text-xs text-emerald-400">AUC: 1.78 × 10⁻³ — 33× lower</span>

</div>
</div>

---

# 2D ConvDiff: Routing Patterns

<img src="./images/convdiff_routing_comparison.png" class="w-full max-h-72 object-contain rounded-lg border-0" />

<div class="grid grid-cols-2 gap-6 mt-3 text-sm">
<div class="opacity-80">

**Oracle Greedy** — the upper bound:
- **SOR(1.6)** dominates (~70%), **SOR(1.0)** ~23%, **SOR(1.3)** ~7%
- FNO used sparingly (~0.5%) for targeted corrections
- AUC: **1.78 × 10⁻³** — 33× better than best classical

</div>
<div class="opacity-80">

**Learned LSTM Router** — trained to imitate:
- Captures SOR(1.6) preference (~45%) with broader spread across variants
- FNO usage (0.3%) approaching oracle rate
- AUC: **0.020** — 2.9× better than best classical, still 11× gap to oracle

</div>
</div>

---

# Results Summary

<div class="mt-2">

<div class="text-xs tracking-widest uppercase opacity-40 mb-2">2D Poisson</div>

| | **Best Classical (SOR 1.3)** | **LSTM Router** | **Oracle Greedy** |
|---|---|---|---|
| **AUC** | 0.371 | **0.024** (15.5×↓) | **8.77 × 10⁻⁴** (423×↓) |
| **Final L2** | 3.41 × 10⁻⁶ | 1.13 × 10⁻⁶ | **5.22 × 10⁻⁸** |

</div>

<div class="mt-3">

<div class="text-xs tracking-widest uppercase opacity-40 mb-2">2D Convection-Diffusion</div>

| | **Best Classical (SOR 1.6)** | **LSTM Router** | **Oracle Greedy** |
|---|---|---|---|
| **AUC** | 0.058 | **0.020** (2.9×↓) | **1.78 × 10⁻³** (33×↓) |
| **Final L2** | 1.41 × 10⁻⁸ | 1.41 × 10⁻⁸ | **1.41 × 10⁻⁸** |

</div>

<div class="mt-4 pl-4 border-l-2 border-emerald-400 opacity-80 text-sm">

**Oracle greedy** routing establishes large ceilings on both equations: **423×** (Poisson) and **33×** (ConvDiff) lower AUC than the best classical solver.

</div>

<div class="mt-3 pl-4 border-l-2 border-cyan-400 opacity-80 text-sm">

The **learned LSTM router** captures **15.5×** (Poisson) and **2.9×** (ConvDiff) improvements without privileged information — with significant room to close the gap to the oracle.

</div>


<style>
:root {
  --slidev-theme-primary: #22d3ee;
}
.slidev-layout {
  background: #000 !important;
  color: #e4e4e7 !important;
  overflow: hidden !important;
}
.slidev-page .slidev-layout {
  padding: 2rem 2.5rem !important;
}
.slidev-layout h1 {
  color: #fff !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em !important;
}
.slidev-layout h2, .slidev-layout h3 {
  color: #e4e4e7 !important;
}
.slidev-layout table {
  font-size: 0.8rem;
}
.slidev-layout th {
  background: rgba(255,255,255,0.05) !important;
  border-color: rgba(255,255,255,0.1) !important;
  color: #a1a1aa !important;
  font-weight: 500 !important;
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.05em;
}
.slidev-layout td {
  border-color: rgba(255,255,255,0.06) !important;
  color: #d4d4d8 !important;
}
.slidev-layout tr:hover td {
  background: rgba(255,255,255,0.03);
}
.slidev-layout strong {
  color: #fff;
}
.slidev-layout code {
  background: rgba(255,255,255,0.08) !important;
  color: #22d3ee !important;
  border: none !important;
}
.slidev-layout a {
  color: #22d3ee !important;
}
.katex {
  color: #e4e4e7 !important;
}
</style>
