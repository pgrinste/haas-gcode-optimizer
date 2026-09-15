# Closed-Loop AI G-Code Optimizer & Telemetry Agent

A local LLM agent iteratively rewrites a Haas VF-2 program while a PyTorch physics
evaluator checks **every block** against the machine's real spindle torque curve and
axis acceleration limits. The loop runs until the toolpath is feasible — then it keeps
the fastest feasible version it ever found, not just the last one.

One notebook, no cloud dependencies:
Ollama serves the LLM locally, the physics runs on CPU tensors.

## Results

`data/sample_program.gcode` is a deliberately bad CAM export: full-width slotting roughing
(T1) followed by an aggressive single-pass deep contour (T2) at HSM defaults. The baseline
**redlines the spindle at 180.6% of the VF-2 envelope** and smashes the lateral acceleration
limit ~14× at four corners.

| metric | baseline | LLM agent (Ollama, Qwen3) | heuristic reference |
|---|---:|---:|---:|
| cycle time | 6.19 min | **2.98 min** | **1.30 min** |
| peak spindle utilization | 180.6% | 94.2% | 95.0% |
| infeasible blocks | 8 | 1 | 0 |
| sustained MRR (mm³/min) | 41,267 | 83,814 | ~190k peak |

The agent's edits are plain G-code changes — feed-rate overrides and inserted trochoidal-style
entry arcs (`G2`/`G3` with computed `R`) at sharp junctions. The optimized program is written
to `data/optimized_program.gcode`; diff it against the input to see exactly what changed.

## How it works

```
 raw G-code ──> modal-state parser (Haas-style, G20)
                        │  blocks + per-pass AP/AE from (PASS ...) comments
                        ▼
              trajectory builder (segments, corner angles, arc insertion)
                        │
                        ▼
        PyTorch physics evaluator   ←── VF-2 spindle envelope:
          · MRR = A·v_z per block         T(n) = min(122 N·m, 22.4 kW→T(n))
          · P = u·A·f / η per cut         axis accel limit 60 in/s²
          · corner kinematics (a_lat)     base speed ≈ 1753 rpm
                        │
                        ▼
              telemetry JSON ──> LLM agent (Ollama, local Qwen3)
                        │        or deterministic heuristic fallback
                        ▼
             edits: set_feed / insert_arc ──> re-parse ──> re-evaluate
                        │  repeat until feasible; keep best-so-far
                        ▼
              optimized program + before/after plots
```

Design notes worth talking about:

- **G-code under-specifies the cut.** Geometry alone can't tell you axial depth or radial
  engagement, so the parser reads structured `(PASS T2 AP 0.500 AE 1.000 CLIMB)` comments —
  something real CAM systems actually emit.
- **Two-region spindle model.** Constant torque below base speed (~1753 rpm for the VF-2),
  constant power above it. Violations are reported as % of available torque/power at each
  block's actual spindle speed, not against a single rated number.
- **CPU torch is deliberate.** Under ~5k blocks the vectorized physics takes milliseconds;
  the CPU wheel avoids CUDA DLL load risk, and Ollama owns the GPUs for inference.
- **The LLM is a proposal engine, not the source of truth.** Every edit it emits goes back
  through the same physics evaluator. A deterministic closed-form heuristic implements the
  identical edit contract as a fallback (and as a reference run in the notebook).
- **Best-so-far selection.** Sampling variance means later iterations can over-tune; the loop
  keeps the fastest *verified feasible* program it ever found — and if no iteration reaches
  full feasibility within budget, the closest-to-feasible state (fewest violations), so the
  output is never worse than its best attempt. You can see this happen live in the executed
  notebook: iterations 5–6 over-tune to 21 and 63 violations, and the loop returns the
  iteration-4 state with a single remaining violation.
- **Machine profile as data.** Spindle envelope, axis limits, tool table and chip-load caps
  live in `data/machine_profiles/vf2.json`; the notebook loads them at setup. Adding another
  machine is a file drop, not a code edit — that's the path to a multi-machine optimizer.

## Roadmap

- **Physics fidelity:** entry dynamics, tool wear and chatter are out of scope for v1; the
  evaluator assumes constant specific cutting energy and checks corners as single-point
  lateral-acceleration events.
- **Multi-machine:** drop another profile into `data/machine_profiles/` (a VF-4, a 5-axis,
  whatever) — the loop is machine-agnostic given the JSON schema.
- **CAM integration:** wrap the pipeline as a post-processor plugin for Fusion 360 / Mastercam
  so the agent runs without Jupyter.

## Machine data used

Haas VF-2 / CT40 direct drive: 30 hp (22.4 kW) spindle, 8100 rpm max, 90 ft·lb (122 N·m)
max torque at 2000 rpm, rapid traverse 1000 ipm, axis acceleration ≈ 60 in/s²
(1524 mm/s²). Specific cutting energy for 6061-T6: u = 0.8 J/mm³ (midpoint of the
0.4–1.1 J/mm³ range for aluminum alloys). Sources at the bottom of the notebook.

## Quickstart

```bat
py -3.13 -m venv .venv
.venv\Scripts\pip install numpy matplotlib
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then open `haas_gcode_optimizer.ipynb` in Jupyter from the repo root and run top to bottom
(~2 min with a warm Ollama model; ~1 min without). If Ollama isn't running, set
`AGENT = "heuristic"` in the config cell — the full pipeline still runs.

Headless execution (what was used to produce `haas_gcode_optimizer_executed.ipynb`):

```bash
.venv/Scripts/pip install nbclient ipykernel
.venv/Scripts/python -m ipykernel install --user --name resume_venv
# then any nbclient driver with kernel_name="resume_venv"
```

Ollama: `ollama pull qwen3` (or any model that returns clean JSON); the notebook probes
`http://localhost:11434/api/tags` and falls back automatically.

## Repo layout

```
haas_gcode_optimizer.ipynb          the whole build, narrated cell by cell
haas_gcode_optimizer_executed.ipynb same notebook with outputs (no need to run it)
data/sample_program.gcode           raw CAM export — the "bad" starting program
data/optimized_program.gcode        agent output from the executed run
data/machine_profiles/vf2.json      machine limits + tool library (JSON profile)
requirements.txt                    pinned-ish deps
```

## Sources

- Haas VF-2 / VF-2V specification tables — haascnc.com (spindle, traverse, feed limits)
- Haas VF Series Service Manual (axis acceleration ≈ 60 in/s²)
- Kalpakjian & Schmid, *Manufacturing Engineering and Technology* — specific cutting energy
  for aluminum alloys (0.4–1.1 J/mm³)
