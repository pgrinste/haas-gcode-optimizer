# Closed-Loop AI G-Code Optimizer & Telemetry Agent

A local LLM agent iteratively rewrites a Haas VF-2 program while a PyTorch physics
evaluator checks **every block** against the machine's real spindle torque curve and
axis acceleration limits. The loop runs until the toolpath is feasible — then it keeps
the fastest feasible version it ever found, not just the last one.

Built for the HAAS job fair (Las Vegas). One notebook, no cloud dependencies:
Ollama serves the LLM locally, the physics runs on CPU tensors.

## Results

`data/sample_program.gcode` is a deliberately bad CAM export: full-width slotting roughing
(T1) followed by an aggressive single-pass deep contour (T2) at HSM defaults. The baseline
**redlines the spindle at 180.6% of the VF-2 envelope** and smashes the lateral acceleration
limit ~14× at four corners.

| metric | baseline | LLM agent (Ollama, Qwen3) | heuristic reference |
|---|---:|---:|---:|
| cycle time | 6.19 min | **3.16 min** | **1.30 min** |
| peak spindle utilization | 180.6% | 90.4% | 95.0% |
| infeasible blocks | 8 | 0 | 0 |
| sustained MRR (mm³/min) | 41,267 | 79,192 | ~190k peak |

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
  keeps the fastest *verified feasible* program, so the output is never worse than the first
  fix. You can see this happen live in the executed notebook (iterations 3–4 regress, and the
  iteration-2 state wins).

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
requirements.txt                    pinned-ish deps
```

## Sources

- Haas VF-2 / VF-2V specification tables — haascnc.com (spindle, traverse, feed limits)
- Haas VF Series Service Manual (axis acceleration ≈ 60 in/s²)
- Kalpakjian & Schmid, *Manufacturing Engineering and Technology* — specific cutting energy
  for aluminum alloys (0.4–1.1 J/mm³)
