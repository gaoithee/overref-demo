"""
analysis/generate_report.py

Generate a self-contained HTML report from raw_results CSV files.
Can read from one or multiple results directories and merge them.

Usage
-----
    # Single family
    python analysis/generate_report.py --results-dirs results/olmo1

    # All families combined (default)
    python analysis/generate_report.py

    # Custom output path
    python analysis/generate_report.py --output report.html
"""

import argparse
import json
import pandas as pd
from pathlib import Path


FAMILY_DIRS = {
    "OLMo 1": "results/olmo1",
    "OLMo 2": "results/olmo2",
    "OLMo 3": "results/olmo3",
}


def load_all_results(dirs: dict[str, str]) -> pd.DataFrame:
    """Load and merge raw_results.csv from multiple directories."""
    frames = []
    for family, path in dirs.items():
        csv = Path(path) / "raw_results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["model_family"] = family
            frames.append(df)
            print(f"  Loaded {len(df):,} rows from {csv}")
        else:
            print(f"  Skipping {csv} (not found)")

    if not frames:
        raise RuntimeError("No raw_results.csv found in any of the specified directories.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined):,} rows across {len(frames)} famil(ies)")
    return combined


def generate_report(df: pd.DataFrame, out_path: str) -> None:
    records = []
    for _, row in df.iterrows():
        records.append({
            "prompt":            str(row.get("prompt", "")),
            "category":          str(row.get("category", "")),
            "checkpoint":        str(row.get("checkpoint", "")),
            "model_family":      str(row.get("model_family", "")),
            "response":          str(row.get("response", "")),
            "predicted_refusal": int(row.get("predicted_refusal", 0)),
            "label":             int(row.get("label", 0)),
            "source":            str(row.get("source", "")),
        })

    data_json    = json.dumps(records, ensure_ascii=False)
    checkpoints  = sorted(df["checkpoint"].unique().tolist())
    categories   = sorted(df["category"].unique().tolist())
    families     = sorted(df["model_family"].unique().tolist()) if "model_family" in df.columns else []
    ckpts_json   = json.dumps(checkpoints)
    cats_json    = json.dumps(categories)
    fams_json    = json.dumps(families)

    total = len(df)
    n_families = len(families)
    summary_text = f"{total:,} rows · {n_families} model famil{'y' if n_families==1 else 'ies'} · {len(checkpoints)} checkpoints · {len(categories)} categories"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OLMo Refusal Results</title>
<style>
  :root{{--bg:#0f1117;--surface:#1a1d27;--border:#2a2d3a;--text:#e2e8f0;--muted:#8892a4;
        --blue:#3b82f6;--red:#ef4444;--green:#22c55e;--teal:#14b8a6;--purple:#a855f7;
        --amber:#f59e0b;--font:'JetBrains Mono',monospace}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:system-ui,sans-serif}}
  header{{padding:1.8rem 2.5rem 1.2rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:1.5rem;flex-wrap:wrap}}
  header h1{{font-size:1.3rem;font-weight:700;letter-spacing:-.02em}}
  #summary{{font-size:.8rem;color:var(--muted)}}
  #statsbar{{display:flex;gap:1.5rem;padding:.7rem 2.5rem;border-bottom:1px solid var(--border);font-size:.8rem;color:var(--muted);background:var(--bg);flex-wrap:wrap}}
  .stat{{display:flex;gap:.4rem;align-items:center}}
  .stat span{{color:var(--text);font-weight:600}}
  .dot{{width:8px;height:8px;border-radius:50%}}
  #filters{{padding:.9rem 2.5rem;display:flex;gap:.75rem;flex-wrap:wrap;align-items:center;border-bottom:1px solid var(--border);background:var(--surface);position:sticky;top:0;z-index:10}}
  select,input[type=text]{{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:.3rem .65rem;font-size:.82rem;outline:none}}
  select:focus,input:focus{{border-color:var(--blue)}}
  label{{font-size:.78rem;color:var(--muted)}}
  #container{{padding:1.5rem 2.5rem}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;margin-bottom:.65rem;overflow:hidden}}
  .card.refusal{{border-left:3px solid var(--red)}}
  .card.safe{{border-left:3px solid var(--green)}}
  .card-header{{display:flex;align-items:flex-start;gap:.75rem;padding:.85rem 1.1rem;cursor:pointer}}
  .card-header:hover{{background:rgba(255,255,255,.02)}}
  .badge{{display:inline-block;border-radius:4px;padding:.12rem .45rem;font-size:.68rem;font-weight:600;white-space:nowrap;flex-shrink:0}}
  .badge-cat{{background:#1e2a3a;color:var(--blue)}}
  .badge-ckpt{{background:#1e2a2a;color:var(--teal);font-family:var(--font)}}
  .badge-fam{{background:#2a1e3a;color:var(--purple)}}
  .badge-ref{{background:#2a1e1e;color:var(--red)}}
  .badge-ok{{background:#1e2a1e;color:var(--green)}}
  .prompt-text{{font-size:.87rem;color:var(--text);line-height:1.5;flex:1;min-width:0}}
  .preview{{font-size:.73rem;color:#4a5568;margin-top:.3rem;font-family:var(--font);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  .chevron{{color:var(--muted);font-size:.7rem;flex-shrink:0;transition:transform .2s;margin-top:.2rem}}
  .card.open .chevron{{transform:rotate(90deg)}}
  .card-body{{display:none;padding:0 1.1rem 1rem;border-top:1px solid var(--border)}}
  .card.open .card-body{{display:block}}
  .rlabel{{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin:.7rem 0 .3rem}}
  .rtext{{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:.7rem 1rem;font-family:var(--font);font-size:.76rem;color:#c8d3e0;line-height:1.6;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow-y:auto}}
  .more{{text-align:center;padding:1rem;color:#4a5568;font-size:.78rem}}
</style>
</head>
<body>
<header>
  <h1>OLMo Refusal Results</h1>
  <div id="summary">{summary_text}</div>
</header>
<div id="statsbar">
  <div class="stat"><div class="dot" style="background:var(--red)"></div>Refusals: <span id="s-ref">—</span></div>
  <div class="stat"><div class="dot" style="background:var(--green)"></div>Answered: <span id="s-ans">—</span></div>
  <div class="stat">FP rate: <span id="s-fp">—</span></div>
  <div class="stat">Showing: <span id="s-show">—</span></div>
</div>
<div id="filters">
  <label>Family <select id="f-family"><option value="">All</option></select></label>
  <label>Checkpoint <select id="f-checkpoint"><option value="">All</option></select></label>
  <label>Category <select id="f-category"><option value="">All</option></select></label>
  <label>Refusal
    <select id="f-refusal">
      <option value="">All</option>
      <option value="1">Refused ✗</option>
      <option value="0">Answered ✓</option>
    </select>
  </label>
  <label>Search <input type="text" id="f-search" placeholder="filtra per testo…" style="width:200px"></label>
</div>
<div id="container"></div>
<script>
const DATA={data_json};
const CKS={ckpts_json};
const CATS={cats_json};
const FAMS={fams_json};
const fF=document.getElementById('f-family');
const fC=document.getElementById('f-checkpoint');
const fK=document.getElementById('f-category');
FAMS.forEach(f=>{{const o=document.createElement('option');o.value=f;o.text=f;fF.appendChild(o)}});
CKS.forEach(c=>{{const o=document.createElement('option');o.value=c;o.text=c;fC.appendChild(o)}});
CATS.forEach(c=>{{const o=document.createElement('option');o.value=c;o.text=c;fK.appendChild(o)}});
function esc(s){{return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}}
function trunc(s,n){{s=s.replace(/\\n/g,' ↵ ').trim();return s.length>n?s.slice(0,n)+'…':s}}
function render(){{
  const fam=fF.value,ck=fC.value,cat=fK.value;
  const ref=document.getElementById('f-refusal').value;
  const q=document.getElementById('f-search').value.toLowerCase();
  const f=DATA.filter(r=>
    (!fam||r.model_family===fam)&&(!ck||r.checkpoint===ck)&&
    (!cat||r.category===cat)&&(ref===''||String(r.predicted_refusal)===ref)&&
    (!q||r.prompt.toLowerCase().includes(q))
  );
  const nref=f.filter(r=>r.predicted_refusal).length;
  const safe=f.filter(r=>r.label===0).length;
  const fp_n=f.filter(r=>r.label===0&&r.predicted_refusal).length;
  const fp=safe>0?(fp_n/safe*100).toFixed(1)+'%':'—';
  document.getElementById('s-ref').textContent=nref;
  document.getElementById('s-ans').textContent=f.length-nref;
  document.getElementById('s-fp').textContent=fp;
  document.getElementById('s-show').textContent=f.length.toLocaleString();
  const c=document.getElementById('container');c.innerHTML='';
  f.slice(0,400).forEach(r=>{{
    const isR=r.predicted_refusal===1;
    const d=document.createElement('div');
    d.className='card '+(isR?'refusal':'safe');
    d.innerHTML=`<div class="card-header" onclick="this.parentElement.classList.toggle('open')">
      <span class="chevron">▶</span>
      <div style="flex:1;min-width:0">
        <div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.35rem">
          ${{r.model_family?`<span class="badge badge-fam">${{esc(r.model_family)}}</span>`:''}}
          <span class="badge badge-cat">${{esc(r.category)}}</span>
          <span class="badge badge-ckpt">${{esc(r.checkpoint)}}</span>
          <span class="badge ${{isR?'badge-ref':'badge-ok'}}">${{isR?'✗ refused':'✓ answered'}}</span>
        </div>
        <div class="prompt-text">${{esc(r.prompt)}}</div>
        <div class="preview">${{esc(trunc(r.response,110))}}</div>
      </div>
    </div>
    <div class="card-body">
      <div class="rlabel">Generated response</div>
      <div class="rtext">${{esc(r.response)}}</div>
    </div>`;
    c.appendChild(d);
  }});
  if(f.length>400){{
    const n=document.createElement('div');n.className='more';
    n.textContent=`Showing 400 of ${{f.length.toLocaleString()}}. Use filters to narrow down.`;
    c.appendChild(n);
  }}
}}
['f-family','f-checkpoint','f-category','f-refusal'].forEach(id=>
  document.getElementById(id).addEventListener('change',render));
document.getElementById('f-search').addEventListener('input',render);
render();
</script>
</body></html>"""

    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Report saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dirs", nargs="+", default=None,
        help="One or more results directories. Default: all three (olmo1, olmo2, olmo3)."
    )
    parser.add_argument("--output", default="results/report.html")
    args = parser.parse_args()

    if args.results_dirs:
        # Build name→path mapping from provided paths
        dirs = {Path(p).name: p for p in args.results_dirs}
    else:
        dirs = FAMILY_DIRS

    print("Loading results...")
    df = load_all_results(dirs)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate_report(df, args.output)


if __name__ == "__main__":
    main()