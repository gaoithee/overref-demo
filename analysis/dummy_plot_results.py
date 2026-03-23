import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configurazione Macro-Categorie ---
# Mappatura basata sul valore della colonna 'source'
SAFETY_SOURCES = [
    "toxicchat", "wildguard", "beavertails", 
    "harmbench", "jailbreakbench", "harmful", "refusal", "safety"
] 

UTILITY_SOURCES = [
    "or_bench", "orbench", "false_reject", "falsereject", 
    "utility", "safe_prompts"
]

STAGE_ORDER = ["base", "sft", "dpo", "final"]
STAGE_LABELS = {"base": "Base", "sft": "SFT", "dpo": "DPO", "final": "Instruct"}

def _split_tag(tag):
    parts = str(tag).split("__", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "none")

def _get_metric_group(source_name):
    s = str(source_name).lower()
    if any(k in s for k in SAFETY_SOURCES):
        return "Safety (Correct Refusal) ↑"
    if any(k in s for k in UTILITY_SOURCES):
        return "Over-refusal (Error) ↓"
    return "Other"

def run_analysis(results_dir):
    rd = Path(results_dir)
    raw_file = rd / "raw_results.csv"
    plots_dir = rd / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    if not raw_file.exists():
        print(f"❌ Errore: File {raw_file} non trovato!")
        return

    # 1. Caricamento dati grezzi
    print(f"🔍 Caricamento dati da {raw_file}...")
    df = pd.read_csv(raw_file)
    
    # Verifica colonne necessarie
    required_cols = ["source", "checkpoint", "predicted_refusal"]
    if not all(c in df.columns for c in required_cols):
        print(f"❌ Errore: Il CSV deve contenere le colonne {required_cols}")
        return

    # 2. Preprocessing
    # Estraiamo stage e prompt dalla colonna checkpoint
    df[["stage", "prompt"]] = pd.DataFrame(
        df["checkpoint"].apply(_split_tag).tolist(), index=df.index
    )
    
    # Assegniamo la macro-categoria basandoci sulla colonna 'source'
    df["metric_group"] = df["source"].apply(_get_metric_group)
    
    # Filtriamo per gli stage che ci interessano
    df = df[df["stage"].isin(STAGE_ORDER)].copy()

    # 3. Calcolo dei tassi di rifiuto (Refusal Rate)
    # Raggruppiamo per dataset specifico (source), stage e prompt
    detailed_stats = df.groupby(["source", "stage", "prompt", "metric_group"])["predicted_refusal"].mean().reset_index()
    detailed_stats["refusal_rate"] = detailed_stats["predicted_refusal"] * 100

    sns.set_theme(style="whitegrid")

    # --- PLOT 1: Dettaglio per 'source' (Linee) ---
    g_detail = sns.relplot(
        data=detailed_stats, x="stage", y="refusal_rate", hue="prompt",
        col="source", col_wrap=3, kind="line", marker="o",
        facet_kws={'sharey': False}, height=4, aspect=1.2
    )
    g_detail.set_titles("{col_name}", size=11, weight='bold')
    g_detail.fig.suptitle(f"Dettaglio per Sorgente (Source): {rd.name.upper()}", y=1.05, fontsize=16)
    g_detail.savefig(plots_dir / "source_breakdown_lines.png", bbox_inches="tight", dpi=150)

    # --- PLOT 2: Sintesi Aggregata (Barre) ---
    summary = detailed_stats.groupby(["metric_group", "stage", "prompt"])["refusal_rate"].mean().reset_index()
    
    g_summary = sns.catplot(
        data=summary, x="stage", y="refusal_rate", hue="prompt",
        col="metric_group", kind="bar", palette="muted",
        order=STAGE_ORDER, height=5, aspect=1.2, sharey=False
    )
    g_summary.set_titles("{col_name}", size=14, weight='bold')
    
    # Annotazioni numeriche
    for ax in g_summary.axes.flat:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 8), 
                            textcoords='offset points', fontsize=8, fontweight='bold')

    g_summary.savefig(plots_dir / "summary_metrics_bars.png", bbox_inches="tight", dpi=180)

    # 4. Report nel Terminale
    print(f"\n✅ Analisi completata per {rd.name}!")
    
    print(f"\n--- 1. REPORT GRANULARE PER SOURCE (Dataset originario) ---")
    detailed_pivot = detailed_stats.pivot_table(
        index=["stage", "prompt"], 
        columns="source", 
        values="refusal_rate"
    )
    print(detailed_pivot.round(2))

    print(f"\n--- 2. SINTESI AGGREGATA PER GRUPPO ---")
    summary_pivot = summary.pivot_table(index=["stage", "prompt"], columns="metric_group", values="refusal_rate")
    print(summary_pivot.round(2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    run_analysis(args.results_dir)