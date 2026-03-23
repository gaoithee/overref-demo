import argparse
import json
import pandas as pd
from pathlib import Path

# Configurazione cartelle
FAMILY_DIRS = {
#    "OLMo 1": "results/olmo1",
    "OLMo 2": "results/olmo2",
    "OLMo 3": "results/olmo3",
    "OLMo 3 - think": "results/olmo3_think",
}

def load_data():
    frames = []
    for name, folder in FAMILY_DIRS.items():
        p = Path(folder) / "raw_results.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["family"] = name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None

def write_report(df, out_path):
    # Sanificazione dati: togliamo i NaN che rompono il JS
    df = df.fillna("")
    records = df.to_dict(orient="records")
    
    # Costruiamo l'HTML come una stringa secca
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Report OLMo</title>
    <style>
        body { font-family: sans-serif; background: #111; color: #eee; padding: 20px; }
        .controls { position: sticky; top: 0; background: #222; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #444; }
        input { width: 100%; padding: 10px; background: #000; color: #0f0; border: 1px solid #444; }
        .card { background: #1a1a1a; border: 1px solid #333; padding: 15px; margin-bottom: 10px; border-radius: 5px; }
        .card.refusal { border-left: 6px solid #f44; }
        .card.safe { border-left: 6px solid #4f4; }
        .meta { font-size: 12px; color: #aaa; margin-bottom: 8px; }
        .tag { background: #333; padding: 2px 6px; border-radius: 3px; margin-right: 5px; }
        .prompt { font-weight: bold; color: #fff; margin: 10px 0; display: block; }
        pre { background: #000; padding: 10px; white-space: pre-wrap; color: #ccc; border-radius: 4px; font-size: 13px; }
    </style>
</head>
<body>
    <div class="controls">
        <input type="text" id="search" placeholder="Cerca nel prompt... (es. 'shogun' o 'hack')" oninput="update()">
        <div id="stats" style="margin-top:10px; font-size:12px; color:#888;"></div>
    </div>
    <div id="list"></div>

    <script>
        // DATI INIETTATI DA PYTHON
        const dataset = __REPLACE_ME__;

        function update() {
            const q = document.getElementById('search').value.toLowerCase();
            const list = document.getElementById('list');
            
            const filtered = dataset.filter(r => 
                String(r.prompt).toLowerCase().includes(q)
            ).slice(0, 50); // Mostriamo i primi 50 per non inchiodare il browser

            document.getElementById('stats').innerText = `Trovati ${filtered.length} risultati su ${dataset.length}`;

            list.innerHTML = filtered.map(r => `
                <div class="card ${r.predicted_refusal ? 'refusal' : 'safe'}">
                    <div class="meta">
                        <span class="tag">${r.family}</span>
                        <span class="tag">${r.checkpoint}</span>
                        <span class="tag">${r.source}</span>
                    </div>
                    <span class="prompt">${r.prompt}</span>
                    <pre>${r.response}</pre>
                </div>
            `).join('');
        }
        
        // Lancio iniziale
        update();
    </script>
</body>
</html>
"""
    # Sostituzione sicura del placeholder con il JSON vero
    final_output = html.replace("__REPLACE_ME__", json.dumps(records, ensure_ascii=False))
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_output)

if __name__ == "__main__":
    data = load_data()
    if data is not None:
        write_report(data, "results/report.html")
        print("✅ Report creato in results/report.html")
    else:
        print("❌ Nessun dato trovato. Controlla i percorsi in FAMILY_DIRS.")