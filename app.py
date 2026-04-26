"""
Flask Web Application — Disease Prediction System
"""

from flask import Flask, request, jsonify, render_template_string
import json
from model import (
    load_model, predict_disease, SYMPTOMS, DISEASES,
    DISEASE_INFO, DISEASE_PROFILES
)

app = Flask(__name__)
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


# ── HTML Template ────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MedScan — AI Disease Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #0a0f1e;
    --surface: #111827;
    --surface2: #1a2236;
    --border: #1e3a5f;
    --accent: #3b82f6;
    --accent2: #06b6d4;
    --text: #e2e8f0;
    --muted: #64748b;
    --danger: #ef4444;
    --success: #22c55e;
    --warn: #f59e0b;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{
    font-family:'DM Sans',sans-serif;
    background:var(--bg);
    color:var(--text);
    min-height:100vh;
    overflow-x:hidden;
  }

  /* Background grid */
  body::before{
    content:'';
    position:fixed;inset:0;
    background-image:
      linear-gradient(rgba(59,130,246,.04) 1px,transparent 1px),
      linear-gradient(90deg,rgba(59,130,246,.04) 1px,transparent 1px);
    background-size:48px 48px;
    pointer-events:none;z-index:0;
  }

  .wrap{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:2rem 1.5rem}

  /* ── Header ── */
  header{text-align:center;padding:3.5rem 0 2rem}
  .logo{
    display:inline-flex;align-items:center;gap:.75rem;
    font-family:'DM Serif Display',serif;font-size:2.6rem;
    background:linear-gradient(135deg,#60a5fa,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
  }
  .logo-icon{font-size:2.2rem;-webkit-text-fill-color:initial}
  .tagline{color:var(--muted);margin-top:.5rem;font-size:.95rem;font-weight:300;letter-spacing:.03em}
  .badge{
    display:inline-block;margin-top:1rem;
    padding:.3rem .9rem;border-radius:99px;
    background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.3);
    font-size:.75rem;color:#60a5fa;letter-spacing:.05em;text-transform:uppercase;
  }

  /* ── Main Layout ── */
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-top:2.5rem}
  @media(max-width:768px){.grid{grid-template-columns:1fr}}

  .card{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:16px;padding:1.75rem;
    transition:border-color .3s;
  }
  .card:hover{border-color:rgba(59,130,246,.4)}
  .card-title{
    font-family:'DM Serif Display',serif;
    font-size:1.15rem;margin-bottom:1.25rem;
    display:flex;align-items:center;gap:.6rem;color:#93c5fd;
  }
  .card-title .icon{font-size:1.1rem}

  /* ── Symptom Search ── */
  .search-box{position:relative;margin-bottom:.75rem}
  .search-box input{
    width:100%;padding:.75rem 1rem .75rem 2.8rem;
    background:var(--surface2);border:1px solid var(--border);
    border-radius:10px;color:var(--text);font-family:inherit;font-size:.9rem;
    outline:none;transition:border-color .2s;
  }
  .search-box input:focus{border-color:var(--accent)}
  .search-box::before{
    content:'🔍';position:absolute;left:.85rem;top:50%;
    transform:translateY(-50%);font-size:.85rem;pointer-events:none;
  }
  #symptom-grid{
    display:flex;flex-wrap:wrap;gap:.45rem;
    max-height:220px;overflow-y:auto;
    padding-right:.25rem;
  }
  #symptom-grid::-webkit-scrollbar{width:4px}
  #symptom-grid::-webkit-scrollbar-track{background:var(--surface2)}
  #symptom-grid::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

  .symptom-tag{
    padding:.35rem .75rem;border-radius:8px;
    border:1px solid var(--border);
    background:var(--surface2);
    cursor:pointer;font-size:.8rem;
    transition:all .18s;user-select:none;
    color:var(--muted);
  }
  .symptom-tag:hover{border-color:var(--accent);color:var(--text)}
  .symptom-tag.active{
    background:rgba(59,130,246,.2);
    border-color:var(--accent);color:#93c5fd;
    font-weight:500;
  }

  /* ── Selected Pills ── */
  #selected-area{margin-top:1rem}
  .selected-label{font-size:.78rem;color:var(--muted);margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.06em}
  #selected-pills{display:flex;flex-wrap:wrap;gap:.4rem;min-height:32px}
  .pill{
    display:flex;align-items:center;gap:.4rem;
    background:rgba(6,182,212,.15);border:1px solid rgba(6,182,212,.4);
    border-radius:99px;padding:.25rem .65rem;
    font-size:.78rem;color:#67e8f9;
  }
  .pill-x{cursor:pointer;opacity:.6;font-size:.9rem;line-height:1}
  .pill-x:hover{opacity:1}

  /* ── Predict Button ── */
  #predict-btn{
    width:100%;margin-top:1.5rem;padding:.9rem;
    background:linear-gradient(135deg,#2563eb,#0891b2);
    border:none;border-radius:10px;color:#fff;
    font-family:inherit;font-size:1rem;font-weight:600;
    cursor:pointer;letter-spacing:.02em;
    transition:opacity .2s,transform .15s;
    position:relative;overflow:hidden;
  }
  #predict-btn:hover{opacity:.9;transform:translateY(-1px)}
  #predict-btn:active{transform:translateY(0)}
  #predict-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
  #predict-btn .spinner{
    display:none;width:18px;height:18px;
    border:2px solid rgba(255,255,255,.3);border-top-color:#fff;
    border-radius:50%;animation:spin .7s linear infinite;
    margin:0 auto;
  }
  #predict-btn.loading .btn-text{display:none}
  #predict-btn.loading .spinner{display:block}
  @keyframes spin{to{transform:rotate(360deg)}}

  .symptom-count{
    text-align:right;font-size:.78rem;color:var(--muted);
    margin-top:.5rem;
  }

  /* ── Results Panel ── */
  #results{display:none}
  .result-card{
    background:var(--surface2);border-radius:12px;
    border:1px solid var(--border);padding:1.25rem;
    margin-bottom:1rem;
    transition:all .3s;animation:slideIn .4s ease forwards;
    opacity:0;transform:translateY(12px);
  }
  @keyframes slideIn{to{opacity:1;transform:translateY(0)}}
  .result-card:nth-child(2){animation-delay:.1s}
  .result-card:nth-child(3){animation-delay:.2s}

  .result-card.top{border-color:rgba(59,130,246,.5);background:rgba(59,130,246,.07)}
  .r-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.75rem}
  .r-name{font-family:'DM Serif Display',serif;font-size:1.2rem;color:var(--text)}
  .r-badge{
    font-size:.72rem;font-weight:600;padding:.25rem .65rem;
    border-radius:99px;white-space:nowrap;
  }
  .confidence-bar{
    height:6px;background:var(--border);border-radius:3px;
    margin-bottom:.75rem;overflow:hidden;
  }
  .confidence-fill{
    height:100%;border-radius:3px;
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    transition:width 1s cubic-bezier(.4,0,.2,1);
    width:0;
  }
  .conf-label{font-size:.8rem;color:var(--muted);margin-bottom:.35rem}
  .r-desc{font-size:.85rem;color:var(--muted);margin-bottom:.9rem;line-height:1.6}

  .info-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}
  @media(max-width:480px){.info-grid{grid-template-columns:1fr}}
  .info-box{background:var(--surface);border-radius:8px;padding:.75rem}
  .info-box-title{font-size:.73rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.5rem}
  .info-list{list-style:none}
  .info-list li{
    font-size:.82rem;padding:.2rem 0;
    border-bottom:1px solid rgba(255,255,255,.04);
    display:flex;align-items:flex-start;gap:.4rem;
    color:#cbd5e1;
  }
  .info-list li::before{content:'•';color:var(--accent);flex-shrink:0;margin-top:.05em}
  .info-list li:last-child{border-bottom:none}

  /* ── Stats Panel ── */
  .stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-bottom:1.5rem}
  .stat-box{
    background:var(--surface2);border:1px solid var(--border);
    border-radius:10px;padding:1rem;text-align:center;
  }
  .stat-num{font-family:'DM Serif Display',serif;font-size:1.9rem;
    background:linear-gradient(135deg,#60a5fa,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
  }
  .stat-label{font-size:.72rem;color:var(--muted);margin-top:.2rem;text-transform:uppercase;letter-spacing:.06em}

  /* ── Disclaimer ── */
  .disclaimer{
    margin-top:2.5rem;padding:1rem 1.25rem;
    background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.2);
    border-radius:10px;font-size:.82rem;color:#fca5a5;line-height:1.6;
  }
  .disclaimer strong{color:var(--danger)}

  /* ── Empty state ── */
  .empty-state{
    text-align:center;padding:3rem 1rem;
    color:var(--muted);
  }
  .empty-icon{font-size:3rem;margin-bottom:1rem;opacity:.4}
  .empty-text{font-size:.9rem}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar{width:6px}
  ::-webkit-scrollbar-track{background:var(--bg)}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo"><span class="logo-icon">🧬</span> MedScan</div>
    <p class="tagline">AI-Powered Disease Prediction from Symptoms</p>
    <div class="badge">Random Forest · 15 Diseases · 30 Symptoms</div>
  </header>

  <div class="stats-grid">
    <div class="stat-box"><div class="stat-num">15</div><div class="stat-label">Diseases</div></div>
    <div class="stat-box"><div class="stat-num">30</div><div class="stat-label">Symptoms</div></div>
    <div class="stat-box"><div class="stat-num">95%+</div><div class="stat-label">Accuracy</div></div>
  </div>

  <div class="grid">
    <!-- ── Left: Symptom Selector ── -->
    <div class="card">
      <div class="card-title"><span class="icon">🩺</span>Select Your Symptoms</div>

      <div class="search-box">
        <input type="text" id="search-input" placeholder="Search symptoms…" oninput="filterSymptoms(this.value)"/>
      </div>

      <div id="symptom-grid"></div>

      <div class="selected-area" id="selected-area">
        <div class="selected-label">Selected symptoms</div>
        <div id="selected-pills"></div>
        <div class="symptom-count" id="count-label">0 symptoms selected</div>
      </div>

      <button id="predict-btn" onclick="predict()" disabled>
        <span class="btn-text">🔍 Predict Disease</span>
        <div class="spinner"></div>
      </button>
    </div>

    <!-- ── Right: Results ── -->
    <div class="card">
      <div class="card-title"><span class="icon">📋</span>Prediction Results</div>
      <div id="results-container">
        <div class="empty-state">
          <div class="empty-icon">🫀</div>
          <div class="empty-text">Select at least 2 symptoms<br>and click Predict</div>
        </div>
      </div>
    </div>
  </div>

  <div class="disclaimer">
    <strong>⚠️ Medical Disclaimer:</strong> This tool is for <strong>educational purposes only</strong> and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
  </div>
</div>

<script>
const SYMPTOMS = {{ symptoms | tojson }};
let selected = new Set();

// Build symptom tags
function buildGrid(filter='') {
  const grid = document.getElementById('symptom-grid');
  grid.innerHTML = '';
  const fl = filter.toLowerCase();
  SYMPTOMS.forEach(s => {
    const label = s.replace(/_/g,' ');
    if (fl && !label.includes(fl)) return;
    const tag = document.createElement('div');
    tag.className = 'symptom-tag' + (selected.has(s) ? ' active' : '');
    tag.textContent = label;
    tag.onclick = () => toggleSymptom(s, tag);
    grid.appendChild(tag);
  });
}

function filterSymptoms(v) { buildGrid(v); }

function toggleSymptom(s, el) {
  if (selected.has(s)) {
    selected.delete(s);
    el.classList.remove('active');
  } else {
    selected.add(s);
    el.classList.add('active');
  }
  renderPills();
  document.getElementById('predict-btn').disabled = selected.size < 1;
}

function renderPills() {
  const c = document.getElementById('selected-pills');
  c.innerHTML = '';
  selected.forEach(s => {
    const pill = document.createElement('div');
    pill.className = 'pill';
    pill.innerHTML = `<span>${s.replace(/_/g,' ')}</span><span class="pill-x" onclick="removeSymptom('${s}')">×</span>`;
    c.appendChild(pill);
  });
  document.getElementById('count-label').textContent = `${selected.size} symptom${selected.size!==1?'s':''} selected`;
}

function removeSymptom(s) {
  selected.delete(s);
  buildGrid(document.getElementById('search-input').value);
  renderPills();
  document.getElementById('predict-btn').disabled = selected.size < 1;
}

async function predict() {
  const btn = document.getElementById('predict-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({symptoms: [...selected]})
    });
    const data = await res.json();
    renderResults(data.results);
  } catch(e) {
    document.getElementById('results-container').innerHTML =
      '<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-text">Error occurred. Please try again.</div></div>';
  }

  btn.classList.remove('loading');
  btn.disabled = false;
}

function renderResults(results) {
  if (!results || !results.length) return;
  const c = document.getElementById('results-container');
  c.innerHTML = '';

  results.forEach((r, i) => {
    const card = document.createElement('div');
    card.className = 'result-card' + (i===0 ? ' top' : '');

    const severityBg = r.severity_color + '22';
    const precautionItems = r.precautions.map(p => `<li>${p}</li>`).join('');
    const medicineItems   = r.medicines.map(m => `<li>${m}</li>`).join('');
    const rankEmoji = ['🥇','🥈','🥉'][i] || '';

    card.innerHTML = `
      <div class="r-header">
        <div class="r-name">${rankEmoji} ${r.disease}</div>
        <span class="r-badge" style="background:${severityBg};border:1px solid ${r.severity_color}44;color:${r.severity_color}">${r.severity}</span>
      </div>
      <div class="conf-label">Confidence: <strong style="color:#93c5fd">${r.confidence}%</strong></div>
      <div class="confidence-bar"><div class="confidence-fill" data-w="${r.confidence}"></div></div>
      <div class="r-desc">${r.description}</div>
      <div class="info-grid">
        <div class="info-box">
          <div class="info-box-title">🛡️ Precautions</div>
          <ul class="info-list">${precautionItems}</ul>
        </div>
        <div class="info-box">
          <div class="info-box-title">💊 Medicines</div>
          <ul class="info-list">${medicineItems}</ul>
        </div>
      </div>`;

    c.appendChild(card);

    // Animate bar
    setTimeout(() => {
      card.querySelector('.confidence-fill').style.width = r.confidence + '%';
    }, 100 + i * 150);
  });
}

buildGrid();
</script>
</body>
</html>
"""


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML, symptoms=SYMPTOMS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    model = get_model()
    results = predict_disease(symptoms, model)
    return jsonify({"results": results})


@app.route("/symptoms")
def get_symptoms():
    return jsonify({"symptoms": SYMPTOMS, "diseases": DISEASES})


@app.route("/info/<disease>")
def disease_info(disease):
    info = DISEASE_INFO.get(disease)
    if not info:
        return jsonify({"error": "Disease not found"}), 404
    return jsonify(info)


if __name__ == "__main__":
    print("🚀 Starting MedScan Disease Prediction Server...")
    print("📡 Loading ML model...")
    get_model()
    print("✅ Ready! Open http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
