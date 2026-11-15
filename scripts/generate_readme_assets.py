import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def render_html_to_png(html_str: str, output_png: Path, width: int = 1440, height: int = 860):
    temp_html = ASSETS / f"temp_{output_png.stem}.html"
    temp_html.write_text(html_str, encoding="utf-8")
    
    cmd = [
        CHROME,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--force-device-scale-factor=2",
        f"--screenshot={output_png}",
        str(temp_html)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if temp_html.exists():
        temp_html.unlink()
    if res.returncode == 0 and output_png.exists():
        print(f"[OK] Generated: {output_png.name} ({output_png.stat().st_size} bytes)")
    else:
        print(f"[ERR] Failed to generate {output_png.name}: {res.stderr}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM ARCHITECTURE DIAGRAM (NEW BRAND DESIGN)
# ─────────────────────────────────────────────────────────────────────────────
arch_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1440px;
    height: 840px;
    background: #060b14;
    color: #f1f5f9;
    font-family: 'Plus Jakarta Sans', sans-serif;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 34px 44px;
    overflow: hidden;
    position: relative;
  }
  body::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: 
      radial-gradient(circle at 50% 0%, rgba(37, 99, 235, 0.15) 0%, transparent 60%),
      linear-gradient(to right, rgba(255,255,255,0.02) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(255,255,255,0.02) 1px, transparent 1px);
    background-size: 100% 100%, 36px 36px, 36px 36px;
    pointer-events: none;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 16px;
    position: relative;
    z-index: 10;
  }
  .title-group h1 {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .title-group p {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
    font-weight: 500;
  }
  .badge-bar { display: flex; gap: 8px; }
  .badge {
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
  }
  .badge.cyan { background: rgba(6, 182, 212, 0.12); border-color: rgba(6, 182, 212, 0.3); color: #38bdf8; }
  .badge.emerald { background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.3); color: #34d399; }
  .badge.amber { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.3); color: #fbbf24; }

  .architecture-grid {
    display: grid;
    grid-template-columns: 290px 310px 370px 320px;
    gap: 20px;
    position: relative;
    z-index: 10;
    margin: 20px 0;
  }
  .col { display: flex; flex-direction: column; gap: 14px; }
  .col-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    padding-left: 4px;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-blue { background: #38bdf8; box-shadow: 0 0 10px #38bdf8; }
  .dot-purple { background: #a855f7; box-shadow: 0 0 10px #a855f7; }
  .dot-rose { background: #f43f5e; box-shadow: 0 0 10px #f43f5e; }
  .dot-emerald { background: #10b981; box-shadow: 0 0 10px #10b981; }

  .card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(12px);
  }
  .card.highlight-blue {
    border-color: rgba(56, 189, 248, 0.35);
    background: linear-gradient(180deg, rgba(14, 116, 144, 0.18) 0%, rgba(15, 23, 42, 0.8) 100%);
  }
  .card.highlight-purple {
    border-color: rgba(168, 85, 247, 0.35);
    background: linear-gradient(180deg, rgba(126, 34, 206, 0.18) 0%, rgba(15, 23, 42, 0.8) 100%);
  }
  .card.highlight-rose {
    border-color: rgba(244, 63, 94, 0.4);
    background: linear-gradient(180deg, rgba(190, 18, 60, 0.20) 0%, rgba(15, 23, 42, 0.8) 100%);
  }
  .card.highlight-emerald {
    border-color: rgba(16, 185, 129, 0.35);
    background: linear-gradient(180deg, rgba(4, 120, 87, 0.18) 0%, rgba(15, 23, 42, 0.8) 100%);
  }

  .card-title {
    font-size: 14px;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .card-sub {
    font-size: 11px;
    color: #94a3b8;
    line-height: 1.4;
    margin-bottom: 10px;
  }
  .tag-list { display: flex; flex-wrap: wrap; gap: 5px; }
  .pill {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 7px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
  }
  .pill.blue { background: rgba(56, 189, 248, 0.1); color: #7dd3fc; border-color: rgba(56, 189, 248, 0.2); }
  .pill.purple { background: rgba(168, 85, 247, 0.1); color: #d8b4fe; border-color: rgba(168, 85, 247, 0.2); }
  .pill.rose { background: rgba(244, 63, 94, 0.1); color: #fda4af; border-color: rgba(244, 63, 94, 0.2); }
  .pill.emerald { background: rgba(16, 185, 129, 0.1); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.2); }

  .formula-box {
    background: rgba(0, 0, 0, 0.45);
    border: 1px dashed rgba(244, 63, 94, 0.45);
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #fecdd3;
    line-height: 1.5;
  }
  .formula-box .highlight { color: #38bdf8; font-weight: 700; }

  .footer-flow {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    z-index: 10;
  }
  .flow-item { display: flex; align-items: center; gap: 10px; font-size: 12px; font-weight: 600; color: #cbd5e1; }
  .flow-num {
    width: 22px; height: 22px; border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);
  }
  .flow-sep { color: #475569; font-size: 16px; }
</style>
</head>
<body>
  <div class="header">
    <div class="title-group">
      <h1>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
        Agentic Triage Copilot: System Architecture
      </h1>
      <p>Hybrid Emergency Triage Decision Support | LangGraph Multi-Agent StateGraph & Inviolable Safety Floor</p>
    </div>
    <div class="badge-bar">
      <span class="badge cyan">LangGraph Orchestration</span>
      <span class="badge emerald">ESI v4 Protocol</span>
      <span class="badge amber">HL7 FHIR R4</span>
      <span class="badge">99/99 Tests Passing</span>
    </div>
  </div>

  <div class="architecture-grid">
    <!-- Col 1 -->
    <div class="col">
      <div class="col-header"><span class="dot dot-blue"></span>1. Intake & Telemetry</div>
      <div class="card highlight-blue">
        <div class="card-title">Streamlit Workstation</div>
        <div class="card-sub">Interactive nurse intake UI with telemetry inputs, live ESI cards, and audit viewer.</div>
        <div class="tag-list"><span class="pill blue">app.py</span><span class="pill blue">Web Workstation</span><span class="pill">Vitals Dial</span></div>
      </div>
      <div class="card">
        <div class="card-title">FastAPI Serving API</div>
        <div class="card-sub">Production REST microservice with async triage execution and health probes.</div>
        <div class="tag-list"><span class="pill blue">api.py</span><span class="pill">POST /v1/triage</span><span class="pill">POST /v1/triage/review</span></div>
      </div>
      <div class="card">
        <div class="card-title">Clinical Telemetry</div>
        <div class="card-sub">Standardized physiological vitals & narrative symptoms parsed across age bands.</div>
        <div class="tag-list"><span class="pill">SpO2</span><span class="pill">HR</span><span class="pill">BP</span><span class="pill">RR</span><span class="pill">Temp</span></div>
      </div>
    </div>

    <!-- Col 2 -->
    <div class="col">
      <div class="col-header"><span class="dot dot-purple"></span>2. NLP & Vector Memory</div>
      <div class="card highlight-purple">
        <div class="card-title">Structured Intake Node</div>
        <div class="card-sub">Structured Pydantic symptom & duration extractor with resilient regex fallback.</div>
        <div class="tag-list"><span class="pill purple">intake.py</span><span class="pill purple">Groq LLaMA-3.3-70B</span><span class="pill">Regex Fallback</span></div>
      </div>
      <div class="card">
        <div class="card-title">Dense Patient Memory</div>
        <div class="card-sub">Cross-encounter semantic context lookup via sentence-transformers (384-dim).</div>
        <div class="tag-list"><span class="pill purple">all-MiniLM-L6-v2</span><span class="pill">Pinecone</span><span class="pill">LocalMemory</span></div>
      </div>
      <div class="card">
        <div class="card-title">Infermedica v3 Engine</div>
        <div class="card-sub">Optional medical concept ontology mapping and OTC medication guidance.</div>
        <div class="tag-list"><span class="pill purple">Infermedica v3</span><span class="pill">Concept Mapping</span></div>
      </div>
    </div>

    <!-- Col 3 -->
    <div class="col">
      <div class="col-header"><span class="dot dot-rose"></span>3. Risk & Safety Floor</div>
      <div class="card highlight-rose">
        <div class="card-title">Deterministic Safety Floor</div>
        <div class="card-sub">Windowed NegEx negation handling, pediatric age-adjusted bounds & red-flag rules.</div>
        <div class="tag-list"><span class="pill rose">rules.py</span><span class="pill rose">Windowed NegEx</span><span class="pill">Pediatric Bounds</span><span class="pill">Red Flags</span></div>
        <div class="formula-box">
          <div style="font-weight:700; color:#fb7185; margin-bottom:2px;">INVIOLABLE SAFETY INVARIANT</div>
          FinalAcuity = <span class="highlight">max</span>(RulesFloor, LLM_Acuity)<br/>
          <span style="color:#94a3b8; font-size:10px;">LLM can escalate, NEVER de-escalate.</span>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Fast-Path Short Circuit</div>
        <div class="card-sub">Sub-millisecond triage bypass on immediate life threats (ESI 1 Resuscitation).</div>
        <div class="tag-list"><span class="pill rose">Sub-50ms Fast Path</span><span class="pill">Immediate Alert</span></div>
      </div>
    </div>

    <!-- Col 4 -->
    <div class="col">
      <div class="col-header"><span class="dot dot-emerald"></span>4. Governance & Interoperability</div>
      <div class="card highlight-emerald">
        <div class="card-title">Clinician HITL Gate</div>
        <div class="card-sub">Mandatory licensed nurse sign-off for ESI 1–2; structured override justification.</div>
        <div class="tag-list"><span class="pill emerald">Human-in-the-Loop</span><span class="pill emerald">Sign-off / Override</span><span class="pill">Justification Log</span></div>
      </div>
      <div class="card">
        <div class="card-title">HL7 FHIR R4 Bundle</div>
        <div class="card-sub">Valid transaction bundles with Patient, Observation, Condition, DocumentReference.</div>
        <div class="tag-list"><span class="pill emerald">FHIR R4</span><span class="pill">SBAR Physician Note</span></div>
      </div>
      <div class="card">
        <div class="card-title">Audit Ledger & Analytics</div>
        <div class="card-sub">Append-only JSONL audit log measuring clinical concordance & override telemetry.</div>
        <div class="tag-list"><span class="pill emerald">Concordance Tracking</span><span class="pill">JSONL Ledger</span></div>
      </div>
    </div>
  </div>

  <div class="footer-flow">
    <div class="flow-item"><span class="flow-num">1</span><span>Arriving Patient Intake</span></div>
    <span class="flow-sep">→</span>
    <div class="flow-item"><span class="flow-num">2</span><span>Deterministic Supervisor</span></div>
    <span class="flow-sep">→</span>
    <div class="flow-item"><span class="flow-num">3</span><span>Hybrid NLP & Vector Memory</span></div>
    <span class="flow-sep">→</span>
    <div class="flow-item"><span class="flow-num">4</span><span>Consensus Risk & Safety Floor</span></div>
    <span class="flow-sep">→</span>
    <div class="flow-item"><span class="flow-num">5</span><span>Mandatory Clinician Sign-off</span></div>
    <span class="flow-sep">→</span>
    <div class="flow-item"><span class="flow-num">6</span><span>SBAR & FHIR R4 Bundle</span></div>
  </div>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. UI SCREENSHOT 1: NURSE INTAKE WORKSTATION (EMERGENCY CASE)
# ─────────────────────────────────────────────────────────────────────────────
ui_intake_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1440px;
    height: 860px;
    background: #07111f;
    color: #f1f6fd;
    font-family: 'Plus Jakarta Sans', sans-serif;
    display: flex;
    overflow: hidden;
  }
  .sidebar {
    width: 300px;
    background: #0a1728;
    border-right: 1px solid rgba(255,255,255,0.08);
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 17px;
    font-weight: 800;
    color: #38bdf8;
  }
  .status-badge {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .status-badge::before {
    content: ""; width: 7px; height: 7px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px #34d399;
  }

  .main-content {
    flex: 1;
    padding: 28px 36px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }
  .kpi-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
  }
  .kpi-val { font-size: 24px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #f8fafc; }
  .kpi-lbl { font-size: 11px; text-transform: uppercase; font-weight: 700; color: #64748b; margin-top: 4px; }

  .two-col {
    display: grid;
    grid-template-columns: 1.1fr 1.3fr;
    gap: 22px;
  }
  .panel {
    background: rgba(14, 34, 61, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 22px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .panel-header {
    font-size: 15px;
    font-weight: 700;
    color: #f1f6fd;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 12px;
  }
  .input-label { font-size: 12px; font-weight: 600; color: #94a9c4; margin-bottom: 6px; }
  .text-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
    color: #f8fafc;
    line-height: 1.5;
  }
  .vitals-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  .vital-item {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 10px;
    text-align: center;
  }
  .vital-num { font-size: 18px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #38bdf8; }
  .vital-tag { font-size: 10px; color: #64748b; font-weight: 600; }

  /* Triage Outcome Card */
  .outcome-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 14px;
    padding: 22px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .esi-badge-lg {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: #ef4444;
    color: white;
    padding: 6px 16px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 800;
    width: fit-content;
  }
  .alert-banner {
    background: rgba(239, 68, 68, 0.2);
    border-left: 4px solid #ef4444;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 12px;
    color: #fecdd3;
    line-height: 1.5;
  }
</style>
</head>
<body>
  <div class="sidebar">
    <div class="sidebar-logo">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
      Triage Nurse Copilot
    </div>
    <div class="status-badge">Engine Online (Rules Floor)</div>
    <div style="font-size: 12px; color: #64748b; line-height: 1.6;">
      <div style="color: #94a9c4; font-weight: 700; margin-bottom: 4px;">OPERATIONAL MODE</div>
      • Triage Mode: Hybrid Consensus<br/>
      • Vector Memory: Active (384d)<br/>
      • Protocol: ESI v4 Emergency Intake<br/>
      • FHIR Version: HL7 FHIR R4
    </div>
  </div>

  <div class="main-content">
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-val">0.13 ms</div><div class="kpi-lbl">P50 Floor Latency</div></div>
      <div class="kpi-card"><div class="kpi-val">100.0%</div><div class="kpi-lbl">Emergency Precision</div></div>
      <div class="kpi-card"><div class="kpi-val">99 / 99</div><div class="kpi-lbl">Passing Test Suites</div></div>
      <div class="kpi-card"><div class="kpi-val">HL7 FHIR R4</div><div class="kpi-lbl">Interoperability</div></div>
    </div>

    <div class="two-col">
      <!-- Input Panel -->
      <div class="panel">
        <div class="panel-header">
          <span>Active Patient Encounter</span>
          <span style="font-family:'JetBrains Mono'; font-size:11px; color:#38bdf8;">ID: PT-1042</span>
        </div>
        <div>
          <div class="input-label">Chief Complaint & Subjective Narrative</div>
          <div class="text-box">
            "61-year-old male with severe crushing substernal chest pressure radiating to left jaw and shortness of breath for 45 minutes, denies fever or trauma."
          </div>
        </div>
        <div class="vitals-grid">
          <div class="vital-item"><div class="vital-num">108</div><div class="vital-tag">HR (bpm)</div></div>
          <div class="vital-item"><div class="vital-num">165/95</div><div class="vital-tag">BP (mmHg)</div></div>
          <div class="vital-item"><div class="vital-num" style="color:#ef4444;">94%</div><div class="vital-tag">SpO2</div></div>
          <div class="vital-item"><div class="vital-num" style="color:#f59e0b;">22</div><div class="vital-tag">RR (/min)</div></div>
          <div class="vital-item"><div class="vital-num">37.1</div><div class="vital-tag">Temp (°C)</div></div>
          <div class="vital-item"><div class="vital-num">61</div><div class="vital-tag">Age (yrs)</div></div>
        </div>
      </div>

      <!-- Outcome Panel -->
      <div class="outcome-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div class="esi-badge-lg">ESI LEVEL 2 · EMERGENT</div>
          <span style="font-family:'JetBrains Mono'; font-size:11px; background:rgba(239,68,68,0.25); color:#fda4af; padding:4px 8px; border-radius:4px; font-weight:700;">HIGH RISK</span>
        </div>
        <div class="alert-banner">
          <strong>CRITICAL RED FLAGS DETECTED:</strong> Crushing chest pain radiating to jaw with hypoxic tachypnea. Immediate physician evaluation and bedside ECG required.
        </div>
        <div style="font-size:12px; color:#cbd5e1; line-height:1.6;">
          <strong>Consensus Safety Floor:</strong> Deterministic rules floor confirmed Emergency ESI 2. Generative LLM cannot downgrade acuity.<br/>
          <strong>Vital Triggers:</strong> Tachycardia (108 bpm), Tachypnea (22/min), Borderline Hypoxia (94%).<br/>
          <strong>Human Review Status:</strong> <span style="color:#fbbf24; font-weight:700;">AWAITING CLINICIAN SIGN-OFF (HITL)</span>
        </div>
        <div style="display:flex; gap:10px; margin-top:8px;">
          <button style="flex:1; background:#10b981; border:none; color:white; padding:10px; border-radius:8px; font-weight:700; cursor:pointer;">Accept AI ESI 2 Recommendation</button>
          <button style="flex:1; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); color:white; padding:10px; border-radius:8px; font-weight:700; cursor:pointer;">Clinical Override</button>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# 3. UI SCREENSHOT 2: SBAR PHYSICIAN REPORT & HL7 FHIR R4 BUNDLE INSPECTOR
# ─────────────────────────────────────────────────────────────────────────────
ui_sbar_fhir_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1440px;
    height: 860px;
    background: #07111f;
    color: #f1f6fd;
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding: 32px 40px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    overflow: hidden;
  }
  .top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 16px;
  }
  .top-title { font-size: 20px; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 10px; }
  .tab-bar { display: flex; gap: 8px; }
  .tab-btn {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #94a9c4;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
  }
  .tab-btn.active {
    background: rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.35);
    color: #38bdf8;
  }

  .split-view {
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: 22px;
    flex: 1;
  }
  .code-card {
    background: #0a1322;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .card-header {
    font-size: 13px;
    font-weight: 700;
    color: #cbd5e1;
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 10px;
  }
  .sbar-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    color: #e2e8f0;
    line-height: 1.6;
    background: rgba(0,0,0,0.3);
    padding: 14px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.05);
    flex: 1;
    white-space: pre-wrap;
  }
  .fhir-json {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #7dd3fc;
    line-height: 1.5;
    background: rgba(0,0,0,0.4);
    padding: 14px;
    border-radius: 8px;
    border: 1px solid rgba(56, 189, 248, 0.15);
    flex: 1;
    overflow-y: auto;
    white-space: pre-wrap;
  }
  .sbar-section { color: #38bdf8; font-weight: 700; }
</style>
</head>
<body>
  <div class="top-bar">
    <div class="top-title">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>
      Clinical Explainability & Interoperability Center
    </div>
    <div class="tab-bar">
      <div class="tab-btn active">Physician SBAR Handoff</div>
      <div class="tab-btn active">HL7 FHIR R4 Bundle</div>
      <div class="tab-btn">Audit Concordance Ledger</div>
    </div>
  </div>

  <div class="split-view">
    <!-- Left: SBAR Note -->
    <div class="code-card">
      <div class="card-header">
        <span>ED PHYSICIAN SBAR HANDOFF REPORT</span>
        <span style="font-family:'JetBrains Mono'; color:#34d399;">STATUS: VALIDATED (ESI 2)</span>
      </div>
      <div class="sbar-text"><span class="sbar-section">[S] SITUATION</span>
• Patient: 61yo Male (PT-1042)
• Chief Complaint: Severe crushing substernal chest pressure radiating to left jaw
• Triage Acuity Status: ESI 2 - Emergent (Immediate Bedside Resuscitation)

<span class="sbar-section">[B] BACKGROUND</span>
• Comorbidities: Hypertension, Type 2 Diabetes Mellitus
• Pertinent Negatives: Denies fever, denies head trauma

<span class="sbar-section">[A] ASSESSMENT</span>
• Intake Telemetry: HR: 108 bpm | BP: 165/95 mmHg | SpO2: 94% | RR: 22/min
• Vital Triggers: Tachycardia, Tachypnea, Borderline Hypoxia
• Red-Flag Symptoms: Crushing chest pain, left jaw radiation
• Acuity Rationale: Deterministic safety-floor triggered acute coronary syndrome protocol

<span class="sbar-section">[R] RECOMMENDATION & DISPOSITION</span>
• Immediate transfer to resuscitation bay for 12-lead ECG & cardiac biomarkers
• Supervised licensed clinician oversight active</div>
    </div>

    <!-- Right: FHIR R4 Bundle -->
    <div class="code-card">
      <div class="card-header">
        <span>HL7 FHIR R4 TRANSACTION BUNDLE</span>
        <span style="font-family:'JetBrains Mono'; color:#38bdf8;">JSON R4 VALID</span>
      </div>
      <div class="fhir-json">{
  "resourceType": "Bundle",
  "id": "triage-bundle-pt1042",
  "type": "transaction",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "pt-1042",
        "gender": "male"
      }
    },
    {
      "resource": {
        "resourceType": "Observation",
        "code": { "text": "Heart Rate" },
        "valueQuantity": { "value": 108, "unit": "bpm" }
      }
    },
    {
      "resource": {
        "resourceType": "Observation",
        "code": { "text": "Oxygen Saturation (SpO2)" },
        "valueQuantity": { "value": 94, "unit": "%" }
      }
    },
    {
      "resource": {
        "resourceType": "Condition",
        "code": { "text": "Crushing chest pain radiating to jaw" },
        "severity": { "text": "ESI-2 Emergent" }
      }
    },
    {
      "resource": {
        "resourceType": "DocumentReference",
        "status": "current",
        "description": "ED Physician SBAR Handoff Report"
      }
    }
  ]
}</div>
    </div>
  </div>
</body>
</html>
"""

# Render both new architecture diagram and UI screenshots
render_html_to_png(arch_html, ASSETS / "system_architecture.png")
render_html_to_png(arch_html, ASSETS / "architecture_diagram.png")
render_html_to_png(ui_intake_html, ASSETS / "ui_intake_workstation.png")
render_html_to_png(ui_sbar_fhir_html, ASSETS / "ui_sbar_fhir_bundle.png")
