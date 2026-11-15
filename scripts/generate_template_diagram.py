import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1380px;
    height: 760px;
    background: #f8fafc;
    color: #0f172a;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    display: flex;
    flex-direction: column;
    padding: 30px 40px;
    position: relative;
    overflow: hidden;
  }

  /* Main Title */
  .title-bar {
    text-align: center;
    margin-bottom: 24px;
    position: relative;
    z-index: 10;
  }
  .title-bar h1 {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
  }

  /* Diagram Area */
  .diagram-container {
    flex: 1;
    display: grid;
    grid-template-columns: 220px 290px 250px 280px;
    justify-content: space-between;
    position: relative;
    z-index: 10;
  }

  /* Columns */
  .cluster-col {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .cluster-heading {
    font-size: 13px;
    font-weight: 700;
    color: #1e293b;
    text-align: center;
    margin-bottom: 16px;
    height: 20px;
  }

  /* Cards */
  .card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 16px 14px;
    position: relative;
    z-index: 2;
  }
  .card-title {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 8px;
    line-height: 1.25;
  }
  .card-desc {
    font-size: 11px;
    font-weight: 500;
    color: #64748b;
    margin-top: 4px;
    line-height: 1.35;
  }

  /* Sizes */
  .w-col1 { width: 190px; }
  .w-col2 { width: 280px; }
  .w-col3 { width: 235px; }
  .w-col4 { width: 260px; }

  /* SVG Canvas for Connectors */
  .svg-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
  }

  /* UI Frame Mockup in Col 4 */
  .ui-mockup-frame {
    background: #090e17;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 10px 12px;
    width: 250px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .mockup-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 4px;
  }
  .mockup-kpis {
    display: flex;
    gap: 6px;
  }
  .mockup-badge {
    background: #ef4444;
    color: white;
    font-size: 10px;
    font-weight: 800;
    padding: 3px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
  }
  .mockup-vitals {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    padding: 4px;
    font-size: 8px;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.4;
  }
</style>
</head>
<body>

  <!-- Top Title -->
  <div class="title-bar">
    <h1>agentic-triage-system</h1>
  </div>

  <!-- SVG Connectors Layer -->
  <svg class="svg-canvas" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="#3b82f6"/>
      </marker>
      <marker id="arrow-coral" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="#f97316"/>
      </marker>
      <marker id="arrow-purple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="#8b5cf6"/>
      </marker>
      <marker id="arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="#10b981"/>
      </marker>
    </defs>

    <!-- 1. Streamlit -> Python Engine (down) -->
    <path d="M 135 242 L 135 410" fill="none" stroke="#f97316" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-coral)"/>
    <text x="145" y="325" font-family="'Plus Jakarta Sans', sans-serif" font-size="11" font-weight="600" fill="#64748b" font-style="italic">webhook / state</text>
    <text x="145" y="340" font-family="'Plus Jakarta Sans', sans-serif" font-size="10.5" font-weight="600" fill="#0f172a">Chief Complaint &amp; Vitals</text>

    <!-- 2. Streamlit -> Regex Query Parser (across) -->
    <path d="M 230 180 L 390 180" fill="none" stroke="#3b82f6" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-blue)"/>

    <!-- 3. Python Engine -> Infermedica & Groq (across) -->
    <path d="M 230 470 L 390 355" fill="none" stroke="#f97316" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-coral)"/>

    <!-- 4. Python Engine -> Sentence-Transformers & Pinecone (across bottom) -->
    <path d="M 230 515 L 390 535" fill="none" stroke="#10b981" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-green)"/>

    <!-- 5. Regex Query Parser -> Vitals Danger-Zone Guard -->
    <path d="M 670 180 L 760 180" fill="none" stroke="#3b82f6" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-blue)"/>

    <!-- 6. Infermedica & Groq -> Consensus Arbitration -->
    <path d="M 670 355 L 760 410" fill="none" stroke="#8b5cf6" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-purple)"/>

    <!-- 7. Pinecone -> Consensus Arbitration (up-right) -->
    <path d="M 670 535 L 715 535 L 715 450 L 760 450" fill="none" stroke="#10b981" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-green)"/>

    <!-- 8. Vitals Guard -> Clinician HITL Gate (across top) -->
    <path d="M 995 180 L 1080 180" fill="none" stroke="#8b5cf6" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-purple)"/>

    <!-- 9. Consensus -> HL7 SBAR & FHIR (across) -->
    <path d="M 995 440 L 1080 355" fill="none" stroke="#8b5cf6" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-purple)"/>

    <!-- 10. Clinician HITL Gate -> HL7 SBAR & FHIR (down) -->
    <path d="M 1210 240 L 1210 285" fill="none" stroke="#8b5cf6" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-purple)"/>

    <!-- 11. HL7 SBAR & FHIR -> Final UI & Visual Results (down) -->
    <path d="M 1210 425 L 1210 500" fill="none" stroke="#10b981" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow-green)"/>
  </svg>

  <!-- Grid of 4 Clusters -->
  <div class="diagram-container">

    <!-- CLUSTER 1: INGESTION -->
    <div class="cluster-col">
      <div class="cluster-heading">Ingestion Cluster</div>

      <!-- Streamlit Web App -->
      <div class="card w-col1" style="height: 140px; margin-top: 10px;">
        <!-- Streamlit Red Crown SVG -->
        <svg width="40" height="28" viewBox="0 0 100 68" fill="none">
          <path d="M49.999 0L75.319 44.029L99.882 17.587L83.003 67.065H16.995L0.117 17.587L24.68 44.029L49.999 0Z" fill="#FF4B4B"/>
        </svg>
        <div class="card-title">Streamlit<br/>Web App</div>
        <div class="card-desc">Nurse Triage Workstation<br/>Free-Text Intake &amp; Vitals</div>
      </div>

      <div style="flex: 1;"></div>

      <!-- Python 3.12 Engine -->
      <div class="card w-col1" style="height: 140px; margin-bottom: 30px;">
        <!-- Python Logo SVG -->
        <svg width="36" height="36" viewBox="0 0 110 110" fill="none">
          <path d="M53.4 2C27.9 2 29.5 13 29.5 13L29.6 24.3H54V27.8H20.1C20.1 27.8 2 25.8 2 51.5C2 77.2 17.8 75.3 17.8 75.3H27.3V62.4C27.3 62.4 26.8 46.8 42.6 46.8H56.5C56.5 46.8 69.8 47.4 69.8 34.2V13C69.8 13 72.8 2 53.4 2ZM39.6 9.8C42.7 9.8 45.2 12.3 45.2 15.4C45.2 18.5 42.7 21 39.6 21C36.5 21 34 18.5 34 15.4C34 12.3 36.5 9.8 39.6 9.8Z" fill="#387eb8"/>
          <path d="M56.6 108C82.1 108 80.5 97 80.5 97L80.4 85.7H56V82.2H89.9C89.9 82.2 108 84.2 108 58.5C108 32.8 92.2 34.7 92.2 34.7H82.7V47.6C82.7 47.6 83.2 63.2 67.4 63.2H53.5C53.5 63.2 40.2 62.6 40.2 75.8V97C40.2 97 37.2 108 56.6 108ZM70.4 100.2C67.3 100.2 64.8 97.7 64.8 94.6C64.8 91.5 67.3 89 70.4 89C73.5 89 76 91.5 76 94.6C76 97.7 73.5 100.2 70.4 100.2Z" fill="#ffe052"/>
        </svg>
        <div class="card-title">Python 3.12<br/>Engine</div>
        <div class="card-desc">LangGraph Orchestrator</div>
      </div>
    </div>

    <!-- CLUSTER 2: CLINICAL NLP & INTENT -->
    <div class="cluster-col">
      <div class="cluster-heading">Clinical NLP &amp; Intent Cluster</div>

      <!-- Regex Query Parser -->
      <div class="card w-col2" style="height: 100px; margin-top: 25px;">
        <div class="card-title">Regex Query Parser</div>
        <div class="card-desc">Deterministic keyword<br/>&amp; vitals fallback</div>
      </div>

      <div style="height: 38px;"></div>

      <!-- Infermedica & Groq Cloud -->
      <div class="card w-col2" style="height: 110px;">
        <div style="display:flex; align-items:center; gap: 12px; margin-bottom: 2px;">
          <!-- Infermedica Plus Icon -->
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <rect width="24" height="24" rx="12" fill="#2563eb"/>
            <path d="M12 6V18M6 12H18" stroke="white" stroke-width="3" stroke-linecap="round"/>
          </svg>
          <span style="font-size: 14.5px; font-weight: 800; color: #1e3a8a;">Infermedica v3</span>
          <span style="font-size: 13px; font-weight: 800; color: #64748b;">&amp; Groq Cloud</span>
          <!-- Groq badge -->
          <span style="font-size: 11px; font-family:'JetBrains Mono'; font-weight: 800; background:#f1f5f9; padding: 2px 5px; border-radius: 4px; color:#f97316;">groq</span>
        </div>
        <div class="card-desc">Clinical Concept Mapping &amp;<br/>LLaMA-3.3-70B</div>
      </div>

      <div style="height: 38px;"></div>

      <!-- Sentence Transformers & Pinecone -->
      <div class="card w-col2" style="height: 125px;">
        <div style="font-size: 13px; font-weight: 800; color: #1e293b; margin-bottom: 4px;">Sentence-Transformers<br/>&amp; Pinecone</div>
        <div style="display:flex; align-items:center; gap: 6px; margin: 2px 0;">
          <!-- Pinecone Icon -->
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2.2">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
          <span style="font-size: 14px; font-weight: 800; color: #0f172a;">Pinecone</span>
        </div>
        <div class="card-desc">all-MiniLM-L6-v2<br/>Dense Patient Memory</div>
      </div>
    </div>

    <!-- CLUSTER 3: RISK & CONSENSUS -->
    <div class="cluster-col">
      <div class="cluster-heading">Risk &amp; Consensus Cluster</div>

      <!-- Vitals Danger-Zone Guard -->
      <div class="card w-col3" style="height: 130px; margin-top: 15px;">
        <div style="display:flex; align-items:center; gap: 8px;">
          <!-- Shield Cross Icon -->
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L3 6v6c0 5.5 3.8 10.7 9 12 5.2-1.3 9-6.5 9-12V6l-9-4z" fill="#dc2626"/>
            <path d="M12 8v8M8 12h8" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
          </svg>
          <span style="font-size: 14.5px; font-weight: 800; color: #0f172a;">Vitals Danger-<br/>Zone Guard</span>
        </div>
        <div class="card-desc" style="margin-top: 8px;">Physiological Triggers<br/>(SpO2 &lt; 90% -&gt; ESI 1)</div>
      </div>

      <div style="height: 55px;"></div>

      <!-- Consensus Arbitration -->
      <div class="card w-col3" style="height: 140px;">
        <div style="display:flex; align-items:center; gap: 8px;">
          <!-- Infinity Consensus Icon -->
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
            <path d="M18.178 8c5.096 0 5.096 8 0 8-2.612 0-4.04-2.072-6.178-4-2.138-1.928-3.566-4-6.178-4-5.096 0-5.096 8 0 8 2.612 0 4.04-2.072 6.178-4 2.138-1.928 3.566-4 6.178-4z" stroke="url(#consensus-grad)" stroke-width="2.5" stroke-linecap="round"/>
            <defs>
              <linearGradient id="consensus-grad" x1="2" y1="8" x2="22" y2="16" gradientUnits="userSpaceOnUse">
                <stop stop-color="#3b82f6"/>
                <stop offset="0.5" stop-color="#f59e0b"/>
                <stop offset="1" stop-color="#10b981"/>
              </linearGradient>
            </defs>
          </svg>
          <span style="font-size: 14.5px; font-weight: 800; color: #0f172a;">Consensus<br/>Arbitration</span>
        </div>
        <div class="card-desc" style="margin-top: 8px;">'Escalate, Never<br/>Downgrade'<br/>Acuity Protocol</div>
      </div>
    </div>

    <!-- CLUSTER 4: DECISION & GOVERNANCE -->
    <div class="cluster-col">
      <div class="cluster-heading">Decision &amp; Governance Cluster</div>

      <!-- Clinician HITL Gate -->
      <div class="card w-col4" style="height: 110px; margin-top: 15px;">
        <div style="display:flex; align-items:center; gap: 8px;">
          <!-- Nurse Stethoscope Icon -->
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
          </svg>
          <span style="font-size: 14.5px; font-weight: 800; color: #0f172a;">Clinician<br/>HITL Gate</span>
        </div>
        <div class="card-desc" style="margin-top: 6px;">Mandatory Nurse Sign-off<br/>(ESI 1-2)</div>
      </div>

      <div style="height: 25px;"></div>

      <!-- HL7 Explainable FHIR SBAR & FHIR -->
      <div class="card w-col4" style="height: 120px;">
        <div style="display:flex; align-items:center; gap: 6px;">
          <span style="font-size: 15px; font-weight: 900; color: #dc2626;">HL7</span>
          <span style="font-size: 14px; font-weight: 800; color: #0f172a;">Explainable<br/>FHIR SBAR &amp; FHIR</span>
        </div>
        <div class="card-desc" style="margin-top: 6px;">Physician SBAR Note &amp;<br/>HL7 FHIR R4 Bundle</div>
      </div>

      <div style="height: 25px;"></div>

      <!-- Final UI & Visual Results -->
      <div class="cluster-heading" style="font-size: 12px; margin-bottom: 6px;">Final UI &amp; Visual Results</div>
      <div class="ui-mockup-frame">
        <div class="mockup-top">
          <div style="display:flex; gap:4px;">
            <div style="width:6px; height:6px; border-radius:50%; background:#ef4444;"></div>
            <div style="width:6px; height:6px; border-radius:50%; background:#f59e0b;"></div>
            <div style="width:6px; height:6px; border-radius:50%; background:#10b981;"></div>
          </div>
          <span style="font-size:8px; font-weight:700; color:#64748b; font-family:'JetBrains Mono';">PT-1042 · TRIAGE LIVE</span>
        </div>
        <div class="mockup-kpis">
          <div class="mockup-badge">ESI 2</div>
          <div class="mockup-vitals" style="flex:1;">
            HR: 108 | SpO2: 94%<br/>
            BP: 165/95 | RR: 22
          </div>
        </div>
        <div style="font-size:8px; color:#cbd5e1; font-family:'JetBrains Mono'; line-height:1.3; background:rgba(255,255,255,0.03); padding:4px; border-radius:4px;">
          • Red Flag: Crushing chest pain<br/>
          • Protocol: Immediate Bedside ECG
        </div>
      </div>
    </div>

  </div>

</body>
</html>
"""

HTML_PATH = ASSETS / "template_diagram.html"
HTML_PATH.write_text(html_content, encoding="utf-8")

def render():
    for name in ["system_architecture.png", "architecture_diagram.png"]:
        out = ASSETS / name
        cmd = [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--window-size=1380,760",
            "--force-device-scale-factor=2",
            f"--screenshot={out}",
            str(HTML_PATH)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and out.exists():
            print(f"[OK] Rendered {name}: {out.stat().st_size} bytes")
        else:
            print(f"[ERR] Failed {name}: {res.stderr}")

if __name__ == "__main__":
    render()
