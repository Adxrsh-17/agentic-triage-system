"""Generate a Graphviz architecture diagram for the healthcare-Agent project.

The script analyzes the workspace, records the source files it finds, and
builds a layered architecture diagram that shows the Streamlit UI, the Groq
ReAct path, the local multi-agent graph, and the external services used by the
project.

It always writes a DOT file. If the Graphviz ``dot`` executable is installed,
it also renders an SVG beside it.
"""

from __future__ import annotations

import argparse
import ast
import html
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
}


@dataclass(frozen=True)
class ProjectAnalysis:
    python_files: List[Path]
    support_files: List[Path]
    local_modules: Set[str]
    import_edges: Dict[str, Set[str]]


def collect_files(root: Path) -> tuple[list[Path], list[Path]]:
    python_files: list[Path] = []
    support_files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix == ".py":
            python_files.append(path)
        elif path.name in {
            "README.md",
            "architecture_diagram.md",
            "requirements.txt",
            ".env",
            ".env.example",
            "Healthcare Agent.docx",
        }:
            support_files.append(path)
    return sorted(python_files), sorted(support_files)


def module_name(root: Path, path: Path) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else path.stem


def parse_local_imports(path: Path, module: str, local_modules: Set[str]) -> Set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name
                if any(imported == local or imported.startswith(f"{local}.") for local in local_modules):
                    edges.add(imported)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                parent_parts = module.split(".")[: max(len(module.split(".")) - node.level, 0)]
                base = ".".join(parent_parts)
                imported = f"{base}.{node.module}" if node.module else base
            else:
                imported = node.module or ""

            if imported and any(imported == local or imported.startswith(f"{local}.") for local in local_modules):
                edges.add(imported)

    return edges


def analyze_project(root: Path) -> ProjectAnalysis:
    python_files, support_files = collect_files(root)
    local_modules = {module_name(root, path) for path in python_files}
    import_edges: dict[str, set[str]] = {}

    for path in python_files:
        module = module_name(root, path)
        import_edges[module] = parse_local_imports(path, module, local_modules)

    return ProjectAnalysis(
        python_files=python_files,
        support_files=support_files,
        local_modules=local_modules,
        import_edges=import_edges,
    )


def quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def node_id(prefix: str, name: str) -> str:
    cleaned = [ch if ch.isalnum() else "_" for ch in name]
    return f"{prefix}_{''.join(cleaned)}"


def file_inventory_text(analysis: ProjectAnalysis) -> str:
    python_names = ", ".join(path.relative_to(ROOT).as_posix() for path in analysis.python_files)
    support_names = ", ".join(path.relative_to(ROOT).as_posix() for path in analysis.support_files)
    return f"Python files: {python_names}\nSupport files: {support_names}"


def build_dot(analysis: ProjectAnalysis) -> str:
    lines: list[str] = [
        "digraph HealthcareAgentArchitecture {",
        "    graph [rankdir=LR, splines=ortho, nodesep=0.55, ranksep=0.9, pad=0.3, bgcolor=white];",
        "    node [shape=box, style=\"rounded,filled\", fontname=Helvetica, fontsize=10, margin=0.16];",
        "    edge [fontname=Helvetica, fontsize=9, color=\"#64748b\", arrowsize=0.8];",
        "",
    ]

    inventory = node_id("note", "inventory")
    lines.extend(
        [
            "    subgraph cluster_inventory {",
            "        label=\"Project Scope\";",
            "        color=\"#cbd5e1\";",
            "        style=\"rounded,dashed\";",
            f"        {inventory} [label={quote(file_inventory_text(analysis))}, shape=note, fillcolor=\"#f8fafc\", color=\"#cbd5e1\"];",
            "    }",
            "",
        ]
    )

    # Core file nodes.
    app_file = node_id("file", "app.py")
    infer_file = node_id("file", "agent_infermedica_client_py")
    loc_file = node_id("file", "agent_location_tools_py")
    multi_file = node_id("file", "agent_multi_agent_py")
    init_file = node_id("file", "agent_init_py")
    site_file = node_id("file", "sitecustomize_py")
    env_file = node_id("file", "env")
    req_file = node_id("file", "requirements_txt")
    readme_file = node_id("file", "readme_md")

    lines.extend(
        [
            "    subgraph cluster_files {",
            "        label=\"Source Files\";",
            "        color=\"#e2e8f0\";",
            "        style=\"rounded\";",
            f"        {app_file} [label=\"app.py\\nStreamlit UI + Dashboard\", fillcolor=\"#dbeafe\", color=\"#3b82f6\"];",
            f"        {init_file} [label=\"agent/__init__.py\\npackage export\", fillcolor=\"#f1f5f9\", color=\"#64748b\"];",
            f"        {multi_file} [label=\"agent/multi_agent.py\\nStateGraph orchestration\", fillcolor=\"#ecfccb\", color=\"#84cc16\"];",
            f"        {infer_file} [label=\"agent/infermedica_client.py\\nInfermedica v3 + OTC Meds\", fillcolor=\"#e0f2fe\", color=\"#0284c7\"];",
            f"        {loc_file} [label=\"agent/location_tools.py\\nGeocoding & Maps routing\", fillcolor=\"#fef3c7\", color=\"#f59e0b\"];",
            f"        {site_file} [label=\"sitecustomize.py\\nPython path bootstrap\", fillcolor=\"#f8fafc\", color=\"#94a3b8\"];",
            f"        {env_file} [label=\".env / .env.example\\nruntime configuration\", shape=note, fillcolor=\"#fff7ed\", color=\"#f59e0b\"];",
            f"        {req_file} [label=\"requirements.txt\\ndependencies\", shape=note, fillcolor=\"#f8fafc\", color=\"#94a3b8\"];",
            f"        {readme_file} [label=\"README.md / architecture_diagram.md\\ndocumentation\", shape=note, fillcolor=\"#f8fafc\", color=\"#94a3b8\"];",
            "    }",
            "",
        ]
    )

    user = node_id("actor", "user")
    streamlit = node_id("component", "streamlit_ui")
    infermedica_engine = node_id("component", "infermedica_engine")
    local_graph = node_id("component", "local_graph")
    groq_api = node_id("service", "groq_api")
    pinecone = node_id("service", "pinecone")
    location_api = node_id("service", "location_service")
    local_store = node_id("service", "local_patient_store")
    review_ui = node_id("component", "review_ui")

    lines.extend(
        [
            "    subgraph cluster_runtime {",
            "        label=\"Runtime Architecture\";",
            "        color=\"#cbd5e1\";",
            "        style=\"rounded\";",
            f"        {user} [label=\"Intake Staff\\nPatient complaint\", shape=ellipse, fillcolor=\"#e0f2fe\", color=\"#0284c7\"];",
            f"        {streamlit} [label=\"Streamlit Clinical App\\napp.py\", fillcolor=\"#dbeafe\", color=\"#3b82f6\"];",
            f"        {infermedica_engine} [label=\"Infermedica Clinical v3\\nagent/infermedica_client.py\", fillcolor=\"#e0f2fe\", color=\"#0284c7\"];",
            f"        {local_graph} [label=\"LangGraph StateGraph\\nagent/multi_agent.py\", fillcolor=\"#ecfccb\", color=\"#84cc16\"];",
            f"        {review_ui} [label=\"Human Review Node (HITL)\\napprove / reject emergent cases\", fillcolor=\"#fff7ed\", color=\"#f59e0b\"];",
            f"        {groq_api} [label=\"Groq API\\nTool-Calling Fallback\", shape=component, fillcolor=\"#fce7f3\", color=\"#ec4899\"];",
            f"        {pinecone} [label=\"Pinecone 384-dim\\nall-MiniLM-L6-v2 embeddings\", shape=component, fillcolor=\"#dcfce7\", color=\"#22c55e\"];",
            f"        {location_api} [label=\"Maps & OpenStreetMap\\nagent/location_tools.py\", shape=component, fillcolor=\"#fef3c7\", color=\"#f59e0b\"];",
            f"        {local_store} [label=\"Local Patient Store\\nfallback memory\", shape=component, fillcolor=\"#f8fafc\", color=\"#94a3b8\"];",
            "    }",
            "",
        ]
    )

    supervisor = node_id("step", "supervisor")
    intake = node_id("step", "intake")
    risk = node_id("step", "risk")
    safety = node_id("step", "safety")
    human_review = node_id("step", "human_review")
    finish = node_id("step", "finish")

    extract_symptoms = node_id("tool", "extract_symptoms")
    retrieve_memory = node_id("tool", "retrieve_patient_memory")
    followups = node_id("tool", "generate_followup_questions")
    assess_risk = node_id("tool", "assess_medical_risk")
    red_flags = node_id("tool", "check_emergency_red_flags")
    safety_check = node_id("tool", "perform_safety_check")
    finalize = node_id("tool", "finalize_response")

    lines.extend(
        [
            "    subgraph cluster_multi_agent {",
            "        label=\"Local multi-agent flow\";",
            "        color=\"#bbf7d0\";",
            "        style=\"rounded\";",
            f"        {supervisor} [label=\"Supervisor\", fillcolor=\"#f0fdf4\", color=\"#22c55e\"];",
            f"        {intake} [label=\"Intake\", fillcolor=\"#f0fdf4\", color=\"#22c55e\"];",
            f"        {risk} [label=\"Risk\", fillcolor=\"#f0fdf4\", color=\"#22c55e\"];",
            f"        {safety} [label=\"Safety\", fillcolor=\"#f0fdf4\", color=\"#22c55e\"];",
            f"        {human_review} [label=\"Human review\", fillcolor=\"#fef3c7\", color=\"#f59e0b\"];",
            f"        {finish} [label=\"Finish\", fillcolor=\"#f0fdf4\", color=\"#22c55e\"];",
            "    }",
            "",
            "    subgraph cluster_tools {",
            "        label=\"Tools used by the agents\";",
            "        color=\"#fbcfe8\";",
            "        style=\"rounded\";",
            f"        {extract_symptoms} [label=\"extract_symptoms\", fillcolor=\"#fdf2f8\", color=\"#ec4899\"];",
            f"        {retrieve_memory} [label=\"retrieve_patient_memory\", fillcolor=\"#fdf2f8\", color=\"#ec4899\"];",
            f"        {followups} [label=\"generate_followup_questions\", fillcolor=\"#fdf2f8\", color=\"#ec4899\"];",
            f"        {assess_risk} [label=\"assess_medical_risk\", fillcolor=\"#fdf2f8\", color=\"#ec4899\"];",
            f"        {red_flags} [label=\"check_emergency_red_flags\", fillcolor=\"#fdf2f8\", color=\"#ec4899\"];",
            f"        {safety_check} [label=\"perform_safety_check\", fillcolor=\"#fdf2f8\", color=\"#ec4899\"];",
            f"        {finalize} [label=\"finalize_response\", fillcolor=\"#fdf2f8\", color=\"#ec4899\"];",
            "    }",
            "",
        ]
    )

    # Runtime file wiring.
    lines.extend(
        [
            f"    {user} -> {streamlit} [label=\"describe symptoms\", color=\"#0284c7\"];",
            f"    {streamlit} -> {backend_router} [label=\"build session state\", color=\"#2563eb\"];",
            f"    {backend_router} -> {groq_agent} [label=\"GROQ_API_KEY present\", color=\"#ec4899\"];",
            f"    {backend_router} -> {local_graph} [label=\"fallback / review flow\", color=\"#84cc16\"];",
            f"    {streamlit} -> {review_ui} [label=\"high-risk approval\", color=\"#f59e0b\"];",
            f"    {groq_agent} -> {groq_api} [label=\"ChatGroq\", color=\"#ec4899\"];",
            f"    {groq_agent} -> {extract_symptoms} [style=dashed, label=\"tool calls\", color=\"#ec4899\"];",
            f"    {groq_agent} -> {retrieve_memory} [style=dashed, label=\"tool calls\", color=\"#ec4899\"];",
            f"    {groq_agent} -> {followups} [style=dashed, label=\"tool calls\", color=\"#ec4899\"];",
            f"    {groq_agent} -> {assess_risk} [style=dashed, label=\"tool calls\", color=\"#ec4899\"];",
            f"    {groq_agent} -> {red_flags} [style=dashed, label=\"tool calls\", color=\"#ec4899\"];",
            f"    {groq_agent} -> {safety_check} [style=dashed, label=\"tool calls\", color=\"#ec4899\"];",
            f"    {groq_agent} -> {finalize} [style=dashed, label=\"tool calls\", color=\"#ec4899\"];",
            f"    {local_graph} -> {supervisor} [label=\"StateGraph\", color=\"#84cc16\"];",
            f"    {supervisor} -> {intake} [label=\"route\", color=\"#22c55e\"];",
            f"    {intake} -> {risk} [label=\"route\", color=\"#22c55e\"];",
            f"    {risk} -> {safety} [label=\"route\", color=\"#22c55e\"];",
            f"    {safety} -> {human_review} [label=\"if HIGH risk\", color=\"#f59e0b\"];",
            f"    {safety} -> {finish} [label=\"safe completion\", color=\"#22c55e\"];",
            f"    {human_review} -> {review_ui} [label=\"approve / reject\", color=\"#f59e0b\"];",
            f"    {intake} -> {extract_symptoms} [label=\"extract\", style=dashed, color=\"#ec4899\"];",
            f"    {intake} -> {retrieve_memory} [label=\"patient context\", style=dashed, color=\"#ec4899\"];",
            f"    {intake} -> {followups} [label=\"clarify\", style=dashed, color=\"#ec4899\"];",
            f"    {risk} -> {assess_risk} [label=\"score risk\", style=dashed, color=\"#ec4899\"];",
            f"    {risk} -> {red_flags} [label=\"emergency checks\", style=dashed, color=\"#ec4899\"];",
            f"    {safety} -> {finalize} [label=\"draft response\", style=dashed, color=\"#ec4899\"];",
            f"    {safety} -> {safety_check} [label=\"safety validation\", style=dashed, color=\"#ec4899\"];",
            f"    {retrieve_memory} -> {pinecone} [label=\"optional\", color=\"#22c55e\"];",
            f"    {retrieve_memory} -> {local_store} [label=\"fallback\", color=\"#94a3b8\"];",
            f"    {env_file} -> {app_file} [style=dotted, label=\"config\", color=\"#f59e0b\"];",
            f"    {env_file} -> {infer_file} [style=dotted, label=\"api key\", color=\"#f59e0b\"];",
            f"    {env_file} -> {multi_file} [style=dotted, label=\"memory config\", color=\"#f59e0b\"];",
            f"    {req_file} -> {app_file} [style=dotted, label=\"dependencies\", color=\"#94a3b8\"];",
            f"    {readme_file} -> {inventory} [style=dotted, label=\"docs\", color=\"#94a3b8\"];",
            f"    {site_file} -> {app_file} [style=dotted, label=\"site path\", color=\"#94a3b8\"];",
        ]
    )

    # Explicit source import edges between local Python files.
    app_module = module_name(ROOT, ROOT / "app.py")
    for source_module, imported_modules in analysis.import_edges.items():
        for imported_module in imported_modules:
            if source_module == app_module and imported_module in {"agent.multi_agent", "agent.infermedica_client", "agent.location_tools"}:
                target_file = multi_file if "multi_agent" in imported_module else (infer_file if "infermedica" in imported_module else loc_file)
                lines.append(f"    {app_file} -> {target_file} [style=bold, color=\"#1d4ed8\", label=\"imports\"];")
            elif source_module == "agent" and imported_module == "agent.multi_agent":
                lines.append(f"    {init_file} -> {multi_file} [style=bold, color=\"#64748b\", label=\"exports\"];")

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_svg(dot_path: Path, svg_path: Path) -> bool:
    dot_executable = shutil.which("dot")
    if not dot_executable:
        return False
    subprocess.run([dot_executable, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    return True


def render_svg_fallback(svg_path: Path, analysis: ProjectAnalysis) -> None:
    width = 2400
    height = 1700

    def text_lines(label: str) -> list[str]:
        return label.split("\n")

    def escape(label: str) -> str:
        return html.escape(label)

    boxes = [
        (120, 120, 220, 90, "User\nSymptoms input", "#e0f2fe", "#0284c7"),
        (420, 120, 280, 100, "Streamlit app\napp.py", "#dbeafe", "#3b82f6"),
        (780, 80, 340, 110, "Backend router\nselects Groq or local graph", "#eff6ff", "#2563eb"),
        (1200, 50, 330, 110, "Groq ReAct backend\nagent/react_agent.py", "#fce7f3", "#ec4899"),
        (1200, 190, 330, 110, "Deterministic triage graph\nagent/multi_agent.py", "#ecfccb", "#84cc16"),
        (1600, 30, 260, 90, "Groq API\nChatGroq / LLM", "#fce7f3", "#ec4899"),
        (1600, 150, 260, 90, "Pinecone\npatient memory index", "#dcfce7", "#22c55e"),
        (1600, 270, 260, 90, "Local patient store\nfallback memory", "#f8fafc", "#94a3b8"),
        (780, 360, 240, 90, "Human review UI\napprove / reject", "#fff7ed", "#f59e0b"),
        (420, 450, 220, 80, "Supervisor", "#f0fdf4", "#22c55e"),
        (700, 450, 220, 80, "Intake", "#f0fdf4", "#22c55e"),
        (980, 450, 220, 80, "Risk", "#f0fdf4", "#22c55e"),
        (1260, 450, 220, 80, "Safety", "#f0fdf4", "#22c55e"),
        (1540, 450, 220, 80, "Human review", "#fef3c7", "#f59e0b"),
        (1820, 450, 220, 80, "Finish", "#f0fdf4", "#22c55e"),
    ]

    tool_boxes = [
        (220, 620, 220, 70, "extract_symptoms"),
        (460, 620, 220, 70, "retrieve_patient_memory"),
        (700, 620, 220, 70, "generate_followup_questions"),
        (940, 620, 220, 70, "assess_medical_risk"),
        (1180, 620, 220, 70, "check_emergency_red_flags"),
        (1420, 620, 220, 70, "perform_safety_check"),
        (1660, 620, 220, 70, "finalize_response"),
    ]

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="60" y="50" font-family="Helvetica, Arial, sans-serif" font-size="28" font-weight="700" fill="#0f172a">Smart Triage AI Architecture</text>',
        '<text x="60" y="82" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#475569">Generated from the current workspace analysis</text>',
    ]

    def draw_box(x: int, y: int, w: int, h: int, label: str, fill: str, stroke: str, radius: int = 16) -> None:
        svg_parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" ry="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        lines = text_lines(label)
        line_y = y + 30
        for idx, line in enumerate(lines):
            svg_parts.append(
                f'<text x="{x + w / 2}" y="{line_y + idx * 18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" font-weight="600" fill="#0f172a">{escape(line)}</text>'
            )

    def arrow(x1: int, y1: int, x2: int, y2: int, label: str | None = None, dashed: bool = False) -> None:
        dash = ' stroke-dasharray="7 6"' if dashed else ''
        svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"{dash}/>' )
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 - 6
            svg_parts.append(f'<text x="{mx}" y="{my}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#475569">{escape(label)}</text>')

    for box in boxes:
        draw_box(*box)

    for box in tool_boxes:
        draw_box(box[0], box[1], box[2], box[3], box[4], "#fdf2f8", "#ec4899", radius=12)

    # Main flow arrows.
    arrow(340, 165, 420, 165, "input")
    arrow(700, 170, 780, 170, "session state")
    arrow(1120, 160, 1200, 105, "GROQ_API_KEY", False)
    arrow(1120, 205, 1200, 245, "fallback / review")
    arrow(1530, 105, 1600, 75, "ChatGroq")
    arrow(1530, 245, 1600, 195, "memory")
    arrow(1530, 245, 1600, 315, "fallback")
    arrow(900, 170, 900, 450, "route")
    arrow(640, 490, 700, 490, "route")
    arrow(920, 490, 980, 490, "route")
    arrow(1200, 490, 1260, 490, "route")
    arrow(1480, 490, 1540, 490, "high risk")
    arrow(1760, 490, 1820, 490, "complete")
    arrow(540, 450, 220, 655, "extract", True)
    arrow(760, 450, 460, 655, "context", True)
    arrow(980, 450, 700, 655, "follow up", True)
    arrow(1090, 490, 940, 655, "score", True)
    arrow(1340, 490, 1180, 655, "red flags", True)
    arrow(1320, 490, 1420, 655, "safety", True)
    arrow(1650, 490, 1660, 655, "finalize", True)

    # Footer note.
    svg_parts.append(f'<text x="60" y="{height - 50}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#64748b">Python files analyzed: {len(analysis.python_files)} | Support files analyzed: {len(analysis.support_files)}</text>')
    svg_parts.append('</svg>')
    svg_path.write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the healthcare-Agent architecture diagram.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT),
        help="Directory where the DOT and SVG files will be written.",
    )
    parser.add_argument(
        "--base-name",
        default="architecture_diagram",
        help="Base file name for the generated diagram files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze_project(ROOT)
    dot_source = build_dot(analysis)

    dot_path = output_dir / f"{args.base_name}.dot"
    svg_path = output_dir / f"{args.base_name}.svg"

    dot_path.write_text(dot_source, encoding="utf-8")

    print(f"Analyzed {len(analysis.python_files)} Python files and {len(analysis.support_files)} support files.")
    print(f"Wrote DOT source to: {dot_path}")

    if render_svg(dot_path, svg_path):
        print(f"Rendered SVG to: {svg_path}")
    else:
        render_svg_fallback(svg_path, analysis)
        print(f"Graphviz dot was not found, so wrote a fallback SVG to: {svg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())