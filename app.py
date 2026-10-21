from __future__ import annotations

import json
import re
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app from backend
# -----------------------------
from bwa_backend import app


# -----------------------------
# Helper Functions
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    import zipfile
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    import zipfile
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


def list_past_blogs() -> List[Path]:
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI Configuration
# -----------------------------
st.set_page_config(
    page_title="Local AI Blog Writer",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS styling for professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.title("Autonomous Technical Blog Writer")
st.caption("Powered by Local Ollama (Qwen 2.5) & LangGraph")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    topic = st.text_area(
        "Blog Topic / Prompt",
        height=120,
        placeholder="e.g., Explain Transformer Self-Attention Mechanism in Detail..."
    )
    as_of = st.date_input("As-of Date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.divider()
    st.subheader("📚 Saved Blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md).")
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:30]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title[:35]}... ({p.name})" if len(title) > 35 else f"{title} ({p.name})"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.selectbox("Load past blog", options=options, label_visibility="collapsed")
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load Selected Blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.success(f"Loaded: {selected_md_file.name}")

# Session State Storage
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Main Tabs Layout
tab_plan, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan & Outline", "📝 Markdown Preview", "🖼️ Generated Images", "🧾 Execution Logs"]
)

logs: List[str] = []

def log(msg: str):
    logs.append(msg)


# Execution Trigger
if run_btn:
    if not topic.strip():
        st.warning("⚠️ Please enter a blog topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    status = st.status("🤖 Running agent graph...", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Active Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode": current_state.get("mode"),
                "tasks_planned": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "sections_completed": len(current_state.get("sections", []) or []),
                "images_generated": len(current_state.get("image_specs", []) or []),
            }
            progress_area.json(summary)
            log(f"[{kind}] {json.dumps(payload, default=str)[:800]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            status.update(label="✅ Blog Generated Successfully!", state="complete", expanded=False)
            log("[final] Pipeline completed.")


# Render Results
out = st.session_state.get("last_out")
if out:
    # --- 1. Plan Tab ---
    with tab_plan:
        st.subheader("📋 Generated Outline & Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan object found in output (loaded from markdown file).")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.markdown(f"### **Title:** {plan_dict.get('blog_title')}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Audience", str(plan_dict.get("audience")))
            col2.metric("Tone", str(plan_dict.get("tone")))
            col3.metric("Blog Kind", str(plan_dict.get("blog_kind")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                st.write("#### Tasks Breakdown")
                df = pd.DataFrame(
                    [
                        {
                            "ID": t.get("id"),
                            "Title": t.get("title"),
                            "Target Words": t.get("target_words"),
                            "Goal": t.get("goal"),
                        }
                        for t in tasks
                    ]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

    # --- 2. Preview Tab ---
    with tab_preview:
        st.subheader("📖 Markdown Rendered Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No markdown content available.")
        else:
            render_markdown_with_local_images(final_md)

            st.divider()
            col_d1, col_d2 = st.columns(2)
            
            plan_obj = out.get("plan")
            blog_title = plan_obj.blog_title if hasattr(plan_obj, "blog_title") else "technical_blog"
            md_filename = f"{safe_slug(blog_title)}.md"

            with col_d1:
                st.download_button(
                    "⬇️ Download Markdown File",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                )

            with col_d2:
                bundle = bundle_zip(final_md, md_filename, Path("images"))
                st.download_button(
                    "📦 Download Full Bundle (.zip)",
                    data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                )

    # --- 3. Images Tab ---
    with tab_images:
        st.subheader("🖼️ Generated Illustrations")
        images_dir = Path("images")

        if not images_dir.exists() or not list(images_dir.iterdir()):
            st.info("No local images found in the `images/` directory.")
        else:
            files = [p for p in images_dir.iterdir() if p.is_file()]
            for p in sorted(files):
                st.image(str(p), caption=p.name, use_container_width=True)

            z = images_zip(images_dir)
            if z:
                st.download_button(
                    "⬇️ Download All Images (.zip)",
                    data=z,
                    file_name="blog_images.zip",
                    mime="application/zip",
                )

    # --- 4. Logs Tab ---
    with tab_logs:
        st.subheader("🧾 System Execution Logs")
        if logs:
            st.text_area("Live Log Output", value="\n\n".join(logs[-50:]), height=450)
        else:
            st.info("No logs recorded yet.")
else:
    st.info("Enter a topic in the sidebar and click **Generate Blog** to start.")