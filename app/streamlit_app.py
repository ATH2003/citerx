"""CiteRx web app.   Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Streamlit Cloud: copy secrets into the environment so Settings can read them.
try:
    for key, value in st.secrets.items():
        if isinstance(value, str) and key.startswith("CITERX_"):
            os.environ.setdefault(key, value)
except Exception:
    pass

from citerx.pipeline import CiteRx  # noqa: E402

st.set_page_config(page_title="CiteRx", page_icon="⚕️", layout="wide")

BADGE = {
    "supported": ("✅", "Supported by the cited source"),
    "weak": ("🟡", "Partly supported: read the source"),
    "unsupported": ("🔴", "Not supported by the cited source"),
    "uncited": ("⚪", "No citation given"),
}
EXAMPLES = [
    "What are the renal contraindications for metformin?",
    "What medications was the patient discharged on?",
    "What is the starting dose of ramipril?",
    "How should hypoglycaemia be treated in a conscious patient?",
]


@st.cache_resource
def engine() -> CiteRx:
    rx = CiteRx()
    if not rx.index.chunks:  # first run: load the bundled samples so the app is usable
        for f in sorted((ROOT / "samples").iterdir()):
            rx.ingest(f)
    return rx


rx = engine()

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("⚕️ CiteRx")
    st.caption("Every answer cited. Every claim checked.")

    uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads and st.button("Index documents", type="primary", use_container_width=True):
        for up in uploads:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / up.name
                path.write_bytes(up.getvalue())
                try:
                    chunks = rx.ingest(path)
                    st.success(f"{up.name}: {len(chunks)} chunks")
                except ValueError as e:
                    st.error(f"{up.name}: {e}")

    st.subheader("Indexed documents")
    for source, chunks in rx.index.documents.items():
        col1, col2 = st.columns([5, 1])
        col1.write(f"📄 {source}  \n<small>{len(chunks)} chunks · {sum(c.has_warning for c in chunks)} warnings</small>", unsafe_allow_html=True)
        if col2.button("✕", key=f"rm-{source}", help=f"Remove {source}"):
            rx.remove(source)
            st.rerun()

    with st.expander("Engine"):
        st.json(rx.describe())

# ---------------- main ----------------
ask_tab, browse_tab, about_tab = st.tabs(["Ask", "Browse", "How it works"])

with ask_tab:
    st.markdown("#### Ask a question about your documents")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    question = st.text_input("Question", key="q", placeholder="e.g. When should metformin be stopped?")

    if question:
        with st.spinner("Retrieving, answering and checking claims…"):
            ans = rx.ask(question)
        if rx.last_error:
            st.warning(rx.last_error)

        if ans.abstained:
            st.info(ans.text)
        else:
            st.markdown(ans.text)
            flagged = len(ans.flagged)
            m1, m2, m3 = st.columns(3)
            m1.metric("Groundedness", f"{ans.groundedness:.0%}")
            m2.metric("Claims checked", len(ans.claims))
            m3.metric("Flagged", flagged, delta=None if not flagged else "review", delta_color="inverse")

            st.markdown("##### Claim check")
            for c in ans.claims:
                icon, label = BADGE[c.status]
                detail = f" · numbers not in source: **{', '.join(c.missing_numbers)}**" if c.missing_numbers else ""
                cites = ", ".join(f"[{i}]" for i in c.cited) or "none"
                st.markdown(f"{icon} {c.sentence}  \n<small>{label} · cites {cites} · support {c.support:.0%}{detail}</small>", unsafe_allow_html=True)

        st.markdown("##### Sources")
        for n, s in enumerate(ans.sources, 1):
            c = s.chunk
            tag = " · 🛡️ added for safety" if s.signals.get("added_for_safety") else ""
            with st.expander(f"[{n}] {c.source} · p.{c.page} · {c.section or '—'}{tag}"):
                st.text(c.text)
                st.caption(f"signals: {s.signals}")
        st.caption(f"Mode: {ans.mode}. CiteRx is a research demo and not a medical device.")

with browse_tab:
    docs = rx.index.documents
    if docs:
        choice = st.selectbox("Document", list(docs))
        for c in docs[choice]:
            st.markdown(f"**{c.section or 'Untitled section'}** · p.{c.page}" + (" · ⚠️ warning" if c.has_warning else ""))
            st.text(c.text)
    else:
        st.info("No documents indexed yet.")

with about_tab:
    st.markdown(
        """
**Pipeline:** structure-aware chunking → hybrid search (dense + BM25) → reciprocal rank fusion →
reranking → safety expansion → answer with numbered citations → claim-level verification.

**Claim verification** checks each sentence against the source it cites. A claim is flagged
when its words aren't found in that source, or when it contains a number (a dose, a threshold,
a duration) that the source does not contain.

**Safety expansion** adds WARNING/CAUTION passages from the same section as the retrieved
content, so a dose is never shown without its warning.

Without an LLM configured, CiteRx answers extractively, quoting source sentences verbatim.
"""
    )
