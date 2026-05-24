"""
app.py — ClaimSense AI: Automated Insurance Claims Processing
Run with: streamlit run app.py

UI Flow:
  Sidebar  → API key, system prompt config, RAG knowledge base upload
  Tab 1    → Upload & submit a new claim (claim form + email + policy)
  Tab 2    → Analysis results (summary, decision, missing info, risk flags)
  Tab 3    → Follow-up: submit additional info + generate/send customer email
  Tab 4    → Audit trail & governance dashboard
"""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from backend.audit_logger import AuditLogger
from backend.claims_agent import DEFAULT_SYSTEM_PROMPT, ClaimsAgent
from backend.email_composer import EmailComposer
from backend.rag_engine import RAGEngine

load_dotenv()

logger = logging.getLogger("pints.app")

# ══════════════════════════════════════════════════════════════════════
#  Page configuration
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ClaimSense AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  Styling
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Header ── */
.cs-header {
    background: linear-gradient(135deg, #0f2942 0%, #1a5276 60%, #1a6ea8 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
}
.cs-header h1 { margin: 0; font-size: 1.9rem; }
.cs-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

/* ── Decision banners ── */
.banner {
    padding: 1rem 1.4rem;
    border-radius: 10px;
    margin: 0.75rem 0 1.25rem;
}
.covered       { background:#d4edda; border-left:5px solid #28a745; }
.not-covered   { background:#f8d7da; border-left:5px solid #dc3545; }
.partial       { background:#fff3cd; border-left:5px solid #ffc107; }
.requires-info { background:#d1ecf1; border-left:5px solid #17a2b8; }
.out-of-scope  { background:#e2e3e5; border-left:5px solid #6c757d; }

/* ── Missing items ── */
.missing-item {
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    border-radius: 6px;
    border-left: 4px solid;
    font-size: 0.9rem;
}
.p-high   { border-color:#dc3545; background:#fff5f5; }
.p-medium { border-color:#fd7e14; background:#fff9f0; }
.p-low    { border-color:#28a745; background:#f5fff7; }

/* ── Confidence bar ── */
.conf-bar-wrap { background:#e9ecef; border-radius:20px; height:12px; margin:6px 0; }
.conf-bar      { border-radius:20px; height:12px; }

/* ── Sidebar doc list ── */
.doc-badge {
    background:#eaf3fc;
    border:1px solid #b8daff;
    border-radius:6px;
    padding:0.35rem 0.6rem;
    margin:0.25rem 0;
    font-size:0.82rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
#  Session state initialisation
# ══════════════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
        "rag_engine": None,
        "claims_agent": None,
        "email_composer": None,
        "audit_logger": AuditLogger(),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        # Current claim state
        "current_claim_id": None,
        "current_analysis": None,
        "claim_text": "",
        # Email state
        "email_draft": None,
        "customer_email_addr": "",
        "customer_name": "",
        # History
        "analysis_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()

# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════
def boot_backend(api_key: str) -> bool:
    """Initialise or re-initialise all backend components."""
    try:
        rag = RAGEngine(api_key)
        agent = ClaimsAgent(api_key, rag, st.session_state.system_prompt)
        composer = EmailComposer(api_key)

        st.session_state.rag_engine = rag
        st.session_state.claims_agent = agent
        st.session_state.email_composer = composer
        st.session_state.api_key = api_key
        return True
    except Exception as exc:
        st.error(f"Initialisation failed: {exc}")
        logger.error(f"Boot error: {exc}")
        return False


def extract_text_from_uploads(uploaded_files) -> str:
    """
    Combine text from all uploaded Streamlit files.
    Supports PDF, TXT, MD.
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    sections = []
    for f in uploaded_files:
        suffix = Path(f.name).suffix.lower()
        f.seek(0)
        raw = f.read()

        header = f"{'═'*60}\nDOCUMENT: {f.name}\n{'═'*60}"

        if suffix == ".pdf":
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(raw)
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                os.unlink(tmp_path)
                content = "\n\n".join(p.page_content for p in pages)
                sections.append(f"{header}\n{content}")
            except Exception as exc:
                sections.append(f"{header}\n[PDF extraction error: {exc}]")
        elif suffix in (".txt", ".md"):
            try:
                sections.append(f"{header}\n{raw.decode('utf-8', errors='ignore')}")
            except Exception as exc:
                sections.append(f"{header}\n[Text decode error: {exc}]")
        else:
            sections.append(f"{header}\n[Unsupported format — please use PDF or TXT]")

    return "\n\n".join(sections)


def decision_html(decision: str, confidence: float) -> str:
    """Return HTML for a styled decision banner."""
    cfg = {
        "covered":            ("✅", "COVERED",              "covered"),
        "not_covered":        ("❌", "NOT COVERED",           "not-covered"),
        "partial":            ("⚠️", "PARTIAL COVERAGE",      "partial"),
        "requires_more_info": ("📋", "MORE INFORMATION REQUIRED", "requires-info"),
        "out_of_scope":       ("🚫", "OUT OF SCOPE",          "out-of-scope"),
    }
    icon, label, css = cfg.get(decision, ("❓", "UNKNOWN", "requires-info"))

    pct = int(confidence * 100)
    bar_color = "#28a745" if pct >= 70 else "#ffc107" if pct >= 40 else "#dc3545"

    return f"""
<div class="banner {css}">
  <div style="font-size:1.6rem;font-weight:700">{icon} {label}</div>
  <div style="margin-top:6px;font-size:0.88rem;color:#444">
    Model confidence: <strong>{pct}%</strong>
  </div>
  <div class="conf-bar-wrap">
    <div class="conf-bar" style="width:{pct}%;background:{bar_color}"></div>
  </div>
</div>
"""


def missing_item_html(item) -> str:
    if hasattr(item, "field"):
        field, reason, priority = item.field, item.reason, item.priority
    else:
        field = item.get("field", "N/A")
        reason = item.get("reason", "")
        priority = item.get("priority", "medium")

    css = {"high": "p-high", "medium": "p-medium", "low": "p-low"}.get(priority, "p-medium")
    badge = {"high": "🔴 HIGH", "medium": "🟡 MEDIUM", "low": "🟢 LOW"}.get(priority, priority)

    return f"""
<div class="missing-item {css}">
  <strong>{field}</strong>
  <span style="float:right;font-size:0.78rem;opacity:.8">{badge}</span><br>
  <span style="color:#555">{reason}</span>
</div>
"""


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ ClaimSense AI")
    st.markdown("---")

    # ── API Key ──────────────────────────────────────────────────────
    st.markdown("### 🔑 Google API Key")
    api_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        label_visibility="collapsed",
        placeholder="Paste your Google Gemini API key…",
    )

    if api_input and (api_input != st.session_state.api_key or st.session_state.claims_agent is None):
        with st.spinner("Starting AI engine…"):
            if boot_backend(api_input):
                st.session_state.audit_logger.log_event(
                    event_type="SYSTEM_INIT",
                    session_id=st.session_state.session_id,
                    user_action="Backend initialised",
                    model_used="gemini-2.5-flash",
                )

    # Status indicator
    if st.session_state.claims_agent:
        st.success("✅ AI Engine Active")
    else:
        st.warning("⚠️ Enter API key to start")

    st.markdown("---")

    # ── System Prompt ─────────────────────────────────────────────────
    st.markdown("### ⚙️ System Prompt")
    with st.expander("View / Edit System Prompt"):
        new_sp = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=220,
            label_visibility="collapsed",
        )
        if st.button("💾 Save Prompt", use_container_width=True):
            st.session_state.system_prompt = new_sp
            if st.session_state.claims_agent:
                st.session_state.claims_agent.update_system_prompt(new_sp)
                st.session_state.audit_logger.log_event(
                    event_type="SYSTEM_CONFIG",
                    session_id=st.session_state.session_id,
                    user_action="System prompt updated",
                )
            st.success("Prompt updated!")

    st.markdown("---")

    # ── RAG Knowledge Base ────────────────────────────────────────────
    st.markdown("### 📚 Knowledge Base (RAG)")
    st.caption("Upload policy manuals, coverage guidelines, and procedure docs")

    if not st.session_state.rag_engine:
        st.info("Initialise the system first")
    else:
        rag_uploads = st.file_uploader(
            "Policy documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="rag_uploader",
            label_visibility="collapsed",
            help="Supported: PDF, TXT, MD",
        )

        if rag_uploads:
            if st.button("📥 Load into Knowledge Base", use_container_width=True, type="primary"):
                with st.spinner("Embedding documents…"):
                    result = st.session_state.rag_engine.add_documents(rag_uploads)

                if result["status"] == "success":
                    st.success(
                        f"✅ Loaded {result['pages']} pages → {result['chunks']} chunks"
                    )
                    st.session_state.audit_logger.log_event(
                        event_type="RAG_UPDATE",
                        session_id=st.session_state.session_id,
                        user_action=f"Added {result['files']} RAG documents",
                        metadata={
                            "files": [f.name for f in rag_uploads],
                            "chunks": result["chunks"],
                        },
                    )
                else:
                    st.error(result["message"])

        # Show registry
        if st.session_state.rag_engine.is_ready():
            st.markdown("**Loaded documents:**")
            for doc in st.session_state.rag_engine.get_registry():
                st.markdown(
                    f'<div class="doc-badge">📄 {doc["filename"]}'
                    f'<span style="float:right;opacity:.6">{doc["pages"]}p · {doc["size_kb"]}KB</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No documents loaded yet")

    st.markdown("---")
    st.caption(f"Session `{st.session_state.session_id[:12]}…`")
    st.caption(f"🕐 {datetime.now().strftime('%d %b %Y %H:%M')}")

# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="cs-header">
  <h1>🛡️ ClaimSense AI — Automated Claims Processing</h1>
  <p>RAG-powered policy verification · Structured analysis · Governance & audit trails · Customer communication</p>
</div>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════
tab_new, tab_results, tab_followup, tab_audit = st.tabs(
    ["📁 New Claim", "📊 Analysis Results", "📧 Follow-up & Email", "📋 Audit Trail"]
)

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — NEW CLAIM
# ══════════════════════════════════════════════════════════════════════
with tab_new:
    st.markdown("## 📁 Submit New Claim")
    st.markdown(
        "Upload all documents related to this claim. The AI will extract text automatically."
    )

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("### 📎 Claim Documents")
        st.caption(
            "Upload the customer's **claim form**, **correspondence / email**, "
            "and their **current policy document** together."
        )
        claim_files = st.file_uploader(
            "Claim files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="claim_uploader",
            label_visibility="collapsed",
        )
        if claim_files:
            for f in claim_files:
                st.markdown(f"  📄 `{f.name}` &nbsp; *{f.size/1024:.1f} KB*")

    with col_b:
        st.markdown("### 📝 Processor Notes")
        st.caption("Add any context you want the AI to factor in (optional)")
        manual_notes = st.text_area(
            "Notes",
            height=140,
            label_visibility="collapsed",
            placeholder="e.g. Customer phoned in, original claim lodged on 12 May…",
        )

    st.markdown("---")

    # Pre-flight check row
    checks = {
        "🤖 AI Engine": st.session_state.claims_agent is not None,
        "📚 Knowledge Base": (
            st.session_state.rag_engine is not None
            and st.session_state.rag_engine.is_ready()
        ),
        "📎 Documents uploaded": bool(claim_files),
    }
    cols = st.columns(3)
    for col, (label, ok) in zip(cols, checks.items()):
        with col:
            if ok:
                st.success(label)
            else:
                st.error(label)

    if not checks["📚 Knowledge Base"]:
        st.warning(
            "⚠️ **Knowledge base is empty.** "
            "Upload policy documents via the sidebar before analysing claims. "
            "Without them, every claim will be marked **Out of Scope**."
        )

    # ── Analyse button ────────────────────────────────────────────────
    can_run = st.session_state.claims_agent and bool(claim_files)

    if st.button(
        "🚀 Analyse Claim",
        disabled=not can_run,
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Extracting document text…"):
            for f in claim_files:
                f.seek(0)
            claim_text = extract_text_from_uploads(claim_files)
            if manual_notes.strip():
                claim_text += f"\n\n{'═'*60}\nPROCESSOR NOTES\n{'═'*60}\n{manual_notes}"
            st.session_state.claim_text = claim_text

        with st.spinner("Analysing claim against policy knowledge base…"):
            analysis, rag_docs = st.session_state.claims_agent.analyze_claim(
                claim_text=claim_text,
                additional_info="",
                claim_id=None,
            )
            st.session_state.current_analysis = analysis
            st.session_state.current_claim_id = analysis.claim_id
            st.session_state.email_draft = None  # reset from previous claim

            st.session_state.audit_logger.log_event(
                event_type="CLAIM_ANALYSIS",
                session_id=st.session_state.session_id,
                user_action="Initial claim analysis submitted",
                claim_id=analysis.claim_id,
                input_summary=claim_text[:300],
                output_summary=analysis.summary,
                model_used="gemini-2.5-flash",
                rag_sources=analysis.rag_sources_used,
                decision=analysis.coverage_decision,
                confidence_score=analysis.confidence_score,
                flags=analysis.risk_flags,
                metadata={
                    "files": [f.name for f in claim_files],
                    "missing_count": len(analysis.missing_information),
                    "rag_docs_retrieved": len(rag_docs),
                },
            )
            st.session_state.analysis_history.append(
                {
                    "ts": datetime.now().isoformat(),
                    "claim_id": analysis.claim_id,
                    "decision": analysis.coverage_decision,
                    "confidence": analysis.confidence_score,
                }
            )

        st.success(f"✅ Analysis complete! Claim ID: **{analysis.claim_id}**")
        st.info("Switch to the **📊 Analysis Results** tab to review the report.")


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYSIS RESULTS
# ══════════════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state.current_analysis:
        st.info("📁 Submit a claim in the **New Claim** tab first.")
    else:
        an = st.session_state.current_analysis

        st.markdown(f"## 📊 Analysis Report")
        st.markdown(
            f"**Claim ID:** `{an.claim_id}` &nbsp;·&nbsp; "
            f"**Type:** {an.claim_type} &nbsp;·&nbsp; "
            f"Generated {datetime.now().strftime('%d %b %Y %H:%M')}"
        )

        # Decision banner
        st.markdown(
            decision_html(an.coverage_decision, an.confidence_score),
            unsafe_allow_html=True,
        )

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Claim Type", an.claim_type)
        m2.metric("Confidence", f"{an.confidence_score:.0%}")
        m3.metric("Missing Items", len(an.missing_information))
        m4.metric("Risk Flags", len(an.risk_flags))

        st.markdown("---")

        left, right = st.columns([3, 2])

        with left:
            st.markdown("### 📝 Claim Summary")
            st.info(an.summary)

            st.markdown("### ⚖️ Coverage Determination")
            st.markdown(an.coverage_reasoning)

            if an.rag_sources_used:
                st.markdown("### 📚 Policy Sources Referenced")
                for src in an.rag_sources_used:
                    st.markdown(f"  &nbsp; 📄 {src}")

            if an.recommended_next_steps:
                st.markdown("### 👣 Recommended Next Steps")
                for i, step in enumerate(an.recommended_next_steps, 1):
                    st.markdown(f"**{i}.** {step}")

        with right:
            # Missing information panel
            st.markdown("### ⚠️ Missing Information")
            if an.missing_information:
                for item in an.missing_information:
                    st.markdown(missing_item_html(item), unsafe_allow_html=True)
            else:
                st.success("No missing information identified ✅")

            # Risk flags
            if an.risk_flags:
                st.markdown("### 🚩 Risk Flags")
                for flag in an.risk_flags:
                    if flag in ("SYSTEM_ERROR",):
                        st.error(f"❌ {flag}")
                    else:
                        st.warning(f"⚠️ {flag}")

        # Out-of-scope callout
        if an.coverage_decision == "out_of_scope":
            st.error(
                "### 🚫 This claim is out of scope\n\n"
                "The claim type or policy referenced was **not found** in the knowledge base. "
                "Please upload the relevant policy documentation to the sidebar and re-analyse. "
                "Do **not** make coverage decisions without the appropriate policy documentation."
            )

        # ── Export ───────────────────────────────────────────────────
        st.markdown("---")
        report = {
            "claim_id": an.claim_id,
            "generated_at": datetime.now().isoformat(),
            "session_id": st.session_state.session_id,
            "summary": an.summary,
            "claim_type": an.claim_type,
            "coverage_decision": an.coverage_decision,
            "coverage_reasoning": an.coverage_reasoning,
            "confidence_score": an.confidence_score,
            "missing_information": [
                (item.__dict__ if hasattr(item, "__dict__") else item)
                for item in an.missing_information
            ],
            "risk_flags": an.risk_flags,
            "rag_sources_used": an.rag_sources_used,
            "recommended_next_steps": an.recommended_next_steps,
        }

        dl_col, info_col = st.columns([1, 2])
        with dl_col:
            st.download_button(
                "📥 Download Report (JSON)",
                data=json.dumps(report, indent=2, default=str),
                file_name=f"{an.claim_id}_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with info_col:
            if an.requires_additional_info:
                st.info(
                    "📧 This claim requires more information. "
                    "Switch to **Follow-up & Email** to generate a customer email."
                )


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — FOLLOW-UP & EMAIL
# ══════════════════════════════════════════════════════════════════════
with tab_followup:
    st.markdown("## 📧 Follow-up & Customer Communication")

    if not st.session_state.current_analysis:
        st.info("📁 No claim analysed yet. Start in the **New Claim** tab.")
    else:
        an = st.session_state.current_analysis
        st.markdown(f"**Active Claim:** `{an.claim_id}` &nbsp;·&nbsp; Decision: **{an.coverage_decision}**")
        st.markdown("---")

        # ── Section A: Submit additional information ──────────────────
        st.markdown("### 📤 A. Submit Additional Information from Customer")
        st.caption(
            "When you receive the additional documents or information from the customer, "
            "upload or paste them here and re-analyse."
        )

        add_col1, add_col2 = st.columns(2)
        with add_col1:
            add_files = st.file_uploader(
                "Additional documents",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True,
                key="add_uploader",
                label_visibility="collapsed",
            )
        with add_col2:
            add_text = st.text_area(
                "Additional information (text)",
                height=130,
                label_visibility="collapsed",
                placeholder="Paste text received from the customer…",
            )

        if st.button(
            "🔄 Re-analyse with Additional Information",
            type="primary",
            disabled=not (add_files or add_text.strip()),
        ):
            combined = add_text.strip()
            if add_files:
                for f in add_files:
                    f.seek(0)
                combined = extract_text_from_uploads(add_files) + "\n\n" + combined

            with st.spinner("Re-analysing…"):
                new_an, rag_docs = st.session_state.claims_agent.analyze_claim(
                    claim_text=st.session_state.claim_text,
                    additional_info=combined,
                    claim_id=an.claim_id,
                )
                st.session_state.current_analysis = new_an

                st.session_state.audit_logger.log_event(
                    event_type="CLAIM_REANALYSIS",
                    session_id=st.session_state.session_id,
                    user_action="Re-analysis with additional information",
                    claim_id=an.claim_id,
                    input_summary=combined[:300],
                    output_summary=new_an.summary,
                    model_used="gemini-2.5-flash",
                    rag_sources=new_an.rag_sources_used,
                    decision=new_an.coverage_decision,
                    confidence_score=new_an.confidence_score,
                    flags=new_an.risk_flags,
                )

            st.success("✅ Re-analysis complete!")
            st.info("Switch to **Analysis Results** to see the updated decision.")

        st.markdown("---")

        # ── Section B: Customer email ─────────────────────────────────
        st.markdown("### 📧 B. Generate Customer Email (Request for Information)")

        if not an.missing_information:
            st.success(
                "✅ No missing information was identified — no follow-up email needed. "
                "Proceed to final decision."
            )
        else:
            ec1, ec2 = st.columns(2)
            with ec1:
                st.session_state.customer_name = st.text_input(
                    "Customer name",
                    value=st.session_state.customer_name or "Valued Customer",
                )
            with ec2:
                st.session_state.customer_email_addr = st.text_input(
                    "Customer email address",
                    value=st.session_state.customer_email_addr,
                    placeholder="customer@example.com",
                )

            if st.button("✉️ Generate Email Draft", type="secondary"):
                with st.spinner("Drafting professional email…"):
                    result = st.session_state.email_composer.compose_missing_info_email(
                        claim_id=an.claim_id,
                        claim_summary=an.summary,
                        missing_items=an.missing_information,
                        customer_name=st.session_state.customer_name,
                    )
                    st.session_state.email_draft = result

                    st.session_state.audit_logger.log_event(
                        event_type="EMAIL_GENERATED",
                        session_id=st.session_state.session_id,
                        user_action="Customer email draft generated",
                        claim_id=an.claim_id,
                        metadata={
                            "to": st.session_state.customer_email_addr,
                            "missing_items": len(an.missing_information),
                        },
                    )

            # ── Email draft display ───────────────────────────────────
            if st.session_state.email_draft:
                ed = st.session_state.email_draft
                st.markdown("#### ✉️ Email Draft")
                st.caption(
                    "Review and edit before sending. "
                    "The subject and body are fully editable below."
                )

                draft_subject = st.text_input("Subject line", value=ed.get("subject", ""))
                draft_body = st.text_area("Email body", value=ed.get("body", ""), height=320)

                # Action buttons
                btn_col1, btn_col2, btn_col3 = st.columns(3)

                with btn_col1:
                    # Download as plain text
                    email_txt = (
                        f"To: {st.session_state.customer_email_addr}\n"
                        f"Subject: {draft_subject}\n\n"
                        f"{draft_body}"
                    )
                    st.download_button(
                        "📥 Download Email",
                        data=email_txt,
                        file_name=f"{an.claim_id}_customer_email.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                with btn_col2:
                    # mailto: link for mail client
                    import urllib.parse
                    mailto = (
                        f"mailto:{urllib.parse.quote(st.session_state.customer_email_addr)}"
                        f"?subject={urllib.parse.quote(draft_subject)}"
                        f"&body={urllib.parse.quote(draft_body)}"
                    )
                    st.markdown(
                        f'<a href="{mailto}" target="_blank">'
                        f'<button style="width:100%;padding:0.5rem;background:#1a5276;'
                        f'color:white;border:none;border-radius:6px;cursor:pointer;font-size:0.9rem">'
                        f"📨 Open in Mail App</button></a>",
                        unsafe_allow_html=True,
                    )

                with btn_col3:
                    # SMTP send (requires .env config) OR just mark as sent
                    if st.button("📤 Mark as Sent", type="primary", use_container_width=True):
                        # Attempt SMTP send if configured
                        smtp_result = st.session_state.email_composer.send_via_smtp(
                            to_address=st.session_state.customer_email_addr,
                            subject=draft_subject,
                            body=draft_body,
                        )

                        st.session_state.audit_logger.log_event(
                            event_type="EMAIL_SENT",
                            session_id=st.session_state.session_id,
                            user_action="Customer email marked as sent",
                            claim_id=an.claim_id,
                            metadata={
                                "to": st.session_state.customer_email_addr,
                                "subject": draft_subject,
                                "smtp_result": smtp_result.get("status"),
                            },
                        )

                        if smtp_result.get("status") == "sent":
                            st.success(f"✅ Email sent to {st.session_state.customer_email_addr}!")
                        else:
                            st.success("✅ Email logged as sent in audit trail.")
                            if smtp_result.get("status") == "error":
                                st.caption(
                                    f"ℹ️ SMTP not configured: {smtp_result.get('message','')}. "
                                    "Configure SMTP in .env for direct sending."
                                )

                # Missing items quick reference
                with st.expander("📋 Missing Items Summary"):
                    for item in an.missing_information:
                        st.markdown(missing_item_html(item), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════
with tab_audit:
    st.markdown("## 📋 Audit Trail & Governance")
    st.caption(
        "Every AI decision, user action, and system event is logged here with full metadata. "
        "This log is persisted to **audit_trail.db** and **claimsense.log** on disk."
    )

    refresh_col, _ = st.columns([1, 4])
    with refresh_col:
        if st.button("🔄 Refresh"):
            st.rerun()

    logs = st.session_state.audit_logger.get_all_logs()
    stats = st.session_state.audit_logger.get_summary_stats()

    # Stats row
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Events", stats["total_events"])
    s2.metric("Claims Processed", stats["total_claims"])
    s3.metric("Sessions", stats["total_sessions"])
    s4.metric("Emails Sent", stats["emails_sent"])
    s5.metric("Out-of-Scope", stats["out_of_scope"])

    st.markdown("---")

    if not logs:
        st.info("No audit events yet.")
    else:
        # Filters
        f1, f2 = st.columns(2)
        with f1:
            all_types = sorted(set(l["event_type"] for l in logs))
            type_filter = st.selectbox("Filter by event type", ["All"] + all_types)
        with f2:
            all_claims = sorted(set(l["claim_id"] for l in logs if l["claim_id"]))
            claim_filter = st.selectbox(
                "Filter by claim ID", ["All"] + all_claims
            )

        filtered = logs
        if type_filter != "All":
            filtered = [l for l in filtered if l["event_type"] == type_filter]
        if claim_filter != "All":
            filtered = [l for l in filtered if l["claim_id"] == claim_filter]

        # Table view
        rows = []
        for l in filtered:
            rows.append(
                {
                    "Timestamp": l["timestamp"][:19].replace("T", " "),
                    "Event": l["event_type"],
                    "Action": l["user_action"],
                    "Claim ID": l["claim_id"] or "—",
                    "Decision": l["decision"] or "—",
                    "Confidence": (
                        f"{l['confidence_score']:.0%}" if l["confidence_score"] else "—"
                    ),
                    "Session": l["session_id"][:8] + "…",
                }
            )

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=350)

        # Export
        exp_col, _ = st.columns([1, 3])
        with exp_col:
            st.download_button(
                "📥 Export Full Log (JSON)",
                data=json.dumps(logs, indent=2, default=str),
                file_name=f"claimsense_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")

        # Log inspector
        st.markdown("### 🔍 Detailed Log Inspector")
        if filtered:
            idx = st.selectbox(
                "Select entry",
                range(len(filtered)),
                format_func=lambda i: (
                    f"{filtered[i]['timestamp'][:19]}  |  "
                    f"{filtered[i]['event_type']}  |  "
                    f"{filtered[i]['claim_id'] or 'no claim'}"
                ),
            )
            if idx is not None:
                entry = dict(filtered[idx])
                # Parse JSON fields for nicer display
                for field in ("rag_sources", "flags", "metadata"):
                    try:
                        entry[field] = json.loads(entry[field])
                    except Exception:
                        pass
                st.json(entry)

        # Claim-level timeline
        st.markdown("---")
        st.markdown("### 🗂️ Claim Event Timeline")
        if st.session_state.current_claim_id:
            claim_logs = st.session_state.audit_logger.get_claim_logs(
                st.session_state.current_claim_id
            )
            if claim_logs:
                for ev in claim_logs:
                    icon = {
                        "CLAIM_ANALYSIS": "🔎",
                        "CLAIM_REANALYSIS": "🔄",
                        "EMAIL_GENERATED": "✉️",
                        "EMAIL_SENT": "📤",
                    }.get(ev["event_type"], "📌")
                    st.markdown(
                        f"{icon} **{ev['timestamp'][:19].replace('T',' ')}** — "
                        f"`{ev['event_type']}` — {ev['user_action']} "
                        + (f"→ **{ev['decision']}**" if ev["decision"] else "")
                    )
        else:
            st.caption("Analyse a claim to see its event timeline here.")


            