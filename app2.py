"""
app2.py — Pints AI: Automated Insurance Claims Processing
Run with: streamlit run app2.py

UI Flow:
  Sidebar  → API key, system prompt config, RAG knowledge base upload
  Tab 1    → Submit claim → instant analysis + auto email on same page
  Tab 2    → Communications log (what customers keep getting wrong)
  Tab 3    → Audit trail (password protected: 123)
"""

import json
import logging
import os
import tempfile
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path

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
#  Page config
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pints AI — Claims",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUDIT_PASSWORD = "123"

# ══════════════════════════════════════════════════════════════════════
#  Styling
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.cs-header {
    background: linear-gradient(135deg, #0f2942 0%, #1a5276 60%, #1a6ea8 100%);
    padding: 1.5rem 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
}
.cs-header h1 { margin: 0; font-size: 1.9rem; }
.cs-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

.banner { padding: 1rem 1.4rem; border-radius: 10px; margin: 0.75rem 0 1.25rem; }
.covered       { background:#d4edda; border-left:5px solid #28a745; }
.not-covered   { background:#f8d7da; border-left:5px solid #dc3545; }
.partial       { background:#fff3cd; border-left:5px solid #ffc107; }
.requires-info { background:#d1ecf1; border-left:5px solid #17a2b8; }
.out-of-scope  { background:#e2e3e5; border-left:5px solid #6c757d; }

.missing-item {
    padding: 0.6rem 0.8rem; margin: 0.3rem 0;
    border-radius: 6px; border-left: 4px solid; font-size: 0.9rem;
}
.p-high   { border-color:#dc3545; background:#fff5f5; }
.p-medium { border-color:#fd7e14; background:#fff9f0; }
.p-low    { border-color:#28a745; background:#f5fff7; }

.conf-bar-wrap { background:#e9ecef; border-radius:20px; height:12px; margin:6px 0; }
.conf-bar      { border-radius:20px; height:12px; }

.doc-badge {
    background:#eaf3fc; border:1px solid #b8daff;
    border-radius:6px; padding:0.35rem 0.6rem;
    margin:0.25rem 0; font-size:0.82rem;
}
.email-box {
    background:#f8f9fa; border:1px solid #dee2e6;
    border-radius:8px; padding:1rem; font-family:monospace;
    font-size:0.85rem; white-space:pre-wrap; margin-top:0.5rem;
}
.auto-sent-badge {
    background:#d4edda; color:#155724; border:1px solid #c3e6cb;
    border-radius:20px; padding:0.2rem 0.8rem; font-size:0.8rem;
    font-weight:600; display:inline-block; margin-bottom:0.5rem;
}
.comm-card {
    background:#fff; border:1px solid #e0e0e0; border-radius:8px;
    padding:0.8rem 1rem; margin:0.4rem 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  Session state
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
        "current_claim_id": None,
        "current_analysis": None,
        "claim_text": "",
        "auto_email": None,         # stores the auto-generated email dict
        "customer_email_addr": "",
        "customer_name": "",
        "analysis_history": [],
        "audit_unlocked": False,    # password gate
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════
def boot_backend(api_key: str) -> bool:
    try:
        rag   = RAGEngine(api_key)
        agent = ClaimsAgent(api_key, rag, st.session_state.system_prompt)
        comp  = EmailComposer(api_key)
        st.session_state.rag_engine    = rag
        st.session_state.claims_agent  = agent
        st.session_state.email_composer = comp
        st.session_state.api_key       = api_key
        return True
    except Exception as exc:
        st.error(f"Initialisation failed: {exc}")
        return False


def extract_text(uploaded_files) -> str:
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
                    tmp.write(raw); tmp_path = tmp.name
                pages = PyPDFLoader(tmp_path).load()
                os.unlink(tmp_path)
                sections.append(header + "\n" + "\n\n".join(p.page_content for p in pages))
            except Exception as e:
                sections.append(f"{header}\n[PDF error: {e}]")
        elif suffix in (".txt", ".md"):
            sections.append(header + "\n" + raw.decode("utf-8", errors="ignore"))
        else:
            sections.append(f"{header}\n[Unsupported format]")
    return "\n\n".join(sections)


def decision_html(decision: str, confidence: float) -> str:
    cfg = {
        "covered":            ("✅", "COVERED",                   "covered"),
        "not_covered":        ("❌", "NOT COVERED",                "not-covered"),
        "partial":            ("⚠️", "PARTIAL COVERAGE",           "partial"),
        "requires_more_info": ("📋", "MORE INFORMATION REQUIRED",  "requires-info"),
        "out_of_scope":       ("🚫", "OUT OF SCOPE",               "out-of-scope"),
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
</div>"""


def missing_item_html(item) -> str:
    if hasattr(item, "field"):
        field, reason, priority = item.field, item.reason, item.priority
    else:
        field    = item.get("field", "N/A")
        reason   = item.get("reason", "")
        priority = item.get("priority", "medium")
    css   = {"high":"p-high","medium":"p-medium","low":"p-low"}.get(priority,"p-medium")
    badge = {"high":"🔴 HIGH","medium":"🟡 MEDIUM","low":"🟢 LOW"}.get(priority, priority)
    return f"""
<div class="missing-item {css}">
  <strong>{field}</strong>
  <span style="float:right;font-size:0.78rem;opacity:.8">{badge}</span><br>
  <span style="color:#555">{reason}</span>
</div>"""


def auto_send_email(analysis, customer_name: str, customer_email: str) -> dict:
    """
    Auto-generate and auto-send (or log) a customer email immediately after analysis.
    Returns the email dict so it can be displayed and stored.
    """
    if not analysis.missing_information:
        return None

    result = st.session_state.email_composer.compose_missing_info_email(
        claim_id=analysis.claim_id,
        claim_summary=analysis.summary,
        missing_items=analysis.missing_information,
        customer_name=customer_name,
    )

    # Attempt real SMTP send; fall back to logged-only
    smtp = st.session_state.email_composer.send_via_smtp(
        to_address=customer_email,
        subject=result.get("subject", ""),
        body=result.get("body", ""),
    )
    result["smtp_status"] = smtp.get("status", "logged")

    # Log it
    st.session_state.audit_logger.log_event(
        event_type="EMAIL_AUTO_SENT",
        session_id=st.session_state.session_id,
        user_action="Email auto-generated and sent after analysis",
        claim_id=analysis.claim_id,
        metadata={
            "to": customer_email,
            "customer_name": customer_name,
            "subject": result.get("subject", ""),
            "body": result.get("body", ""),
            "missing_items": [
                (i.__dict__ if hasattr(i, "__dict__") else i)
                for i in analysis.missing_information
            ],
            "smtp_status": result["smtp_status"],
        },
    )
    return result


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Pints AI")
    st.markdown("---")

    st.markdown("### 🔑 Google API Key")
    api_input = st.text_input(
        "API Key", value=st.session_state.api_key, type="password",
        label_visibility="collapsed", placeholder="Paste your Google Gemini API key…",
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

    if st.session_state.claims_agent:
        st.success("✅ AI Engine Active")
    else:
        st.warning("⚠️ Enter API key to start")

    st.markdown("---")

    st.markdown("### 👤 Customer Details")
    st.caption("Pre-fill so emails are sent automatically on analysis")
    st.session_state.customer_name = st.text_input(
        "Customer name",
        value=st.session_state.customer_name or "",
        placeholder="e.g. Jane Smith",
    )
    st.session_state.customer_email_addr = st.text_input(
        "Customer email",
        value=st.session_state.customer_email_addr or "",
        placeholder="customer@example.com",
    )

    st.markdown("---")

    st.markdown("### ⚙️ System Prompt")
    with st.expander("View / Edit"):
        new_sp = st.text_area("SP", value=st.session_state.system_prompt,
                              height=200, label_visibility="collapsed")
        if st.button("💾 Save", use_container_width=True):
            st.session_state.system_prompt = new_sp
            if st.session_state.claims_agent:
                st.session_state.claims_agent.update_system_prompt(new_sp)
            st.success("Saved!")

    st.markdown("---")

    st.markdown("### 📚 Knowledge Base (RAG)")
    st.caption("Upload policy manuals and coverage guidelines")

    if not st.session_state.rag_engine:
        st.info("Initialise system first")
    else:
        rag_uploads = st.file_uploader(
            "Policy docs", type=["pdf","txt","md"],
            accept_multiple_files=True, key="rag_uploader",
            label_visibility="collapsed",
        )
        if rag_uploads:
            if st.button("📥 Load into Knowledge Base", use_container_width=True, type="primary"):
                with st.spinner("Embedding…"):
                    result = st.session_state.rag_engine.add_documents(rag_uploads)
                if result["status"] == "success":
                    st.success(f"✅ {result['pages']} pages → {result['chunks']} chunks")
                    st.session_state.audit_logger.log_event(
                        event_type="RAG_UPDATE",
                        session_id=st.session_state.session_id,
                        user_action=f"Added {result['files']} RAG documents",
                        metadata={"files":[f.name for f in rag_uploads],"chunks":result["chunks"]},
                    )
                else:
                    st.error(result["message"])

        if st.session_state.rag_engine.is_ready():
            for doc in st.session_state.rag_engine.get_registry():
                st.markdown(
                    f'<div class="doc-badge">📄 {doc["filename"]}'
                    f'<span style="float:right;opacity:.6">{doc["pages"]}p</span></div>',
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
st.markdown("""
<div class="cs-header">
  <h1>🛡️ Pints AI — Automated Claims Processing</h1>
  <p>RAG-powered policy verification · Instant analysis · Auto customer communication · Audit trails</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════
tab_submit, tab_comms, tab_audit = st.tabs([
    "📁 Submit & Review Claim",
    "📬 Communications Log",
    "📋 Audit Trail 🔒",
])

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — SUBMIT & REVIEW CLAIM
# ══════════════════════════════════════════════════════════════════════
with tab_submit:

    # ── Upload section ────────────────────────────────────────────────
    st.markdown("## 📎 Upload Claim Documents")
    st.caption(
        "Upload the customer's **claim form**, **email correspondence**, "
        "and their **policy document** together. Analysis runs instantly."
    )

    col_a, col_b = st.columns([3, 2])
    with col_a:
        claim_files = st.file_uploader(
            "Claim files", type=["pdf","txt","md"],
            accept_multiple_files=True, key="claim_uploader",
            label_visibility="collapsed",
        )
        if claim_files:
            for f in claim_files:
                st.markdown(f"  📄 `{f.name}` &nbsp; *{f.size/1024:.1f} KB*")

    with col_b:
        manual_notes = st.text_area(
            "Processor notes (optional)", height=120,
            label_visibility="collapsed",
            placeholder="e.g. Customer phoned to follow up on 12 May…",
        )

    # Pre-flight checks
    st.markdown("---")
    checks = {
        "🤖 AI Engine":       st.session_state.claims_agent is not None,
        "📚 Knowledge Base":  st.session_state.rag_engine is not None and st.session_state.rag_engine.is_ready(),
        "📎 Documents":       bool(claim_files),
        "👤 Customer email":  bool(st.session_state.customer_email_addr),
    }
    for col, (label, ok) in zip(st.columns(4), checks.items()):
        with col:
            (st.success if ok else st.error)(label)

    if not checks["📚 Knowledge Base"]:
        st.warning("⚠️ Upload policy documents via the sidebar before analysing. Without them every claim is Out of Scope.")
    if not checks["👤 Customer email"]:
        st.warning("⚠️ Add the customer's email in the sidebar so the follow-up email can be sent automatically.")

    can_run = st.session_state.claims_agent and bool(claim_files)

    if st.button("🚀 Analyse Claim", disabled=not can_run, type="primary", use_container_width=True):
        with st.spinner("Extracting document text…"):
            for f in claim_files: f.seek(0)
            claim_text = extract_text(claim_files)
            if manual_notes.strip():
                claim_text += f"\n\n{'═'*60}\nPROCESSOR NOTES\n{'═'*60}\n{manual_notes}"
            st.session_state.claim_text  = claim_text
            st.session_state.auto_email  = None  # reset

        with st.spinner("Analysing claim against knowledge base…"):
            analysis, rag_docs = st.session_state.claims_agent.analyze_claim(
                claim_text=claim_text, additional_info="", claim_id=None,
            )
            st.session_state.current_analysis  = analysis
            st.session_state.current_claim_id  = analysis.claim_id

            st.session_state.audit_logger.log_event(
                event_type="CLAIM_ANALYSIS",
                session_id=st.session_state.session_id,
                user_action="Claim analysis submitted",
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
                },
            )
            st.session_state.analysis_history.append({
                "ts": datetime.now().isoformat(),
                "claim_id": analysis.claim_id,
                "decision": analysis.coverage_decision,
                "confidence": analysis.confidence_score,
            })

        # Auto-send email if there's missing info
        if analysis.missing_information and st.session_state.customer_email_addr:
            with st.spinner("Auto-generating and sending customer email…"):
                email_result = auto_send_email(
                    analysis,
                    customer_name=st.session_state.customer_name or "Valued Customer",
                    customer_email=st.session_state.customer_email_addr,
                )
                st.session_state.auto_email = email_result

    # ── Results section (shown immediately below if analysis exists) ──
    if st.session_state.current_analysis:
        an = st.session_state.current_analysis
        st.markdown("---")
        st.markdown(f"## 📊 Analysis Result — `{an.claim_id}`")
        st.markdown(decision_html(an.coverage_decision, an.confidence_score), unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Claim Type",    an.claim_type)
        m2.metric("Confidence",    f"{an.confidence_score:.0%}")
        m3.metric("Missing Items", len(an.missing_information))
        m4.metric("Risk Flags",    len(an.risk_flags))

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
            st.markdown("### ⚠️ Missing Information")
            if an.missing_information:
                for item in an.missing_information:
                    st.markdown(missing_item_html(item), unsafe_allow_html=True)
            else:
                st.success("No missing information identified ✅")

            if an.risk_flags:
                st.markdown("### 🚩 Risk Flags")
                for flag in an.risk_flags:
                    (st.error if flag == "SYSTEM_ERROR" else st.warning)(f"{'❌' if flag=='SYSTEM_ERROR' else '⚠️'} {flag}")

        # ── Auto-email confirmation ───────────────────────────────────
        if st.session_state.auto_email:
            em = st.session_state.auto_email
            st.markdown("---")
            st.markdown("### 📧 Customer Email")
            st.markdown('<span class="auto-sent-badge">✅ AUTO-SENT</span>', unsafe_allow_html=True)
            st.caption(f"Sent to: **{st.session_state.customer_email_addr}**")

            with st.expander("View email that was sent", expanded=True):
                st.markdown(f"**Subject:** {em.get('subject','')}")
                st.markdown(
                    f'<div class="email-box">{em.get("body","")}</div>',
                    unsafe_allow_html=True,
                )
                # Fallback mailto link if SMTP not configured
                mailto = (
                    f"mailto:{urllib.parse.quote(st.session_state.customer_email_addr)}"
                    f"?subject={urllib.parse.quote(em.get('subject',''))}"
                    f"&body={urllib.parse.quote(em.get('body',''))}"
                )
                st.markdown(
                    f'<a href="{mailto}" target="_blank" style="font-size:0.85rem">📨 Open in mail client instead</a>',
                    unsafe_allow_html=True,
                )
        elif an.missing_information and not st.session_state.customer_email_addr:
            st.warning("⚠️ Add customer email in the sidebar to auto-send the follow-up email.")

        if an.coverage_decision == "out_of_scope":
            st.error(
                "### 🚫 Out of Scope\n\n"
                "This claim type was **not found** in the knowledge base. "
                "Upload the relevant policy documentation to the sidebar and re-analyse."
            )

        # ── Re-analysis with additional info ─────────────────────────
        st.markdown("---")
        st.markdown("### 🔄 Submit Additional Information")
        st.caption("Once the customer responds with the missing documents, upload or paste them here.")

        add_col1, add_col2 = st.columns(2)
        with add_col1:
            add_files = st.file_uploader(
                "Additional documents", type=["pdf","txt","md"],
                accept_multiple_files=True, key="add_uploader",
                label_visibility="collapsed",
            )
        with add_col2:
            add_text = st.text_area(
                "Additional information (text)", height=120,
                label_visibility="collapsed",
                placeholder="Paste text received from the customer…",
            )

        if st.button(
            "🔄 Re-analyse with Additional Information", type="primary",
            disabled=not (add_files or add_text.strip()),
        ):
            combined = add_text.strip()
            if add_files:
                for f in add_files: f.seek(0)
                combined = extract_text(add_files) + "\n\n" + combined

            with st.spinner("Re-analysing…"):
                new_an, _ = st.session_state.claims_agent.analyze_claim(
                    claim_text=st.session_state.claim_text,
                    additional_info=combined,
                    claim_id=an.claim_id,
                )
                st.session_state.current_analysis = new_an
                st.session_state.auto_email = None

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

            # Auto-send again if still missing info
            if new_an.missing_information and st.session_state.customer_email_addr:
                with st.spinner("Sending updated email to customer…"):
                    st.session_state.auto_email = auto_send_email(
                        new_an,
                        customer_name=st.session_state.customer_name or "Valued Customer",
                        customer_email=st.session_state.customer_email_addr,
                    )

            st.rerun()

        # ── Export report ─────────────────────────────────────────────
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
                (i.__dict__ if hasattr(i,"__dict__") else i)
                for i in an.missing_information
            ],
            "risk_flags": an.risk_flags,
            "rag_sources_used": an.rag_sources_used,
            "recommended_next_steps": an.recommended_next_steps,
        }
        st.download_button(
            "📥 Download Report (JSON)",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"{an.claim_id}_report.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — COMMUNICATIONS LOG
# ══════════════════════════════════════════════════════════════════════
with tab_comms:
    st.markdown("## 📬 Customer Communications Log")
    st.caption(
        "Every auto-sent email is logged here. Use this to identify patterns "
        "in what customers are repeatedly getting wrong."
    )

    if st.button("🔄 Refresh", key="refresh_comms"):
        st.rerun()

    all_logs = st.session_state.audit_logger.get_all_logs()
    email_logs = [
        l for l in all_logs
        if l["event_type"] == "EMAIL_AUTO_SENT"
    ]

    if not email_logs:
        st.info("No customer emails sent yet. Emails appear here automatically after each claim analysis.")
    else:
        # ── Summary stats ─────────────────────────────────────────────
        st.markdown(f"### 📊 Overview — {len(email_logs)} emails sent")

        # Aggregate all missing items across all emails
        all_missing: dict = {}
        for log in email_logs:
            try:
                meta = json.loads(log["metadata"]) if isinstance(log["metadata"], str) else log["metadata"]
                for item in meta.get("missing_items", []):
                    field = item.get("field", "Unknown") if isinstance(item, dict) else getattr(item, "field", "Unknown")
                    all_missing[field] = all_missing.get(field, 0) + 1
            except Exception:
                pass

        if all_missing:
            sorted_missing = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)
            st.markdown("### 🔁 Most Common Missing Items")
            st.caption("These are the things customers repeatedly fail to include in their claims.")

            max_count = sorted_missing[0][1]
            for field, count in sorted_missing:
                pct = int((count / max_count) * 100)
                bar_color = "#dc3545" if count == max_count else "#fd7e14" if pct > 50 else "#ffc107"
                st.markdown(f"**{field}** — {count} {'time' if count==1 else 'times'}")
                st.markdown(
                    f'<div class="conf-bar-wrap"><div class="conf-bar" '
                    f'style="width:{pct}%;background:{bar_color}"></div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Email log table ───────────────────────────────────────────
        st.markdown("### 📋 All Sent Emails")
        rows = []
        for log in email_logs:
            try:
                meta = json.loads(log["metadata"]) if isinstance(log["metadata"], str) else log["metadata"]
            except Exception:
                meta = {}
            rows.append({
                "Timestamp":     log["timestamp"][:19].replace("T", " "),
                "Claim ID":      log["claim_id"] or "—",
                "Sent To":       meta.get("to", "—"),
                "Customer":      meta.get("customer_name", "—"),
                "Missing Items": len(meta.get("missing_items", [])),
                "SMTP Status":   meta.get("smtp_status", "logged"),
                "Subject":       meta.get("subject", "—")[:60],
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=300)

        # ── Individual email viewer ───────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 View Individual Email")
        idx = st.selectbox(
            "Select email",
            range(len(email_logs)),
            format_func=lambda i: (
                f"{email_logs[i]['timestamp'][:19]}  |  "
                f"{email_logs[i]['claim_id']}  |  "
                f"{json.loads(email_logs[i]['metadata']).get('to','?') if isinstance(email_logs[i]['metadata'],str) else email_logs[i]['metadata'].get('to','?')}"
            ),
        )
        if idx is not None:
            log  = email_logs[idx]
            try:
                meta = json.loads(log["metadata"]) if isinstance(log["metadata"], str) else log["metadata"]
            except Exception:
                meta = {}

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**To:** {meta.get('to','—')}")
                st.markdown(f"**Subject:** {meta.get('subject','—')}")
                st.markdown(
                    f'<div class="email-box">{meta.get("body","No content")}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown("**Missing items in this email:**")
                for item in meta.get("missing_items", []):
                    field    = item.get("field","?") if isinstance(item,dict) else item.field
                    priority = item.get("priority","medium") if isinstance(item,dict) else item.priority
                    icon     = {"high":"🔴","medium":"🟡","low":"🟢"}.get(priority,"•")
                    st.markdown(f"{icon} {field}")

        # Export comms log
        st.markdown("---")
        st.download_button(
            "📥 Export Communications Log (JSON)",
            data=json.dumps(email_logs, indent=2, default=str),
            file_name=f"comms_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — AUDIT TRAIL (password protected)
# ══════════════════════════════════════════════════════════════════════
with tab_audit:
    st.markdown("## 📋 Audit Trail & Governance")

    # ── Password gate ─────────────────────────────────────────────────
    if not st.session_state.audit_unlocked:
        st.markdown("### 🔒 Restricted Access")
        st.caption("This section is for authorised personnel only.")
        pwd = st.text_input("Enter password", type="password", key="audit_pwd")
        if st.button("Unlock", type="primary"):
            if pwd == AUDIT_PASSWORD:
                st.session_state.audit_unlocked = True
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
        st.stop()

    # ── Unlocked ──────────────────────────────────────────────────────
    col_head, col_lock = st.columns([4, 1])
    with col_head:
        st.caption("Every AI decision, user action, and system event is logged here.")
    with col_lock:
        if st.button("🔒 Lock"):
            st.session_state.audit_unlocked = False
            st.rerun()

    if st.button("🔄 Refresh", key="refresh_audit"):
        st.rerun()

    logs  = st.session_state.audit_logger.get_all_logs()
    stats = st.session_state.audit_logger.get_summary_stats()

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Events",     stats["total_events"])
    s2.metric("Claims Processed", stats["total_claims"])
    s3.metric("Sessions",         stats["total_sessions"])
    s4.metric("Emails Sent",      stats["emails_sent"])
    s5.metric("Out-of-Scope",     stats["out_of_scope"])

    st.markdown("---")

    if not logs:
        st.info("No audit events yet.")
    else:
        f1, f2 = st.columns(2)
        with f1:
            type_filter = st.selectbox(
                "Filter by event type",
                ["All"] + sorted(set(l["event_type"] for l in logs)),
            )
        with f2:
            claim_filter = st.selectbox(
                "Filter by claim ID",
                ["All"] + sorted(set(l["claim_id"] for l in logs if l["claim_id"])),
            )

        filtered = logs
        if type_filter  != "All": filtered = [l for l in filtered if l["event_type"] == type_filter]
        if claim_filter != "All": filtered = [l for l in filtered if l["claim_id"]   == claim_filter]

        rows = []
        for l in filtered:
            rows.append({
                "Timestamp":  l["timestamp"][:19].replace("T"," "),
                "Event":      l["event_type"],
                "Action":     l["user_action"],
                "Claim ID":   l["claim_id"] or "—",
                "Decision":   l["decision"] or "—",
                "Confidence": f"{l['confidence_score']:.0%}" if l["confidence_score"] else "—",
                "Session":    l["session_id"][:8] + "…",
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)

        st.download_button(
            "📥 Export Full Log (JSON)",
            data=json.dumps(logs, indent=2, default=str),
            file_name=f"pints_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

        st.markdown("---")
        st.markdown("### 🔍 Log Inspector")
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
                for field in ("rag_sources","flags","metadata"):
                    try: entry[field] = json.loads(entry[field])
                    except Exception: pass
                st.json(entry)

        st.markdown("---")
        st.markdown("### 🗂️ Claim Event Timeline")
        if st.session_state.current_claim_id:
            for ev in st.session_state.audit_logger.get_claim_logs(st.session_state.current_claim_id):
                icon = {"CLAIM_ANALYSIS":"🔎","CLAIM_REANALYSIS":"🔄",
                        "EMAIL_AUTO_SENT":"📤","EMAIL_GENERATED":"✉️"}.get(ev["event_type"],"📌")
                st.markdown(
                    f"{icon} **{ev['timestamp'][:19].replace('T',' ')}** — "
                    f"`{ev['event_type']}` — {ev['user_action']}"
                    + (f" → **{ev['decision']}**" if ev["decision"] else "")
                )
        else:
            st.caption("Analyse a claim to see its timeline here.")

            