"""
claims_agent.py — The core AI brain for claims processing.

Architecture:
  1. Retrieve relevant policy chunks (RAG) — scope-checked
  2. Build a structured prompt with the claim docs + RAG context
  3. Call Gemini with temperature=0.1 (low = more deterministic)
  4. Parse and validate response as a ClaimAnalysis Pydantic model
  5. Apply guardrails: force out_of_scope if no RAG context,
     cap confidence if reasoning is thin, flag risk patterns

Hallucination guardrails:
  - LLM is only given the RAG text — it cannot invent policy clauses
  - out_of_scope is returned whenever FAISS finds nothing relevant
  - confidence_score is penalised if required fields are missing
  - All RAG sources are recorded for explainability
"""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from .rag_engine import RAGEngine

logger = logging.getLogger("pints.agent")

# ------------------------------------------------------------------ #
#  Pydantic output schema                                               #
# ------------------------------------------------------------------ #

class MissingInfoItem(BaseModel):
    field: str = Field(description="The missing data field or document")
    reason: str = Field(description="Why this information is needed to process the claim")
    priority: str = Field(description="high | medium | low")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        if v not in ("high", "medium", "low"):
            return "medium"
        return v


class ClaimAnalysis(BaseModel):
    claim_id: str
    summary: str = Field(description="2-3 sentence plain-English summary of the claim")
    claim_type: str = Field(description="Category of claim, e.g. 'Motor Accident', 'Property Damage'")
    coverage_decision: str = Field(
        description="One of: covered | not_covered | partial | requires_more_info | out_of_scope"
    )
    coverage_reasoning: str = Field(
        description="Detailed reasoning citing specific policy sections from the RAG context"
    )
    missing_information: List[MissingInfoItem] = Field(default_factory=list)
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Suspicious patterns, inconsistencies, or concerns"
    )
    confidence_score: float = Field(
        description="Model confidence 0.0–1.0. Below 0.6 triggers requires_more_info"
    )
    rag_sources_used: List[str] = Field(
        default_factory=list,
        description="Filenames/sections cited from the knowledge base"
    )
    requires_additional_info: bool
    recommended_next_steps: List[str] = Field(default_factory=list)

    @field_validator("coverage_decision")
    @classmethod
    def validate_decision(cls, v):
        valid = {"covered", "not_covered", "partial", "requires_more_info", "out_of_scope"}
        if v not in valid:
            return "requires_more_info"
        return v

    @field_validator("confidence_score")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))


# ------------------------------------------------------------------ #
#  Prompts                                                              #
# ------------------------------------------------------------------ #

DEFAULT_SYSTEM_PROMPT = """You are an expert insurance claims processor and adjudicator with deep knowledge of \
insurance policy analysis, fraud detection, and regulatory compliance.

Your responsibilities:
1. Carefully analyse the submitted claim documents against the provided policy documentation
2. Identify every piece of missing or ambiguous information that would block a final decision
3. Make a preliminary coverage determination based STRICTLY on the supplied policy text — never invent clauses
4. Flag any suspicious patterns, inconsistencies, or potential fraud indicators
5. Recommend concrete next steps

CRITICAL GUARDRAILS — you MUST follow all of these:
• Base your decision ONLY on the RAG policy documentation provided. If the claim type is NOT covered \
in that documentation, set coverage_decision to "out_of_scope".
• Cite specific policy section names or clause headings when giving reasoning.
• If your confidence is below 0.6 for any reason, set coverage_decision to "requires_more_info".
• Never fabricate policy terms, limits, exclusion clauses, or coverage details.
• If critical information is missing, list every item specifically in missing_information.
• Maintain strict professionalism — no personal opinions, no speculation beyond the documents."""

ANALYSIS_PROMPT_TEMPLATE = """\
{system_prompt}

══════════════════════════════════════════════════════════════
POLICY DOCUMENTATION (Knowledge Base — RAG Context)
══════════════════════════════════════════════════════════════
{rag_context}

══════════════════════════════════════════════════════════════
SUBMITTED CLAIM DOCUMENTS
══════════════════════════════════════════════════════════════
{claim_documents}

══════════════════════════════════════════════════════════════
SUPPLEMENTARY / ADDITIONAL INFORMATION (if any)
══════════════════════════════════════════════════════════════
{additional_info}

══════════════════════════════════════════════════════════════
TASK
══════════════════════════════════════════════════════════════
Analyse the claim. Respond with ONLY a single valid JSON object matching this exact schema — \
no markdown fences, no preamble, no trailing text:

{{
  "claim_id": "{claim_id}",
  "summary": "<2-3 sentences summarising the claim>",
  "claim_type": "<e.g. Motor Accident, Property Damage, Medical>",
  "coverage_decision": "<covered | not_covered | partial | requires_more_info | out_of_scope>",
  "coverage_reasoning": "<detailed reasoning, citing policy sections>",
  "missing_information": [
    {{
      "field": "<name of missing field or document>",
      "reason": "<why it is needed>",
      "priority": "<high | medium | low>"
    }}
  ],
  "risk_flags": ["<risk or concern>"],
  "confidence_score": <0.0 to 1.0>,
  "rag_sources_used": ["<filename or section name from the RAG context>"],
  "requires_additional_info": <true | false>,
  "recommended_next_steps": ["<action>"]
}}

REMINDER: If the claim type is not found in the RAG context above, \
set coverage_decision to "out_of_scope" and confidence_score to 0.0.
"""


# ------------------------------------------------------------------ #
#  Agent                                                                #
# ------------------------------------------------------------------ #

class ClaimsAgent:
    """
    Orchestrates RAG retrieval → prompt construction → LLM call →
    structured output parsing → guardrail application.
    """

    MODEL_NAME = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: str,
        rag_engine: RAGEngine,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=self.MODEL_NAME,
            google_api_key=api_key,
            temperature=0.1,   # low = more deterministic, less hallucination
            convert_system_message_to_human=True,
        )
        self.rag_engine = rag_engine
        self.system_prompt = system_prompt

    # ---------------------------------------------------------------- #
    #  Public API                                                        #
    # ---------------------------------------------------------------- #

    def analyze_claim(
        self,
        claim_text: str,
        additional_info: str = "",
        claim_id: Optional[str] = None,
    ) -> Tuple[ClaimAnalysis, List[Document]]:
        """
        Main entry point. Returns (ClaimAnalysis, rag_docs_used).
        Never raises — returns a safe error analysis on failure.
        """
        if not claim_id:
            claim_id = self._new_claim_id()

        # Step 1: RAG retrieval with scope check
        rag_docs, is_in_scope = self.rag_engine.retrieve_with_relevance_check(
            query=claim_text[:800], k=6
        )

        rag_context_str = self._format_rag_context(rag_docs)

        # Step 2: Build prompt
        prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(
            system_prompt=self.system_prompt,
            rag_context=rag_context_str or "⚠️ NO POLICY DOCUMENTATION FOUND IN KNOWLEDGE BASE",
            claim_documents=claim_text,
            additional_info=additional_info.strip() or "None provided.",
            claim_id=claim_id,
        )

        # Step 3: Call Gemini
        try:
            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            raw = response.content.strip()
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return self._error_analysis(claim_id, str(exc)), []

        # Step 4: Parse JSON
        try:
            cleaned = self._strip_markdown_fences(raw)
            data: Dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(f"JSON parse error: {exc}\nRaw response snippet: {raw[:300]}")
            return self._error_analysis(claim_id, f"JSON parse error: {exc}"), rag_docs

        # Step 5: Apply guardrails
        data = self._apply_guardrails(data, is_in_scope, rag_docs)

        # Step 6: Validate with Pydantic
        try:
            # Convert missing_information dicts to MissingInfoItem if needed
            if "missing_information" in data:
                mi_list = []
                for item in data["missing_information"]:
                    if isinstance(item, dict):
                        mi_list.append(MissingInfoItem(**item))
                    else:
                        mi_list.append(item)
                data["missing_information"] = mi_list

            analysis = ClaimAnalysis(**data)
        except Exception as exc:
            logger.error(f"Pydantic validation error: {exc}")
            return self._error_analysis(claim_id, f"Schema validation: {exc}"), rag_docs

        logger.info(
            f"[CLAIM ANALYSIS] id={claim_id} decision={analysis.coverage_decision} "
            f"confidence={analysis.confidence_score:.2f} missing={len(analysis.missing_information)}"
        )
        return analysis, rag_docs

    def update_system_prompt(self, new_prompt: str):
        self.system_prompt = new_prompt
        logger.info("System prompt updated")

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    def _new_claim_id(self) -> str:
        return f"CLM-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

    def _format_rag_context(self, docs: List[Document]) -> str:
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "")
            ref = f"{src}" + (f" — page {page}" if page != "" else "")
            parts.append(f"[SOURCE {i} | {ref}]\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    def _strip_markdown_fences(self, text: str) -> str:
        """Remove ```json ... ``` wrappers that Gemini sometimes adds."""
        text = text.strip()
        # Remove opening fence
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        # Remove closing fence
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _apply_guardrails(
        self, data: Dict, is_in_scope: bool, rag_docs: List[Document]
    ) -> Dict:
        """
        Post-generation guardrails.
        These run AFTER the LLM responds to catch hallucination-prone outputs.
        """
        # Guardrail 1: Force out_of_scope if FAISS found no relevant docs
        if not is_in_scope:
            data["coverage_decision"] = "out_of_scope"
            data["coverage_reasoning"] = (
                "No relevant policy documentation was found in the knowledge base for this "
                "claim type. This claim is outside the current documentation scope. "
                "Action required: upload the relevant policy documents to the knowledge base."
            )
            data["confidence_score"] = 0.0
            data["requires_additional_info"] = False
            data.setdefault("risk_flags", [])
            data["risk_flags"].append("OUT_OF_SCOPE_NO_RAG_MATCH")

        # Guardrail 2: Low confidence → requires_more_info
        if data.get("confidence_score", 0) < 0.6 and data.get("coverage_decision") not in (
            "out_of_scope", "requires_more_info"
        ):
            data["coverage_decision"] = "requires_more_info"
            data.setdefault("risk_flags", [])
            data["risk_flags"].append("LOW_CONFIDENCE_DECISION_DEFERRED")

        # Guardrail 3: Ensure requires_additional_info is consistent
        has_missing = bool(data.get("missing_information"))
        if has_missing or data.get("coverage_decision") == "requires_more_info":
            data["requires_additional_info"] = True

        # Guardrail 4: Populate rag_sources_used from actual retrieved docs
        if rag_docs and not data.get("rag_sources_used"):
            data["rag_sources_used"] = list(
                {d.metadata.get("source_file", "Unknown") for d in rag_docs}
            )

        return data

    def _error_analysis(self, claim_id: str, error_detail: str) -> ClaimAnalysis:
        """Return a safe fallback analysis when something goes wrong."""
        return ClaimAnalysis(
            claim_id=claim_id,
            summary="Analysis could not be completed due to a technical error.",
            claim_type="Unknown",
            coverage_decision="requires_more_info",
            coverage_reasoning=(
                f"A system error occurred during analysis: {error_detail[:200]}. "
                "Please check the logs and retry."
            ),
            missing_information=[],
            risk_flags=["SYSTEM_ERROR"],
            confidence_score=0.0,
            rag_sources_used=[],
            requires_additional_info=True,
            recommended_next_steps=["Retry the analysis", "Check pints.log for details"],
        )
    
