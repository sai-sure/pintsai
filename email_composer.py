"""
email_composer.py — Generates professional follow-up emails to customers.

Two output channels:
  1. In-app editable draft (displayed in the UI)
  2. mailto: link so the user can open directly in their mail client
  3. Download as .txt for use in any email client

Optional: uncomment the smtplib block and configure .env to send directly.
"""

import logging
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger("claimsense.email")

EMAIL_GENERATION_PROMPT = """\
You are a professional insurance claims correspondent. Your writing is clear, empathetic, and precise.

Draft a customer email requesting the missing information listed below.

Claim Reference: {claim_id}
Claim Summary: {claim_summary}
Customer Name: {customer_name}

Missing Information Required:
{missing_items_formatted}

Guidelines:
- Open with empathy — acknowledge the customer's situation
- Explain clearly (in plain English) why each piece of information is needed
- Group HIGH priority items first, MEDIUM second, LOW third
- Give a deadline of 7 business days
- State the consequence of not providing info (claim may be paused)
- Do NOT make any coverage promises or decisions
- Sign off professionally as "Claims Processing Team"
- Keep paragraphs short and scannable

Format your response EXACTLY as:
SUBJECT: <subject line here>
---
<full email body here>
"""


class EmailComposer:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.4,  # slightly higher for natural writing tone
            convert_system_message_to_human=True,
        )

    def compose_missing_info_email(
        self,
        claim_id: str,
        claim_summary: str,
        missing_items: List,  # list of MissingInfoItem or dicts
        customer_name: str = "Valued Customer",
    ) -> Dict:
        """
        Generate a professional email draft.
        Returns dict with keys: status, subject, body, claim_id
        """
        # Format missing items grouped by priority
        def get_priority(item):
            p = item.get("priority", "medium") if isinstance(item, dict) else getattr(item, "priority", "medium")
            return {"high": 0, "medium": 1, "low": 2}.get(p, 1)

        sorted_items = sorted(missing_items, key=get_priority)

        lines = []
        for item in sorted_items:
            if isinstance(item, dict):
                field = item.get("field", "N/A")
                reason = item.get("reason", "")
                priority = item.get("priority", "medium").upper()
            else:
                field = item.field
                reason = item.reason
                priority = item.priority.upper()

            priority_marker = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "•")
            lines.append(f"  {priority_marker} [{priority}] {field}: {reason}")

        missing_items_formatted = "\n".join(lines) if lines else "  (none specified)"

        prompt = EMAIL_GENERATION_PROMPT.format(
            claim_id=claim_id,
            claim_summary=claim_summary,
            customer_name=customer_name,
            missing_items_formatted=missing_items_formatted,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # Parse subject and body
            subject, body = self._parse_email_response(content, claim_id)

            return {
                "status": "success",
                "subject": subject,
                "body": body,
                "claim_id": claim_id,
            }

        except Exception as exc:
            logger.error(f"Email generation failed: {exc}")
            # Return a sensible fallback
            fallback_body = self._fallback_email(claim_id, customer_name, lines)
            return {
                "status": "fallback",
                "subject": f"Action Required: Additional Information Needed — Ref {claim_id}",
                "body": fallback_body,
                "claim_id": claim_id,
            }

    def _parse_email_response(self, content: str, claim_id: str):
        """Split SUBJECT / body from model output."""
        subject = f"Action Required: Additional Information — Ref {claim_id}"
        body = content

        if "SUBJECT:" in content:
            parts = content.split("---", 1)
            subject_line = parts[0].strip()
            subject = subject_line.replace("SUBJECT:", "").strip()
            body = parts[1].strip() if len(parts) > 1 else content
        elif "\n" in content:
            # Try first line as subject
            first_line, rest = content.split("\n", 1)
            if len(first_line) < 120:
                subject = first_line.strip()
                body = rest.strip()

        return subject, body

    def _fallback_email(self, claim_id: str, customer_name: str, item_lines: List[str]) -> str:
        items_block = "\n".join(item_lines) if item_lines else "  • Please contact us for details."
        return f"""Dear {customer_name},

Thank you for submitting your insurance claim (Reference: {claim_id}).

We are currently reviewing your claim and require the following additional information to continue processing:

{items_block}

Please provide the requested documents or information within 7 business days. Failure to do so may result in your claim being placed on hold.

If you have any questions, please do not hesitate to contact our claims team.

Kind regards,
Claims Processing Team"""

    # ---------------------------------------------------------------- #
    #  Optional: real SMTP send (configure .env first)                  #
    # ---------------------------------------------------------------- #

    def send_via_smtp(
        self,
        to_address: str,
        subject: str,
        body: str,
        from_name: str = "Claims Processing Team",
    ) -> Dict:
        """
        Send email via SMTP. Requires env vars:
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
        """
        smtp_host = os.getenv("SMTP_HOST", "")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")

        if not all([smtp_host, smtp_user, smtp_password]):
            return {
                "status": "error",
                "message": "SMTP not configured. Add SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD to .env",
            }

        try:
            msg = MIMEMultipart()
            msg["From"] = f"{from_name} <{smtp_user}>"
            msg["To"] = to_address
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent to {to_address} for claim (subject={subject[:50]})")
            return {"status": "sent", "to": to_address}

        except Exception as exc:
            logger.error(f"SMTP send failed: {exc}")
            return {"status": "error", "message": str(exc)}
        