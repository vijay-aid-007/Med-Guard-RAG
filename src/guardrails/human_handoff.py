# import re
# from src.core.config import settings
# from src.core.logging_config import logger

# NEGATIVE = [
#     r"\bwrong\b", r"\bnot helpful\b", r"\bbad answer\b",
#     r"\buseless\b", r"\bdoesn'?t (help|make sense)\b",
#     r"\bstill (confused|don'?t understand)\b",
# ]
# POSITIVE = [
#     r"\bthanks?\b", r"\bthat helps?\b", r"\bgot it\b",
#     r"\bmakes sense\b", r"\bperfect\b", r"\bgreat\b",
# ]
# HANDOFF_REQUEST = [
#     r"\btalk to (a )?human\b",
#     r"\bconnect me to (an? )?agent\b",
#     r"\bspeak (to|with) (a )?(doctor|person)\b",
#     r"\breal (doctor|person)\b",
# ]


# class SatisfactionTracker:

#     def score_turn(self, user_message: str) -> int:
#         low = user_message.lower()
#         if any(re.search(p, low) for p in NEGATIVE):
#             return -1
#         if any(re.search(p, low) for p in POSITIVE):
#             return +1
#         return 0

#     def is_explicit_handoff_request(self, user_message: str) -> bool:
#         low = user_message.lower()
#         return any(re.search(p, low) for p in HANDOFF_REQUEST)

#     def should_handoff(
#         self,
#         frustration_score: float,
#         repeated_question_count: int,
#         explicit_request: bool,
#     ) -> dict:
#         if explicit_request:
#             logger.info("Handoff — explicit request")
#             return {"handoff": True, "trigger": "explicit_request"}

#         if repeated_question_count >= settings.repetition_handoff_count:
#             logger.info(f"Handoff — repetition ({repeated_question_count}x)")
#             return {"handoff": True, "trigger": "repetition"}

#         if frustration_score <= -settings.frustration_block_threshold:
#             logger.info(f"Handoff — frustration ({frustration_score})")
#             return {"handoff": True, "trigger": "frustration_threshold"}

#         return {"handoff": False, "trigger": None}


# _instance: SatisfactionTracker | None = None


# def get_satisfaction_tracker() -> SatisfactionTracker:
#     global _instance
#     if _instance is None:
#         _instance = SatisfactionTracker()
#     return _instance 





"""
One sentence job: Tracks user satisfaction and routes frustrated or
explicitly requesting users to human support via email, Slack webhook,
and PostgreSQL audit log.

Production channels:
- Email (SMTP)      → formal handoff notification
- Slack webhook     → instant ops team alert
- PostgreSQL        → audit trail + agent dashboard
- All three fire on handoff — ops sees it everywhere

Pattern files loaded from data/handoff_patterns.yaml — ops-editable.
"""

import json
import re
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import yaml

from src.core.config import settings
from src.core.logging_config import logger


# ── Pattern loading ───────────────────────────────────────────────────────
_PATTERNS_PATH = Path("data/handoff_patterns.yaml")


def _load_patterns(key: str, fallback: list[str]) -> list[str]:
    if not _PATTERNS_PATH.exists():
        return fallback
    with open(_PATTERNS_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return data.get(key, fallback)


_DEFAULT_NEGATIVE = [
    r"\bwrong\b", r"\bnot helpful\b", r"\bbad answer\b",
    r"\buseless\b", r"\bdoesn'?t (help|make sense)\b",
    r"\bstill (confused|don'?t understand)\b",
    r"\bthat('s| is) (incorrect|wrong|not right)\b",
    r"\byou (don'?t|didn'?t) (understand|answer)\b",
    r"\bterrible (answer|response)\b",
    r"\bcompletely (wrong|incorrect|off)\b",
]

_DEFAULT_POSITIVE = [
    r"\bthanks?\b", r"\bthat helps?\b", r"\bgot it\b",
    r"\bmakes sense\b", r"\bperfect\b", r"\bgreat\b",
    r"\bexcellent\b", r"\bthank you\b", r"\bvery helpful\b",
    r"\bappreciate (it|that|this)\b", r"\bwonderful\b",
]

_DEFAULT_HANDOFF_REQUEST = [
    r"\btalk to (a )?human\b",
    r"\bconnect me to (an? )?agent\b",
    r"\bspeak (to|with) (a )?(doctor|person|human|agent|representative)\b",
    r"\breal (doctor|person|human|agent)\b",
    r"\bhuman (support|agent|representative|help)\b",
    r"\bi (want|need|would like) (to speak|to talk) (with|to) (a |)(human|person|doctor|agent)\b",
    r"\bescalate (this|my (issue|case|question))\b",
    r"\b(get|speak to|transfer to) (a )?supervisor\b",
    r"\bi (want|need) (a |)(real|actual|human) (doctor|physician|medical professional)\b",
]


class HandoffNotifier:
    """
    Handles all notification channels when handoff is triggered.
    Each channel is independent — failure in one doesn't block others.
    """

    def notify(
        self,
        session_id: str,
        query: str,
        trigger: str,
        frustration_score: float,
        conversation_history: list[dict] | None = None,
    ) -> None:
        context = {
            "session_id":   session_id,
            "query":        query,
            "trigger":      trigger,
            "frustration":  frustration_score,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "history_len":  len(conversation_history or []),
        }

        # Fire all channels — log failures but don't crash
        self._notify_slack(context)
        self._notify_email(context)
        self._log_to_db(context, conversation_history or [])

    def _notify_slack(self, ctx: dict) -> None:
        webhook_url = getattr(settings, "slack_webhook_url", "")
        if not webhook_url:
            logger.debug("Slack webhook not configured — skipping")
            return
        try:
            emoji = {
                "explicit_request":    "🙋",
                "repetition":          "🔄",
                "frustration_threshold": "😤",
            }.get(ctx["trigger"], "⚠️")

            payload = {
                "text": f"{emoji} *MedGuard Handoff Required*",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"{emoji} *MedGuard Handoff Required*\n"
                                f"*Trigger:* {ctx['trigger']}\n"
                                f"*Session:* `{ctx['session_id']}`\n"
                                f"*Query:* {ctx['query'][:200]}\n"
                                f"*Frustration score:* {ctx['frustration']}\n"
                                f"*Time:* {ctx['timestamp']}"
                            ),
                        },
                    }
                ],
            }
            data    = json.dumps(payload).encode("utf-8")
            req     = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info("Handoff notification sent to Slack")
        except Exception as e:
            logger.warning(f"Slack notification failed: {e}")

    def _notify_email(self, ctx: dict) -> None:
        smtp_host = getattr(settings, "smtp_host", "")
        smtp_user = getattr(settings, "smtp_user", "")
        smtp_pass = getattr(settings, "smtp_password", "")
        to_email  = getattr(settings, "handoff_email", "")

        if not all([smtp_host, smtp_user, smtp_pass, to_email]):
            logger.debug("Email not configured — skipping")
            return

        try:
            body = (
                f"MedGuard Handoff Alert\n"
                f"{'='*40}\n"
                f"Trigger    : {ctx['trigger']}\n"
                f"Session    : {ctx['session_id']}\n"
                f"Query      : {ctx['query']}\n"
                f"Frustration: {ctx['frustration']}\n"
                f"Time       : {ctx['timestamp']}\n"
            )
            msg              = MIMEText(body)
            msg["Subject"]   = f"[MedGuard] Handoff Required — {ctx['trigger']}"
            msg["From"]      = smtp_user
            msg["To"]        = to_email

            with smtplib.SMTP_SSL(smtp_host, 465) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)

            logger.info(f"Handoff email sent to {to_email}")
        except Exception as e:
            logger.warning(f"Email notification failed: {e}")

    def _log_to_db(self, ctx: dict, history: list[dict]) -> None:
        db_url = getattr(settings, "postgres_url", "")
        if not db_url or "localhost" in db_url:
            logger.debug("PostgreSQL not available — skipping DB log")
            return
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(db_url)
            with engine.begin() as conn:
                conn.execute(
                    sqlalchemy.text("""
                        INSERT INTO handoff_queue
                            (session_id, query, trigger_reason,
                             frustration_score, conversation_history, status)
                        VALUES
                            (:session_id, :query, :trigger,
                             :frustration, :history::jsonb, 'pending')
                    """),
                    {
                        "session_id":  ctx["session_id"],
                        "query":       ctx["query"],
                        "trigger":     ctx["trigger"],
                        "frustration": ctx["frustration"],
                        "history":     json.dumps(history),
                    },
                )
            logger.info("Handoff logged to PostgreSQL")
        except Exception as e:
            logger.warning(f"DB handoff log failed: {e}")


class SatisfactionTracker:

    def __init__(self):
        self._negative_patterns = _load_patterns("negative", _DEFAULT_NEGATIVE)
        self._positive_patterns = _load_patterns("positive", _DEFAULT_POSITIVE)
        self._handoff_patterns  = _load_patterns("handoff_request", _DEFAULT_HANDOFF_REQUEST)
        self._notifier          = HandoffNotifier()
        logger.info("SatisfactionTracker ready")

    def score_turn(self, user_message: str) -> int:
        low = user_message.lower()
        if any(re.search(p, low) for p in self._negative_patterns):
            return -1
        if any(re.search(p, low) for p in self._positive_patterns):
            return +1
        return 0

    def is_explicit_handoff_request(self, user_message: str) -> bool:
        low = user_message.lower()
        return any(re.search(p, low) for p in self._handoff_patterns)

    def should_handoff(
        self,
        frustration_score: float,
        repeated_question_count: int,
        explicit_request: bool,
        session_id: str = "unknown",
        query: str = "",
        conversation_history: list[dict] | None = None,
    ) -> dict:

        trigger = None

        if explicit_request:
            trigger = "explicit_request"
        elif repeated_question_count >= settings.repetition_handoff_count:
            trigger = "repetition"
        elif frustration_score <= -settings.frustration_block_threshold:
            trigger = "frustration_threshold"

        if trigger:
            logger.info(f"Handoff triggered — {trigger} (score={frustration_score})")
            self._notifier.notify(
                session_id          = session_id,
                query               = query,
                trigger             = trigger,
                frustration_score   = frustration_score,
                conversation_history= conversation_history,
            )
            return {"handoff": True, "trigger": trigger}

        return {"handoff": False, "trigger": None}


_instance: SatisfactionTracker | None = None


def get_satisfaction_tracker() -> SatisfactionTracker:
    global _instance
    if _instance is None:
        _instance = SatisfactionTracker()
    return _instance