"""
Invoice data extraction module.
1. Extract text from invoice using Azure Document Intelligence OCR
2. Use Azure OpenAI to parse invoice data into a structured format (flat extraction approach)
3. Group line items by property/site/location in Python to build the final schema
4. Return structured invoice data for each file

Reliability features:
- Exponential backoff with jitter on OCR and GPT failures
- Correlation ID tracking per extraction request
- Circuit breaker to fail fast when Azure services are down
- max_retries=2 on OpenAI client for transient network errors
- Bulk-aware settings: semaphore in main.py limits parallel GPT calls
"""

import os
import re
import json
import time
import uuid
import random
import logging
from collections import OrderedDict
from typing import List, Dict, Any, Optional

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI, APITimeoutError, APIConnectionError, RateLimitError, InternalServerError

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ==================== ENV VALIDATION ====================

def validate_env_vars():
    """
    Validate all required environment variables are set.
    Called at module import time so the app refuses to start
    with a clear error rather than failing on the first request.
    """
    required = [
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "AZURE_DOCUMENT_INTELLIGENCE_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


validate_env_vars()


# ==================== CIRCUIT BREAKER ====================

class CircuitBreaker:
    """
    Circuit breaker with three states:

    CLOSED   → Normal operation. All requests pass through.
    OPEN     → Too many failures. Requests fail immediately without
               calling the service — avoids hammering a down service.
    HALF_OPEN→ After cooldown, one test request is allowed through.
               If it succeeds → CLOSED. If it fails → back to OPEN.

    Bulk processing note:
    - OCR: threshold=5, cooldown=120s (OCR is fast, more tolerance)
    - GPT: threshold=3, cooldown=180s (each GPT call costs ~100s,
      3 failures = ~300s wasted — open the circuit sooner)
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, name: str, failure_threshold: int = 5, cooldown_seconds: int = 120):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = self.CLOSED

    def record_success(self):
        """Reset circuit after a successful call."""
        if self.state != self.CLOSED:
            logger.info(f"[CircuitBreaker:{self.name}] → CLOSED after successful call.")
        self.failure_count = 0
        self.state = self.CLOSED

    def record_failure(self):
        """Record a failure and open circuit if threshold reached."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            logger.warning(
                f"[CircuitBreaker:{self.name}] → OPEN after {self.failure_count} failures. "
                f"Cooldown: {self.cooldown_seconds}s"
            )

    def allow_request(self) -> bool:
        """Return True if the request should be allowed through."""
        if self.state == self.CLOSED:
            return True

        if self.state == self.OPEN:
            elapsed = time.time() - (self.last_failure_time or 0)
            if elapsed >= self.cooldown_seconds:
                self.state = self.HALF_OPEN
                logger.info(f"[CircuitBreaker:{self.name}] → HALF_OPEN. Testing service...")
                return True
            remaining = int(self.cooldown_seconds - elapsed)
            logger.warning(
                f"[CircuitBreaker:{self.name}] Still OPEN. "
                f"Retry in ~{remaining}s."
            )
            return False

        # HALF_OPEN — allow the one test request through
        return True

    def is_open(self) -> bool:
        """Returns True if circuit is blocking requests."""
        return not self.allow_request()


# One circuit breaker per Azure service — module-level singletons
# shared across all concurrent requests in the process
_ocr_circuit = CircuitBreaker(name="AzureOCR", failure_threshold=5, cooldown_seconds=120)
_gpt_circuit = CircuitBreaker(name="AzureGPT", failure_threshold=3, cooldown_seconds=180)


# ==================== RETRY WITH EXPONENTIAL BACKOFF + JITTER ====================

# Transient exceptions worth retrying
_RETRYABLE_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


def retry_with_backoff(
    func,
    max_attempts: int = 3,
    base_delay: float = 10.0,
    correlation_id: str = "",
):
    """
    Retry a callable with exponential backoff and jitter.

    Delay formula: base_delay * 2^(attempt-1) ± 30% jitter
    e.g. base_delay=10s → delays of ~10s, ~20s, ~40s (with jitter)

    Jitter is critical for bulk processing — without it, all retrying
    files hit Azure at exactly the same time causing cascading rate limits.

    Only retries on transient errors (timeout, connection, rate limit,
    server error). Non-retryable errors (auth, bad JSON) raise immediately.
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except _RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt == max_attempts:
                logger.error(
                    f"[{correlation_id}] All {max_attempts} attempts failed. "
                    f"Last error: {type(e).__name__}: {e}"
                )
                break

            # Exponential backoff with ±30% jitter
            delay = base_delay * (2 ** (attempt - 1))
            jitter = delay * random.uniform(-0.3, 0.3)
            sleep_time = max(1.0, delay + jitter)

            logger.warning(
                f"[{correlation_id}] Attempt {attempt}/{max_attempts} failed "
                f"({type(e).__name__}). Retrying in {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)

        except Exception as e:
            # Non-retryable — raise immediately without retry
            logger.error(
                f"[{correlation_id}] Non-retryable error on attempt {attempt}: "
                f"{type(e).__name__}: {e}"
            )
            raise

    raise last_exception


# ==================== OCR ====================

def extract_text_with_ocr(file_content: bytes, correlation_id: str = "") -> str:
    """
    Extract all text from document using Azure Document Intelligence (prebuilt-read).

    Reliability:
    - Circuit breaker: fails fast if OCR is repeatedly down
    - Retry: up to 3 attempts with exponential backoff + jitter
    - base_delay=5s (OCR is fast ~5s, short delay is fine)
    """
    if _ocr_circuit.is_open():
        raise RuntimeError(
            f"[{correlation_id}] Azure OCR service is unavailable (circuit OPEN). "
            "Try again later."
        )

    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    def _call():
        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
            api_version="2024-11-30",
        )
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=file_content,
            content_type="application/octet-stream",
        )
        result = poller.result()

        if not result or not hasattr(result, "content"):
            raise ValueError("No text extracted from document")

        return result.content

    try:
        ocr_text = retry_with_backoff(
            _call,
            max_attempts=3,
            base_delay=5.0,        # OCR is fast, 5s base is enough
            correlation_id=correlation_id,
        )
        _ocr_circuit.record_success()
        logger.info(f"[{correlation_id}] OCR succeeded. Text length: {len(ocr_text)} chars")
        return ocr_text
    except Exception as e:
        _ocr_circuit.record_failure()
        raise


# ==================== OPENAI CLIENT ====================

def get_openai_client() -> AzureOpenAI:
    """
    Initialize Azure OpenAI client.

    timeout=300s: each GPT call on a large invoice can take ~100s.
    max_retries=0: SDK-level retry disabled, relying on our own retry_with_backoff.
    """
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        timeout=300.0,
        max_retries=0,
    )


def parse_gpt_json(text: str):
    """Strip markdown fences and parse JSON."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return json.loads(text.strip())


# ==================== GPT EXTRACTION ====================

def extract_flat_invoice(ocr_text: str, correlation_id: str = "") -> Dict[str, Any]:
    """
    Single GPT call — flat line item extraction.

    Reliability:
    - Circuit breaker: fails fast after 3 GPT failures
      (each failure costs ~100s, open the circuit early)
    - Retry: up to 3 attempts, base_delay=15s with ±30% jitter
      Worst case per file with 2 retries: ~100s + ~15s + ~100s + ~30s + ~100s = ~345s
    - Jitter: prevents all bulk-processing retries hitting Azure simultaneously
    """
    if _gpt_circuit.is_open():
        raise RuntimeError(
            f"[{correlation_id}] Azure OpenAI service is unavailable (circuit OPEN). "
            "Try again later."
        )

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    prompt = f"""Extract ALL invoice data from this text and return a JSON object.

Required JSON structure:
{{
  "invoice_number": "string or null",
  "invoice_date": "ISO 8601 format e.g. 2026-03-01T00:00:00 or null",
  "vendor_name": "string or null",
  "customer_name": "the bill-to / customer name or null",
  "notes": "concise important notes only (max two sentence), or null",
  "line_items": [
    {{
      "description": "line item description",
      "property_name": "property, site, location, department name this line item belongs to, or null if no grouping",
      "quantity": number or null,
      "unit_price": number or null,
      "tax": number or null,
      "overhead": number or null,
      "freight": number or null,
      "discount": number or null,
      "total_price": number or null
    }}
  ]
}}

Rules:
- Extract ALL line items without exception
- For each line item, identify which property/site/location/department it belongs to and set property_name
- If no grouping exists in the document, set property_name to null for all items
- All numeric values must be plain numbers — no $ signs or commas
- Dates must be ISO 8601 format
- Notes must be brief, include only important details, and be no more than two sentences
- Missing fields must be null
- Return ONLY the JSON object, no markdown, no explanation

Invoice text:
{ocr_text}"""

    def _call():
        client = get_openai_client()
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=32000,
        )

        # Log token usage for cost tracking and rate limit monitoring
        if response.usage:
            logger.info(
                f"[{correlation_id}] GPT token usage",
                extra={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )

        return parse_gpt_json(response.choices[0].message.content)

    try:
        result = retry_with_backoff(
            _call,
            max_attempts=3,
            base_delay=15.0,       # GPT takes ~100s, need longer recovery gaps
            correlation_id=correlation_id,
        )
        _gpt_circuit.record_success()
        logger.info(f"[{correlation_id}] GPT extraction succeeded.")
        return result
    except Exception as e:
        _gpt_circuit.record_failure()
        raise


# ==================== GROUPING ====================

def group_by_property(flat_result: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Group flat line items by property_name to build the final schema.
    Line items with no property_name are grouped under a single null-property invoice.
    Uses OrderedDict to preserve the order properties appear in the document.
    """
    line_items = flat_result.get("line_items", [])
    groups: OrderedDict = OrderedDict()

    for item in line_items:
        prop_name = item.get("property_name") or "__NO_PROPERTY__"
        if prop_name not in groups:
            groups[prop_name] = []
        groups[prop_name].append(item)

    properties = []
    for prop_name, items in groups.items():
        invoice_items = []
        for item in items:
            invoice_items.append({
                "description": item.get("description"),
                "quantity": item.get("quantity"),
                "price": item.get("unit_price"),
                "overhead": item.get("overhead"),
                "freight": item.get("freight"),
                "tax": item.get("tax"),
                "discount": item.get("discount"),
                "totalPrice": item.get("total_price"),
            })

        properties.append({
            "propertyName": None if prop_name == "__NO_PROPERTY__" else prop_name,
            "notes": flat_result.get("notes"),
            "invoiceItems": invoice_items,
        })

    return {
        "companyName": flat_result.get("customer_name"),
        "vendorName": flat_result.get("vendor_name"),
        "invoiceNumber": flat_result.get("invoice_number"),
        "invoiceDate": flat_result.get("invoice_date"),
        "uploadInvoice": filename,
        "properties": properties,
    }


# ==================== MAIN ENTRY POINT ====================

def extract_invoice_data(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    Main extraction function. Called once per file from main.py.

    Pipeline:
      1. Generate correlation ID for end-to-end log tracing
      2. OCR the document (retry + circuit breaker, base_delay=5s)
      3. GPT flat extraction (retry + circuit breaker, base_delay=15s + jitter)
      4. Group line items by property_name in Python (no extra GPT call)
      5. Return structured result — IDs assigned by the caller in main.py

    For bulk uploads, main.py uses asyncio.Semaphore(3) to limit
    concurrent GPT calls and avoid Azure rate limits.
    """
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(f"[{correlation_id}] Starting extraction: {filename}")

    # Step 1 — OCR
    ocr_text = extract_text_with_ocr(file_content, correlation_id)
    if not ocr_text or len(ocr_text.strip()) < 50:
        raise ValueError(
            f"[{correlation_id}] Insufficient text extracted from document"
        )

    # Step 2 — GPT flat extraction
    try:
        flat_result = extract_flat_invoice(ocr_text, correlation_id)
    except Exception as e:
        raise ValueError(f"[{correlation_id}] Failed to extract invoice data: {e}")

    # Step 3 — Group by property in Python
    result = group_by_property(flat_result, filename)
    logger.info(
        f"[{correlation_id}] Complete — "
        f"properties: {len(result['properties'])} | "
        f"file: {filename}"
    )
    return result