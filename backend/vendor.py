import os
import uuid
import time
import random
import hashlib
import json
import logging
from typing import Dict, Any

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


def extract_vendor_data_ocr(file_content: bytes) -> Dict[str, Any]:
    """
    Extract vendor/certificate data using Azure Document Intelligence
    prebuilt-read (OCR), then structure with GPT.
    """
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not key:
        raise ValueError("Azure Document Intelligence credentials not configured")

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
        api_version="2024-11-30"
    )

    # Use prebuilt-read for OCR text extraction
    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body=file_content,
        content_type="application/octet-stream",
        pages="1-10"
    )

    result = poller.result()

    # Extract full text content
    full_text = result.content if hasattr(result, "content") else ""

    logger.info(f"Extracted text length: {len(full_text)} characters")

    if not full_text:
        raise ValueError(
            "No text could be extracted from document. "
            "It may be corrupted, image-only, or not a valid document."
        )

    structured_data = enhance_with_gpt(full_text)
    return structured_data


def enhance_with_gpt(full_text: str) -> Dict[str, Any]:
    """
    Use GPT to extract and structure vendor/certificate information
    from OCR text.
    """
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

    if not openai_endpoint or not openai_key:
        raise ValueError(
            "Azure OpenAI credentials not configured. Please set "
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables."
        )

    client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_key=openai_key,
        api_version="2024-02-15-preview"
    )

    # Cache-breaking identifiers
    extraction_id = str(uuid.uuid4())
    random_seed = random.randint(100000, 999999)
    doc_hash = hashlib.sha256(full_text.encode()).hexdigest()[:16]

    time.sleep(0.5)

    cache_breaker = random.choice([
        "You must extract data with precision.",
        "Your task is to extract vendor certificate information.",
        "Extract the certificate data accurately.",
        "Parse the following vendor document carefully."
    ])

    prompt = f"""
Extract all data from this ACORD Certificate of Liability Insurance document.

OCR TEXT:
{full_text}

INSTRUCTIONS:

1) HEADER
- certificate_date: Date shown at top right of the certificate in YYYY-MM-DD format. Return null if not present.
- certificate_number: Certificate number. Return null if not present.
- revision_number: Revision number. Return null if not present.

2) PRODUCER
- producer_name: Name of the producer/agency. Return null if not present.
- producer_address: Full address of the producer. Return null if not present.

3) INSURED
- insured_name: Full name(s) of the insured. If multiple names, separate with comma. Return null if not present.
- insured_address: Full address of the insured. Return null if not present.

4) INSURERS
- insurer_a: Insurer A name and NAIC number as a single string (e.g. "Liberty Mutual Fire Insurance Company, NAIC: 23035"). Return null if not present.
- insurer_b: Insurer B name and NAIC number as a single string. Return null if not present.
- insurer_c: Insurer C name and NAIC number as a single string. Return null if not present.
- insurer_d: Insurer D name and NAIC number as a single string. Return null if not present.
- insurer_e: Insurer E name and NAIC number as a single string. Return null if not present.
- insurer_f: Insurer F name and NAIC number as a single string. Return null if not present.

5) POLICIES
- policies: Array of policy objects. Extract ALL policy rows from the coverages table.
  Each policy object must have:
  - insurer_letter: The insurer letter (A, B, C, etc.) for this policy row. Return null if not present.
  - policy_type: Type of insurance (e.g. "Commercial General Liability", "Automobile Liability", "Workers Compensation and Employers Liability", "Excess Workers Compensation"). Return null if not present.
    - If the policy type is Umbrella Liability, Excess Liability, or any combination of both, ALWAYS return "Umbrella Liability/Excess Liability".
  - policy_number: Policy number. Return null if not present.
  - policy_effective_date: Policy effective date in YYYY-MM-DD format. Return null if not present.
  - policy_expiration_date: Policy expiration date in YYYY-MM-DD format. Return null if not present.
  - limits: Single string containing ALL limits for this policy row, formatted as "Limit Type=$amount, Limit Type=$amount". Return null if not present.

6) DESCRIPTION OF OPERATIONS
- description_of_operations: Full text from the "Description of Operations / Locations / Vehicles" section. Return null if empty or not present.

7) CERTIFICATE HOLDER
- certificate_holder_name: Name of the certificate holder. Return null if not present.
- certificate_holder_address: Full address of the certificate holder. Return null if not present.

8) ADDITIONAL REMARKS
- additional_remarks: Full text from the Additional Remarks Schedule page if present. Return null if not present.

GENERAL RULES:
- Extract ALL policies found — do not skip any rows.
- Do NOT invent data that is not present in the document.
- Dates must be in YYYY-MM-DD format.
- Return null for any field that cannot be confidently determined.

Return ONLY a valid JSON object with these exact keys:
- certificate_date
- certificate_number
- revision_number
- producer_name
- producer_address
- insured_name
- insured_address
- insurer_a
- insurer_b
- insurer_c
- insurer_d
- insurer_e
- insurer_f
- policies
- description_of_operations
- certificate_holder_name
- certificate_holder_address
- additional_remarks

Return just the JSON, no explanation text.
"""

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{cache_breaker} SESSION ID: {extraction_id} "
                    f"DOCUMENT HASH: {doc_hash} "
                    "You are a precise insurance certificate data extractor. "
                    "Return only valid JSON."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=4000,
        seed=random_seed,
        user=f"extraction_{extraction_id}"
    )

    result_text = response.choices[0].message.content.strip()
    result_text = result_text.replace("```json", "").replace("```", "").strip()

    structured_data = json.loads(result_text)
    return structured_data