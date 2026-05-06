"""
Invoice data extraction module.
1. Extract text from invoice using Azure Document Intelligence OCR
2. Use Azure OpenAI to parse ALL line items flat (one object per line item with invoice/property context)
3. Python groups line items by invoice_number + property_name
4. Return structured invoice data for each file
"""

import os
import re
import json
import uuid
import time
import random
import hashlib
import logging
from collections import OrderedDict
from typing import List, Dict, Any, Optional

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ==================== OCR ====================

def extract_text_with_ocr(file_content: bytes) -> str:
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not key:
        raise ValueError("Azure Document Intelligence credentials not configured")

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


# ==================== GPT EXTRACTION ====================

def extract_flat_invoices(ocr_text: str) -> Dict[str, Any]:
    """
    Single GPT call — flat extraction of ALL line items across ALL invoices.
    Each line item carries its invoice context (invoice_number, property_name, etc.)
    Python then groups them.
    """
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not openai_endpoint or not openai_key:
        raise ValueError("Azure OpenAI credentials not configured")

    client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_key=openai_key,
        api_version=api_version,
    )

    # Cache-breaking identifiers
    extraction_id = str(uuid.uuid4())
    random_seed = random.randint(100000, 999999)
    doc_hash = hashlib.sha256(ocr_text.encode()).hexdigest()[:16]

    time.sleep(0.5)

    cache_breaker = random.choice([
        "You must extract data with precision.",
        "Your task is to extract invoice information.",
        "Extract the invoice data accurately.",
        "Parse the following invoice carefully."
    ])

    prompt = f"""Extract ALL invoice data from this document and return a single JSON object.

The document may contain:
- A single invoice with multiple properties/locations (same invoice number, different Bill To per page)
- Multiple separate invoices (different invoice numbers, different Bill To per page)
- A mix of both

Required JSON structure:
{{
  "company_name": "the top-level parent company or organization being billed, or null if not present. Do NOT use the property/location name here.",
  "vendor_name": "the vendor/supplier name, or null",
  "invoice_number": "primary invoice number, or null",
  "invoice_date": "ISO 8601 format e.g. 2026-03-26T00:00:00 or null",
  "notes": "concise important notes only (max two sentences), or null",
  "line_items": [
    {{
      "invoice_number": "invoice number this line item belongs to (use page-level invoice number if different invoices per page)",
      "invoice_date": "ISO 8601 date for this specific invoice, or null",
      "property_name": "the Bill To property, site, or location name this line item belongs to, or null",
      "description": "line item description",
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
- Extract ALL line items from ALL pages without exception
- For each line item, set invoice_number and property_name from the page it appears on
- If all pages share the same invoice number, use that invoice number for all line items
- If pages have different invoice numbers, use each page's invoice number per line item
- property_name should be the Bill To name on that page (e.g. "Yorktown", "Windrush")
- company_name is the parent organization above property level — if not explicitly stated, return null
- All numeric values must be plain numbers — no $ signs or commas
- Dates must be ISO 8601 format
- Return ONLY the JSON object, no markdown, no explanation

Invoice text:
{ocr_text}"""

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{cache_breaker} SESSION ID: {extraction_id} "
                    f"DOCUMENT HASH: {doc_hash} "
                    "You are a precise invoice data extractor. Return only valid JSON."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=32000,
        seed=random_seed,
        user=f"extraction_{extraction_id}"
    )

    result_text = response.choices[0].message.content.strip()
    result_text = re.sub(r"```json\s*", "", result_text)
    result_text = re.sub(r"```\s*", "", result_text)
    result_text = result_text.strip()

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        # Remove trailing commas before ] or }
        result_text = re.sub(r",\s*([}\]])", r"\1", result_text)
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"GPT returned invalid JSON: {e}\n"
                f"Raw response (first 500 chars): {result_text[:500]}"
            )

    return result


# ==================== GROUPING ====================

def group_by_invoice_and_property(flat_result: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Group flat line items by (invoice_number, property_name) to build the properties list.
    Each unique (invoice_number, property_name) combination becomes one property entry.
    Preserves the order properties appear in the document.
    """
    line_items = flat_result.get("line_items", [])

    # Use OrderedDict to preserve document order
    groups: OrderedDict = OrderedDict()

    for item in line_items:
        inv_num = item.get("invoice_number") or flat_result.get("invoice_number") or "__NO_INV__"
        prop_name = item.get("property_name") or "__NO_PROPERTY__"
        inv_date = item.get("invoice_date") or flat_result.get("invoice_date")

        key = (inv_num, prop_name)
        if key not in groups:
            groups[key] = {
                "invoice_number": inv_num if inv_num != "__NO_INV__" else None,
                "invoice_date": inv_date,
                "property_name": prop_name if prop_name != "__NO_PROPERTY__" else None,
                "items": [],
            }
        groups[key]["items"].append(item)

    properties = []
    for group in groups.values():
        invoice_items = []
        for item in group["items"]:
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
            "propertyName": group["property_name"],
            "invoiceNumber": group["invoice_number"],
            "invoiceDate": group["invoice_date"],
            "notes": flat_result.get("notes"),
            "invoiceItems": invoice_items,
        })

    return {
        "companyName": flat_result.get("company_name") or None,
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
      1. OCR the document
      2. GPT flat extraction of ALL line items with invoice/property context
      3. Python groups by (invoice_number, property_name)
      4. Return structured result — IDs assigned by the caller in main.py
    """
    # Step 1 — OCR
    ocr_text = extract_text_with_ocr(file_content)
    if not ocr_text or len(ocr_text.strip()) < 50:
        raise ValueError("Insufficient text extracted from document")

    # Step 2 — GPT flat extraction
    try:
        flat_result = extract_flat_invoices(ocr_text)
    except Exception as e:
        raise ValueError(f"Failed to extract invoice data: {e}")

    if not flat_result:
        raise ValueError("No invoice data extracted from document")

    # Step 3 — Group by invoice + property in Python
    result = group_by_invoice_and_property(flat_result, filename)

    logger.info(
        f"Extraction complete — properties: {len(result['properties'])} | file: {filename}"
    )
    # print(json.dumps(result, indent=2))

    return result