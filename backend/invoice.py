import os
import re
import json
import uuid
import time
import random
import hashlib
from typing import List, Dict, Any, Optional

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI


def extract_text_with_ocr(file_content: bytes) -> str:
    """
    Extract all text from document using Azure Read OCR
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

    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body=file_content,
        content_type="application/octet-stream",
    )

    result = poller.result()

    if not result or not hasattr(result, 'content'):
        raise ValueError("No text extracted from document")

    return result.content


def parse_invoice_with_gpt(ocr_text: str) -> List[Dict[str, Any]]:
    """
    Parse extracted text using GPT-4o via Azure OpenAI
    Returns list of invoices (supports multiple invoices per document)
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    
    if not azure_endpoint or not azure_api_key:
        raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY not configured")
    
    # Generate unique identifiers for cache breaking
    extraction_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000000)
    random_seed = random.randint(100000, 999999)
    doc_hash = hashlib.sha256(ocr_text.encode()).hexdigest()[:16]
    
    time.sleep(0.5)
    
    client = AzureOpenAI(
        api_key=azure_api_key,
        api_version=azure_api_version,
        azure_endpoint=azure_endpoint,
        timeout=60.0,
        max_retries=0
    )
    
    cache_breaker = random.choice([
        "You must extract data with precision.",
        "Your task is to extract invoice information.",
        "Extract the invoice data accurately.",
        "Parse the following invoice carefully."
    ])
    
    system_prompt = f"""{cache_breaker}

SESSION ID: {extraction_id}
DOCUMENT HASH: {doc_hash}

This is a brand new extraction. You have NO knowledge of previous documents. Extract ONLY from the text provided below. Return valid JSON array only."""

    user_prompt = f"""Extract ALL invoice data from this text and return a JSON array.

Required JSON structure - ALWAYS return an array, even for single invoice:
[
  {{
    "invoice_id": "string or null",
    "invoice_date": "MM-DD-YYYY or null",
    "vendor_name": "string or null",
    "vendor_address": "string or null",
    "vendor_phone_number": "string or null",
    "purchase_order": "string or null",
    "customer_name": "string or null",
    "customer_address": "string or null",
    "sub_total": number or null,
    "tax": number or null,
    "total": number or null,
    "line_items": [
      {{
        "line_number": number,
        "description": "string",
        "apartment_units": ["unit numbers array"],
        "quantity": number or null,
        "unit_price": number or null,
        "amount": number or null
      }}
    ]
  }}
]

Critical Rules:
- ALWAYS return a JSON array (use [] brackets)
- Extract ALL invoices found in the document
- Each invoice is a separate object in the array
- Extract ALL line items from each invoice
- Look for unit/apartment numbers in headers OR in line item rows
- If unit is in header, apply to all line items of that invoice
- If unit is in a row, use it for that line only
- Remove # symbol from unit numbers
- Convert dates to MM-DD-YYYY format
- Extract amounts as numbers only (no $ or commas)
- If field is missing or unreadable, use null
- NEVER mix data from different invoices
- Works with any language (English, Spanish, French, etc.)

Document text:
{ocr_text}

Return ONLY the JSON array, no markdown, no explanation.
"""
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            seed=random_seed,
            user=f"extraction_{extraction_id}"
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        result_text = re.sub(r'```json\s*', '', result_text)
        result_text = re.sub(r'```\s*', '', result_text)
        result_text = result_text.strip()
        
        # Parse JSON
        invoice_data = json.loads(result_text)
        
        # Handle both array and single object responses
        if isinstance(invoice_data, dict):
            # Single invoice returned as object, wrap in array
            invoice_data = [invoice_data]
        
        # Validate it's a list
        if not isinstance(invoice_data, list):
            raise ValueError("Response must be a JSON array")
        
        # Validate each invoice against source text
        validated_invoices = []
        for inv in invoice_data:
            validated_inv = validate_data_against_source(inv, ocr_text, doc_hash)
            validated_invoices.append(validated_inv)
        
        return validated_invoices
        
    except json.JSONDecodeError as e:
        raise ValueError(f"GPT returned invalid JSON: {e}")
    except Exception as e:
        raise ValueError(f"GPT parsing error: {e}")


def validate_data_against_source(invoice_data: Dict[str, Any], original_text: str, doc_hash: str) -> Dict[str, Any]:
    """
    Validate that extracted data exists in source text
    """
    text_lower = original_text.lower()
    
    # Validate vendor name
    if invoice_data.get('vendor_name'):
        vendor_name = invoice_data['vendor_name']
        vendor_words = [w for w in vendor_name.lower().split() if len(w) > 2]
        
        if vendor_words:
            found_count = sum(1 for word in vendor_words if word in text_lower)
            match_ratio = found_count / len(vendor_words)
            
            if match_ratio < 0.7:
                invoice_data['vendor_name'] = None
    
    # Validate customer name
    if invoice_data.get('customer_name'):
        customer_name = invoice_data['customer_name']
        customer_words = [w for w in customer_name.lower().split() if len(w) > 2]
        
        if customer_words:
            found_count = sum(1 for word in customer_words if word in text_lower)
            match_ratio = found_count / len(customer_words)
            
            if match_ratio < 0.7:
                invoice_data['customer_name'] = None
    
    # Validate phone number
    if invoice_data.get('vendor_phone_number'):
        phone = re.sub(r'\D', '', str(invoice_data['vendor_phone_number']))
        text_digits = re.sub(r'\D', '', original_text)
        
        if phone not in text_digits:
            invoice_data['vendor_phone_number'] = None
    
    # Validate apartment units
    for item in invoice_data.get('line_items', []):
        units = item.get('apartment_units', [])
        if units:
            verified_units = []
            for unit in units:
                unit_normalized = str(unit).upper().strip()
                unit_digits = re.sub(r'\D', '', unit_normalized)
                
                found = False
                search_patterns = [
                    unit_normalized,
                    unit_digits,
                    f"#{unit_normalized}",
                    f"unit {unit_normalized}",
                    f"apt {unit_normalized}"
                ]
                
                for pattern in search_patterns:
                    if pattern.lower() in text_lower:
                        found = True
                        break
                
                if found:
                    verified_units.append(unit)
            
            item['apartment_units'] = verified_units
    
    # Validate invoice_id
    if invoice_data.get('invoice_id'):
        inv_id = str(invoice_data['invoice_id'])
        inv_id_normalized = re.sub(r'\s+', '', inv_id)
        
        if inv_id_normalized.lower() not in re.sub(r'\s+', '', text_lower):
            inv_id_parts = re.findall(r'\w+', inv_id)
            match_count = sum(1 for part in inv_id_parts if len(part) > 2 and part.lower() in text_lower)
            
            if match_count == 0:
                invoice_data['invoice_id'] = None
    
    return invoice_data


def clean_newlines(text: Optional[str]) -> Optional[str]:
    """
    Remove newlines and extra whitespace from text
    """
    if not text:
        return text
    cleaned = re.sub(r'\s+', ' ', str(text).replace('\n', ' ').replace('\r', ' '))
    return cleaned.strip()


def validate_and_clean_invoice(invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean the GPT-parsed invoice data
    """
    # Clean text fields
    if invoice_data.get('vendor_name'):
        invoice_data['vendor_name'] = clean_newlines(invoice_data['vendor_name'])
    
    if invoice_data.get('customer_name'):
        invoice_data['customer_name'] = clean_newlines(invoice_data['customer_name'])
    
    if invoice_data.get('vendor_address'):
        invoice_data['vendor_address'] = clean_newlines(invoice_data['vendor_address'])
    
    if invoice_data.get('customer_address'):
        invoice_data['customer_address'] = clean_newlines(invoice_data['customer_address'])
    
    # Ensure line_items is a list
    if not isinstance(invoice_data.get('line_items'), list):
        invoice_data['line_items'] = []
    
    # Clean line items
    for item in invoice_data['line_items']:
        if 'description' in item and item['description']:
            item['description'] = clean_newlines(item['description'])
        
        # Ensure apartment_units is a list
        if not isinstance(item.get('apartment_units'), list):
            item['apartment_units'] = []
        
        # Convert string apartment_units to list if needed
        if isinstance(item.get('apartment_units'), str):
            item['apartment_units'] = [item['apartment_units']]
    
    # Convert amounts to float if they're strings
    for field in ['sub_total', 'tax', 'total']:
        if invoice_data.get(field) and isinstance(invoice_data[field], str):
            try:
                invoice_data[field] = float(re.sub(r'[^\d.]', '', invoice_data[field]))
            except:
                invoice_data[field] = None
    
    for item in invoice_data.get('line_items', []):
        for field in ['quantity', 'unit_price', 'amount']:
            if item.get(field) and isinstance(item[field], str):
                try:
                    item[field] = float(re.sub(r'[^\d.]', '', item[field]))
                except:
                    item[field] = None
    
    return invoice_data


def extract_invoice_data(file_content: bytes, pages: str = "1-10") -> List[Dict[str, Any]]:
    """
    Extract invoice data using Azure Read OCR + GPT-4o
    
    Supports multiple invoices per document
    Returns list of invoice dictionaries
    """
    
    ocr_text = extract_text_with_ocr(file_content)
    
    if not ocr_text or len(ocr_text.strip()) < 50:
        raise ValueError("Insufficient text extracted from document")
    
    # Get list of invoices (handles single or multiple)
    invoice_list = parse_invoice_with_gpt(ocr_text)
    
    # Validate and clean each invoice
    cleaned_invoices = []
    for invoice_data in invoice_list:
        cleaned = validate_and_clean_invoice(invoice_data)
        cleaned_invoices.append(cleaned)
    
    return cleaned_invoices