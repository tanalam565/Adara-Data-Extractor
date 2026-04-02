import os
import re
import unicodedata
from typing import List, Dict, Any

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

def extract_city_from_address(address_field):
    """Extract city from Azure AI address field"""
    if not address_field:
        return None
    
    value = address_field.value if hasattr(address_field, 'value') else address_field
    
    if value is None or value == "":
        return None
    
    if hasattr(value, '__dict__'):
        if hasattr(value, 'city') and value.city:
            return str(value.city).strip()
    
    return None

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    # remove accents/diacritics
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text

def map_document_type(id_doc) -> str:
    """
    Map Azure AI document type to our schema.
    Prefer the top-level doc_type (idDocument.passport, etc),
    and fall back to the DocumentType field or the number pattern if needed.
    """
    # First, use doc_type: e.g. "idDocument.passport"
    raw_doc_type = getattr(id_doc, "doc_type", "") or ""
    doc_type = normalize_text(raw_doc_type)  # e.g. "iddocument.passport"

    # Social Security detection from doc_type 
    if any(k in doc_type for k in ("socialsecurity", "social_security", "ssn")):
        return "social_security"

    if "passport" in doc_type:
        return "passport"
    if "driverlicense" in doc_type or "driver_license" in doc_type:
        return "driver_license"
    if "nationalidentitycard" in doc_type or "national_identity_card" in doc_type:
        return "national_id"
    if "residencepermit" in doc_type or "residence_permit" in doc_type:
        return "residence_permit"
    if "visa" in doc_type:
        return "visa"

    # Fallback: use the DocumentType field inside fields (often "P" for passports)
    fields = getattr(id_doc, "fields", {}) or {}
    doc_type_field = fields.get("DocumentType")
    if doc_type_field and getattr(doc_type_field, "value", None):
        raw_field_value = str(doc_type_field.value).strip()
        norm_field_value = normalize_text(raw_field_value)

        # "P" means passport
        if raw_field_value.upper() == "P":
            return "passport"

        # Social Security detection from DocumentType text 
        social_keywords = [
            "social security",
            "seguro social",
            "ssn",
        ]
        if any(k in norm_field_value for k in social_keywords):
            return "social_security"

        passport_keywords = [
            "passport",     # EN
            "pasaporte",    # ES
            "passaporte",   # PT
            "pasaport",     # TR/RO
            "passaporto",   # IT
            "reisepass",    # DE
            "паспорт",      # RU/UK/etc.
        ]

        if any(x in norm_field_value for x in ["driver", "licence", "license", "dl"]):
            return "driver_license"
        elif any(x in norm_field_value for x in passport_keywords):
            return "passport"
        elif any(
            x in norm_field_value
            for x in ["residence", "residency", "permit", "green card", "permanent resident"]
        ):
            return "residence_permit"
        elif "visa" in norm_field_value:
            return "visa"
        elif any(
            x in norm_field_value
            for x in ["national id", "national identification", "citizen id", "citizenship card"]
        ):
            return "national_id"
        elif any(
            x in norm_field_value
            for x in ["foreign id", "alien id", "immigration id", "alien registration"]
        ):
            return "foreign_id"
        elif any(x in norm_field_value for x in ["id", "identity", "identification"]):
            return "state_id"

    # Detect SSN from the number pattern itself
    doc_number_field = fields.get("DocumentNumber")
    if doc_number_field and getattr(doc_number_field, "value", None):
        raw_number = str(doc_number_field.value).strip()
        # Classic SSN pattern: 123-45-6789
        if re.fullmatch(r"\d{3}-\d{2}-\d{4}", raw_number):
            return "social_security"

    # Final fallback
    return "id"


# Clean the extracted name
def clean_name_value(value: str | None) -> str | None:
    """Remove SSNs, digits lines, and newlines from a name-like string."""
    if not value:
        return value

    # Remove SSN-like patterns completely
    value = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "", value)

    # Split on newlines; drop any line that contains digits (likely noise)
    lines = value.splitlines()
    lines = [l for l in lines if not re.search(r"\d", l)]

    # Strip extra spaces and join back into a single line
    cleaned = " ".join(l.strip() for l in lines if l.strip())

    return cleaned or None


def format_name(fields: dict) -> dict:
    """Extract first name, last name, and full name from Azure AI fields"""
    raw_first_name = get_field_value(fields.get("FirstName"))
    raw_last_name = get_field_value(fields.get("LastName"))

    # Clean the raw values (handles \n and noisy text like 'OLIVIA 178-82-6396')
    first_name = clean_name_value(raw_first_name)
    last_name = clean_name_value(raw_last_name)

    if first_name and last_name:
        full_name = f"{first_name} {last_name}"
    elif first_name:
        full_name = first_name
    elif last_name:
        full_name = last_name
    else:
        full_name = None

    return {
        "first_name": first_name,
        "last_name": last_name,
        "full_name": full_name,
    }

def format_date(date_field) -> str:
    """Format date to YYYY-MM-DD"""
    if not date_field or not date_field.value:
        return None
    
    try:
        date_obj = date_field.value
        if hasattr(date_obj, 'year'):
            return f"{date_obj.year:04d}-{date_obj.month:02d}-{date_obj.day:02d}"
        return str(date_obj)
    except:
        return None

def get_field_value(field):
    """Extract value from Azure AI field"""
    if not field:
        return None
    
    value = field.value if hasattr(field, 'value') else field
    
    if value is None or value == "":
        return None
    
    if hasattr(value, '__dict__'):
        if hasattr(value, 'street_address'):
            parts = []
            if hasattr(value, 'street_address') and value.street_address:
                street = str(value.street_address)
                import re
                street = re.sub(r'^\d+\.\s*', '', street)
                parts.append(street)
            if hasattr(value, 'city') and value.city:
                parts.append(str(value.city))
            if hasattr(value, 'state') and value.state:
                parts.append(str(value.state))
            if hasattr(value, 'postal_code') and value.postal_code:
                parts.append(str(value.postal_code))
            return ", ".join(parts) if parts else None
        return str(value)
    
    result = str(value).strip()
    import re
    result = re.sub(r'^\d+\.\s*', '', result)
    # strip trailing commas / semicolons / spaces
    result = result.rstrip(",; ").strip()
    return result

def extract_id_data(file_content: bytes) -> List[Dict[str, Any]]:
    """
    Extract ID data using Azure Document Intelligence.

    - Uses prebuilt-idDocument model.
    - For multi-page PDFs, only the first 2 pages are analyzed (pages 1–2).
    - Supports multiple IDs in a single document: returns a list of results.
    - Filters out docs that don't look like real IDs.
    """
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not key:
        raise ValueError("Azure Document Intelligence credentials not configured")

    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    # Detect if the file is a PDF (multi-page possible)
    is_pdf = file_content[:4] == b"%PDF"

    if is_pdf:
        # Limit to first 2 pages only
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-idDocument",
            document=file_content,
            pages="1-2", 
        )
    else:
        # Images / single-page docs: Azure handle it without pages
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-idDocument",
            document=file_content,
        )

    result = poller.result()

    if not result.documents:
        raise ValueError("No ID document detected in the image")

    extracted_list: List[Dict[str, Any]] = []

    # Build raw list of extracted docs
    for id_doc in result.documents:
        fields = id_doc.fields

        city = extract_city_from_address(fields.get("Address"))
        names = format_name(fields)
        extracted_data: Dict[str, Any] = {
            "document_type": map_document_type(id_doc),
            "id_number": get_field_value(fields.get("DocumentNumber")),
            "first_name": names["first_name"],
            "last_name": names["last_name"],
            "full_name": names["full_name"],
            "date_of_birth": format_date(fields.get("DateOfBirth")),
            "issue_date": format_date(fields.get("DateOfIssue")),
            "expiry_date": format_date(fields.get("DateOfExpiration")),
            "place_of_birth": get_field_value(fields.get("PlaceOfBirth")),
            "nationality": get_field_value(fields.get("Nationality")),
            "address": get_field_value(fields.get("Address")),
            "city": city,
            "state": get_field_value(fields.get("Region")),
            "country": get_field_value(fields.get("CountryRegion")),
            "gender": get_field_value(fields.get("Sex")),
        }

        extracted_list.append(extracted_data)

    # Filter: keep only IDs that actually have some identity info
    def looks_like_id(d: Dict[str, Any]) -> bool:
        vital_values = [
            d.get("id_number"),
            d.get("full_name"),
            d.get("date_of_birth"),
        ]
        # At least one vital field must be present
        return any(v not in (None, "", []) for v in vital_values)

    valid_ids = [d for d in extracted_list if looks_like_id(d)]

    if not valid_ids:
        raise ValueError("The provided document does not appear to be an ID document.")

    return valid_ids