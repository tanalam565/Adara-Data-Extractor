import os
import uuid
import time
import random
import hashlib
from typing import Dict, Any
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI

def extract_insurance_data_ocr(file_content: bytes) -> Dict[str, Any]:
    """
    Extract insurance data using Azure Form Recognizer prebuilt-read (OCR)
    to get raw text, then enhance with GPT-4.
    """
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    if not endpoint or not key:
        raise ValueError("Azure Document Intelligence credentials not configured")
    
    # Initialize Azure Form Recognizer client
    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )
    
    # Analyze document with read model (OCR) - first 10 pages
    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        document=file_content,
        pages="1-10"
    )
    
    result = poller.result()
    
    # Extract all text content
    full_text = result.content if hasattr(result, 'content') else ""
    
    # Extract text by pages if available
    pages_text = []
    if hasattr(result, 'pages') and result.pages:
        for page in result.pages:
            page_text = ""
            if hasattr(page, 'lines'):
                for line in page.lines:
                    if hasattr(line, 'content'):
                        page_text += line.content + "\n"
            pages_text.append(page_text)
    
    # Debug logging
    print(f"Extracted text length: {len(full_text)} characters")
    print(f"Extracted {len(pages_text)} pages")
    
    # Check if we extracted anything
    if not full_text and not pages_text:
        raise ValueError("No text could be extracted from document. It may be corrupted, image-only, or not a valid document.")
    
    # Use GPT-4 to structure the data (required - no fallback)
    structured_data = enhance_with_gpt4_ocr(full_text, pages_text)
    
    return structured_data


def enhance_with_gpt4_ocr(full_text: str, pages_text: list) -> Dict[str, Any]:
    """Use GPT-4 to extract and structure insurance information from OCR text."""
    
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    if not openai_endpoint or not openai_key:
        raise ValueError("Azure OpenAI credentials not configured. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.")
    
    client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_key=openai_key,
        api_version="2024-02-15-preview"
    )
    
    # Use more text for better context 
    text_sample = full_text

    # Cache-breaking identifiers
    extraction_id = str(uuid.uuid4())
    random_seed = random.randint(100000, 999999)
    doc_hash = hashlib.sha256(full_text.encode()).hexdigest()[:16]

    time.sleep(0.5)

    cache_breaker = random.choice([
        "You must extract data with precision.",
        "Your task is to extract insurance information.",
        "Extract the insurance data accurately.",
        "Parse the following insurance document carefully."
    ])
    
    prompt = f"""
Extract insurance policy information from the following OCR text. This is raw text from an insurance document, so look carefully for dates, amounts, coverages, and deductibles that may be scattered throughout.

OCR Text:
{text_sample}

INSTRUCTIONS:

1) POLICY & DATES
- policy_number: Policy or certificate number.
- insured_name: Name of the PRIMARY insured person or entity only. If multiple people are insured, separate the names with comma.
  - Do NOT include Additional Insured, Interested Party, Mortgagee, or Lienholder names.
- insurance_company: Insurance company name.
- start_date: Original policy inception/effective date in YYYY-MM-DD format (e.g. "2024-03-15").
  - Do NOT use the cancellation, termination, voidance, or notice date as start_date.
  - Do NOT use "Date of Coverage Voidance", "Effective Date of Policy/Certificate Voidance", or any date associated with a cancellation/termination/voidance event.
  - On cancellation, termination, or voidance documents, "Effective Date of Policy/Certificate" refers to the voidance effective date, NOT the original policy start date. Do NOT use it as start_date.
  - Only use a date that is clearly labeled as the original policy inception or start date, separate from any cancellation/termination/voidance context.
  - If the document is a reinstatement notice, cancellation notice, voidance notice, acknowledgment of cancellation, or any status/event document that does not explicitly state the original policy start date, return null.
  - Do NOT infer or guess start_date from form numbers, document identifiers, prepared dates, processed dates, or any date not explicitly labeled as a policy start or inception date.
  - If the policy start date is not explicitly stated, return null.
- end_date: Policy expiration date in YYYY-MM-DD format. Do not assume. Return null if not present.
  - If the document is a reinstatement notice, cancellation notice, voidance notice, acknowledgment of cancellation, or any status/event document that does not explicitly state the policy expiration date, return null.
  - Do NOT infer or guess end_date from any date not explicitly labeled as a policy expiration or end date.
- premium_amount: The total/regular policy premium cost. Do NOT use outstanding or overdue amounts here.
                  If multiple premiums are shown, mention them like field all_coverages and all_deductibles field only if necessary.
- premium_due: The outstanding or overdue premium amount currently owed (e.g. labeled as "Premium Due", "Premium Amount Due", "Amount Due").
               This is different from the regular policy premium. Return null if not present.
- property_address: The insured property/location address (not just mailing address).

2) ALL_COVERAGES FIELD (VERY IMPORTANT)
- all_coverages must be a SINGLE STRING or null.
- If multiple coverages or limits exist, list ALL of them in this ONE string.
- Format each item as: "<coverage name>=$<amount or other limit text>".
- Separate items with comma and a space: ", ".
- Examples of coverage items:
  - "Coverage C - Personal Property=$18,700"
  - "Coverage D - Loss of Use=ALS up to 24 months"
  - "Coverage E - Personal Liability (Each Occurrence)=$300,000"
  - "Biological Deterioration or Damage=$10,000"
  - "Building Additions and Alterations=$1,870"
  - "Loss Assessment=$1,000"
  - "Water Backup Limited=$5,000"
  - "Tools=$2,500"
- The final all_coverages string should look like:
  "Coverage C - Personal Property=$18,700, Coverage D - Loss of Use=ALS up to 24 months, Coverage E - Personal Liability (Each Occurrence)=$300,000, ..."

- coverage_amount: This field should reflect the coverage amount related ONLY to personal liability (for example, coverage often labeled as "Coverage E - Personal Liability", "Personal Liability", "Liability (Each Occurrence)", or similar), from any insurer.
  - Extract ONLY the amount part, like "300,000", with no additional words, dollar sign or labels.
  - If multiple personal liability amounts appear, choose the main personal liability limit (usually the primary Coverage E / Personal Liability limit).
  - If no clear personal liability coverage amount can be found, set coverage_amount to null.

3) ALL_DEDUCTIBLES FIELD (VERY IMPORTANT)
- all_deductibles must be a SINGLE STRING or null.
- If multiple deductibles exist, list ALL of them in this ONE string.
- Format each item as: "<peril or description>=$<amount or other deductible text>".
- Separate items with comma and a space: ", ".
- Examples:
  - "All perils=$1,000"
  - "All other perils=$1,000"
  - "Water backup=$1,000"
  - "Wind/hail=2% of Coverage A"
- The final all_deductibles string should look like:
  "All perils=$1,000, Water backup=$1,000"

- deductible: This field should return the standard or main deductible (for example, "All perils", "All other perils", "Section I Deductible", or similar base deductible), not special deductibles like separate wind/hail, hurricane, storm, or named-peril deductibles.
  - Extract ONLY the amount part, like "1,000", with no additional words, dollar sign  or labels.
  - If multiple deductibles exist, choose the standard/base deductible that generally applies to most property losses.
  - If no clear standard/base deductible can be found, set deductible to null.

4) GENERAL RULES
- Include ALL distinct coverages and deductibles you can find.
- Do NOT invent anything that is not clearly present.
- If something is present but ambiguous, use the best short text for it.
- If you cannot find any coverages, set all_coverages to null.
- If you cannot find any deductibles, set all_deductibles to null.
- If you cannot find any personal liability coverage amount, set coverage_amount to null.
- If you cannot find a standard/base deductible, set deductible to null.
- If multiple people are insured, separate the names with comma.

5) DOCUMENT SUBTYPE
- document_subtype: Identify what kind of insurance document this is.
  - Possible values: "policy", "cancellation_notice", "termination_notice", "voidance_notice", "replacement_notice", "reinstatement_notice", "renewal_notice", "out_of_force_notice", "additional_insured_removal", null
  - If the document contains multiple subtypes, return the most prominent one.
  - If it cannot be determined, return null.

6) CANCELLATION & TERMINATION
- cancellation_termination_date: The date the policy was cancelled or terminated, in YYYY-MM-DD format. Return null if not present.
- cancellation_reason: The stated reason for cancellation (e.g. "Non-payment of premium", "Insured request"). Return null if not present.

7) REPLACEMENT
- replaced_policy_number: If this document indicates the policy replaced a previous policy, return the old policy number. Return null if not present.
- replacement_date: The date the replacement took effect, in YYYY-MM-DD format. Return null if not present.

8) REINSTATEMENT
- reinstatement_date: The date the policy was reinstated after cancellation or lapse, in YYYY-MM-DD format. Return null if not present.

9) RENEWAL
- renewal_due_date: The date by which the policy must be renewed, in YYYY-MM-DD format. Return null if not present.

10) OUT OF FORCE
- out_of_force: Indicates whether the policy has gone out of force.
  - Return "Once" if the document explicitly uses the phrase "out-of-force" or "out of force" exactly one time.
  - Return "Multiple" if the document explicitly uses the phrase "out-of-force" or "out of force" more than once, or states "multiple time-out-of-force periods exist" or similar.
  - Return null in ALL other cases including cancellation, termination, voidance, or any event that does not explicitly use the words "out-of-force" or "out of force".
  - Do NOT infer out_of_force from cancellation dates, voidance dates, or termination dates.

- out_of_force_start_date: The date the policy went out of force, in YYYY-MM-DD format.
  - ONLY extract if out_of_force is not null.
  - Look ONLY for explicit phrases like "out-of-force from [date]" or "out of force from [date]".
  - Do NOT use cancellation dates, voidance dates, or termination dates for this field.
  - Return null if not present or if out_of_force is null.

- out_of_force_end_date: The date the policy was restored from out-of-force, in YYYY-MM-DD format.
  - ONLY extract if out_of_force is not null.
  - Look ONLY for explicit phrases like "out-of-force from [date] to [date]", where the second date is the end.
  - Do NOT use cancellation dates, voidance dates, or termination dates for this field.
  - Return null if not present or if out_of_force is null.

11) ADDITIONAL INSURED REMOVAL
- additional_insured_removal: If the document indicates that an additional insured has been removed or that the tenant requested removal of a specific entity as an additional insured:
  - Return the name of the entity being removed as stated in the BODY of the letter (e.g. "Adara Communities").
  - The removed entity is typically referenced in phrases like "delete your organization's name", "remove [entity] as additional insured", "requested to delete [entity]", or similar.
  - Do NOT return the name of the letter recipient, mortgagee, lienholder, or addressee at the top of the document — these are being NOTIFIED of the removal, not being removed.
  - Do NOT return the insured's name or the insurance company name.
  - If multiple entities are being removed, separate names with a comma.
  - Return null if no additional insured removal is mentioned.

Return ONLY a valid JSON object with these keys:
- policy_number
- insured_name
- insurance_company
- all_coverages
- coverage_amount
- start_date
- end_date
- premium_amount
- premium_due
- all_deductibles
- deductible
- property_address
- document_subtype
- cancellation_termination_date
- cancellation_reason
- replaced_policy_number
- replacement_date
- reinstatement_date
- renewal_due_date
- out_of_force
- out_of_force_start_date
- out_of_force_end_date
- additional_insured_removal

Return just the JSON, no explanation text.
"""

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": f"{cache_breaker} SESSION ID: {extraction_id} DOCUMENT HASH: {doc_hash} You are a precise insurance document data extractor. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1500,
        seed=random_seed,
        user=f"extraction_{extraction_id}"
    )
    
    import json
    result_text = response.choices[0].message.content.strip()
    # Remove markdown code blocks if present
    result_text = result_text.replace("```json", "").replace("```", "").strip()
    
    structured_data = json.loads(result_text)
    return structured_data