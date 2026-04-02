from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import os
import json
import logging
import uvicorn
import traceback
import asyncio
from typing import Optional, Tuple, Dict, Any, List
import re

# Import all extractors
from proof_of_income import extract_employment_data_ocr
from id_document import extract_id_data
from insurance import extract_insurance_data_ocr
from invoice import extract_invoice_data
from bulk_invoice import extract_invoice_data as extract_bulk_invoice_data

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Document Extractor API", version="1.0.0")

# Configuration - Different API keys for different document types
API_KEY_INVOICE     = os.getenv("API_KEY_INVOICE", "")
API_KEY_POI         = os.getenv("API_KEY_POI", "")
API_KEY_ID          = os.getenv("API_KEY_ID", "")
API_KEY_INSURANCE   = os.getenv("API_KEY_INSURANCE", "")
API_KEY_BULKINVOICE = os.getenv("API_KEY_BULKINVOICE", "")
API_KEY_MASTER      = os.getenv("API_KEY_MASTER", "")  # Master key for all services
ENABLE_FRONTEND     = os.getenv("ENABLE_FRONTEND", "true").lower() == "true"

# File validation constants
MAX_FILE_SIZE      = 15 * 1024 * 1024  # 15MB
BULK_MAX_FILE_SIZE = 5  * 1024 * 1024  # 5MB per file for bulk invoice
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.heic', '.heif'}

# Semaphore for bulk invoice — limit concurrent GPT calls
_gpt_semaphore = asyncio.Semaphore(3)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Private-Network"] = "true"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

app.add_middleware(PrivateNetworkAccessMiddleware)

# ==================== API KEY VERIFICATION ====================

async def verify_api_key_invoice(x_api_key: Optional[str] = Header(None)):
    """Verify API key for invoice extraction"""
    if not API_KEY_INVOICE and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_INVOICE and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for invoice extraction")
    
    return True

async def verify_api_key_poi(x_api_key: Optional[str] = Header(None)):
    """Verify API key for POI extraction"""
    if not API_KEY_POI and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_POI and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for POI extraction")
    
    return True

async def verify_api_key_id(x_api_key: Optional[str] = Header(None)):
    """Verify API key for ID extraction"""
    if not API_KEY_ID and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_ID and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for ID extraction")
    
    return True

async def verify_api_key_insurance(x_api_key: Optional[str] = Header(None)):
    """Verify API key for insurance extraction"""
    if not API_KEY_INSURANCE and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_INSURANCE and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for insurance extraction")
    
    return True

async def verify_api_key_bulkinvoice(x_api_key: Optional[str] = Header(None)):
    """Verify API key for bulk invoice extraction"""
    if not API_KEY_BULKINVOICE and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_BULKINVOICE and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for bulk invoice extraction")
    
    return True

# ==================== HELPER FUNCTIONS ====================

def count_non_null_fields(data: Any) -> int:
    """Recursively count number of fields with actual values"""
    if isinstance(data, dict):
        return sum(count_non_null_fields(v) for v in data.values())
    if isinstance(data, list):
        return sum(count_non_null_fields(item) for item in data)
    return 0 if data in (None, "", []) else 1

# ==================== INVOICE HELPERS ====================

def requires_human_review_invoice(extracted_data) -> Tuple[bool, str]:
    """Determine if invoice needs human review"""
    if not extracted_data:
        return True, "No data extracted"

    invoices = [extracted_data] if isinstance(extracted_data, dict) else extracted_data
    reasons: list[str] = []

    for idx, data in enumerate(invoices, start=1):
        vital_fields = ["invoice_id", "vendor_name", "line_items"]
        missing_vital = [
            field for field in vital_fields
            if data.get(field) in (None, "", [])
        ]
        if missing_vital:
            reasons.append(f"Invoice {idx}: missing vital field(s): {', '.join(missing_vital)}")

    if reasons:
        return True, " ; ".join(reasons)
    return False, ""

# ==================== POI HELPERS ====================

def _value_is_empty(value: Any) -> bool:
    """Check if a value is considered empty"""
    return value in (None, "", [])

def _iter_jobs(extracted_data: Dict[str, Any]):
    """Helper generator to iterate through (applicant_name, job_dict) pairs"""
    applicants = extracted_data.get("applicants", []) or []
    for applicant in applicants:
        applicant_name = applicant.get("applicant")
        jobs = applicant.get("jobs", []) or []
        for job in jobs:
            yield applicant_name, job

def requires_human_review_poi(extracted_data: dict) -> Tuple[bool, str]:
    """Determine if POI needs human review"""
    if not extracted_data:
        return True, "No data extracted"

    applicants = extracted_data.get("applicants", []) or []
    if not applicants:
        return True, "No applicants found"

    has_complete_record = False
    missing_vital_fields: set[str] = set()

    for applicant_name, job in _iter_jobs(extracted_data):
        avg_income = job.get("average_monthly_income")

        missing_for_this = []
        if _value_is_empty(applicant_name):
            missing_for_this.append("applicant")
        if _value_is_empty(avg_income):
            missing_for_this.append("average_monthly_income")

        if not missing_for_this:
            has_complete_record = True
        else:
            missing_vital_fields.update(missing_for_this)

    if has_complete_record:
        return False, ""

    if not missing_vital_fields:
        return True, "Missing vital fields"

    missing_str = ", ".join(sorted(missing_vital_fields))
    return True, f"Missing vital field(s): {missing_str}"

def has_proof_of_employment(extracted_data: dict) -> bool:
    """Determine if the document appears to be proof of employment/income"""
    for applicant_name, job in _iter_jobs(extracted_data):
        if not _value_is_empty(applicant_name):
            return True
    return False

def filter_verification_applications(extracted_data: dict) -> dict:
    """Remove jobs that are only from employment verification applications"""
    if not extracted_data or "applicants" not in extracted_data:
        return extracted_data
    
    filtered_applicants = []
    for applicant in extracted_data.get("applicants", []):
        valid_jobs = [
            job for job in applicant.get("jobs", [])
            if job.get("document_type") not in ["employment verification application"]
        ]
        if valid_jobs:
            filtered_applicants.append({
                "applicant": applicant.get("applicant"),
                "jobs": valid_jobs
            })
    
    return {"applicants": filtered_applicants}

def get_vital_fields_status_poi(extracted_data: dict) -> Dict[str, bool]:
    """Return a summary of whether we found any non-empty values for each vital field"""
    has_applicant = False
    for applicant_name, job in _iter_jobs(extracted_data):
        if not _value_is_empty(applicant_name):
            has_applicant = True
    return {"applicant": has_applicant}

PERIODS_PER_YEAR = {
    "weekly": 52,
    "bi-weekly": 26,
    "biweekly": 26,
    "semi-monthly": 24,
    "semimonthly": 24,
    "monthly": 12,
}

def _parse_currency(value: Any) -> Optional[float]:
    """Parse a numeric amount into a float"""
    if not value or not isinstance(value, str):
        return None
    matches = re.findall(r"[\d,]+(?:\.\d+)?", value)
    if not matches:
        return None
    num_str = matches[-1]
    try:
        return float(num_str.replace(",", ""))
    except ValueError:
        return None

def recompute_average_monthly_income(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute average_monthly_income and keep income_calculation_details consistent"""
    if not extracted_data:
        return extracted_data

    for _, job in _iter_jobs(extracted_data):
        yearly_salary_str = job.get("yearly_salary")
        yearly = _parse_currency(yearly_salary_str) if yearly_salary_str else None

        if yearly:
            monthly = yearly / 12.0
            monthly_rounded = round(monthly, 2)
            monthly_str = f"{monthly_rounded:.2f}"
            job["average_monthly_income"] = monthly_str
            job["income_calculation_details"] = (
                f"Based on yearly salary: {yearly_salary_str} / 12 months = {monthly_str}/month"
            )
            continue

        checks = job.get("three_consecutive_checks") or []
        pay_freq_raw = job.get("pay_frequency")
        pay_freq = pay_freq_raw.strip().lower() if isinstance(pay_freq_raw, str) else None
        periods_per_year = PERIODS_PER_YEAR.get(pay_freq) if pay_freq else None

        amounts: List[float] = []
        for entry in checks:
            amt = _parse_currency(entry)
            if amt is not None:
                amounts.append(amt)

        if amounts and periods_per_year:
            if pay_freq == "monthly":
                avg_monthly = sum(amounts) / len(amounts)
                avg_monthly_rounded = round(avg_monthly, 2)
                monthly_str = f"{avg_monthly_rounded:.2f}"
                job["average_monthly_income"] = monthly_str
                job["income_calculation_details"] = (
                    f"Based on {len(amounts)} monthly paychecks: "
                    f"average monthly income = {monthly_str}/month"
                )
            else:
                avg_check = sum(amounts) / len(amounts)
                annual = avg_check * periods_per_year
                monthly = annual / 12.0
                avg_check_rounded = round(avg_check, 2)
                monthly_rounded = round(monthly, 2)
                avg_check_str = f"{avg_check_rounded:.2f}"
                monthly_str = f"{monthly_rounded:.2f}"
                job["average_monthly_income"] = monthly_str
                job["income_calculation_details"] = (
                    f"Based on {len(amounts)} {pay_freq} paychecks: "
                    f"average paycheck = {avg_check_str}. "
                    f"{(pay_freq_raw or pay_freq).capitalize()} pay: "
                    f"{avg_check_str} × {periods_per_year} pay periods / 12 months = {monthly_str}/month"
                )

    return extracted_data

# ==================== ID HELPERS ====================

def filter_valid_ids(ids: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out IDs that have neither full_name nor id_number.
    These are likely noise/artifacts and not actual ID extractions.
    """
    valid_ids = []
    for id_data in ids:
        has_name = id_data.get("full_name") not in (None, "", [])
        has_id_number = id_data.get("id_number") not in (None, "", [])
        
        # Keep ID only if it has at least one of these critical fields
        if has_name or has_id_number:
            valid_ids.append(id_data)
    
    return valid_ids

def count_non_null_fields_id(data: Dict[str, Any]) -> int:
    """Count number of fields with actual values in a single ID dict"""
    if not data:
        return 0
    return sum(1 for value in data.values() if value not in (None, "", []))

def requires_human_review_id(extracted_data) -> Tuple[bool, str]:
    """Determine if ID needs human review"""
    if not extracted_data:
        return True, "No data extracted"

    if isinstance(extracted_data, dict):
        ids = [extracted_data]
    else:
        ids = extracted_data

    # Filter out invalid IDs (missing both name and ID number)
    ids = filter_valid_ids(ids)
    
    # If all IDs were filtered out, return error
    if not ids:
        return True, "No valid ID data found"

    reasons: list[str] = []

    for idx, data in enumerate(ids, start=1):
        doc_type = (data.get("document_type") or "").lower()

        if doc_type == "social_security":
            vital_fields = ["id_number", "full_name"]
            min_total_fields = 2
        else:
            vital_fields = ["id_number", "full_name", "date_of_birth"]
            min_total_fields = 6

        missing_vital = [
            field for field in vital_fields
            if data.get(field) in (None, "", [])
        ]

        if missing_vital:
            missing_str = ", ".join(missing_vital)
            reasons.append(f"ID {idx}: missing vital field(s): {missing_str}")
            continue

        field_count = count_non_null_fields_id(data)
        if field_count < min_total_fields:
            reasons.append(
                f"ID {idx}: insufficient data extracted "
                f"({field_count}/{min_total_fields} required fields)"
            )

    if reasons:
        return True, " ; ".join(reasons)
    return False, ""

# ==================== INSURANCE HELPERS ====================

def requires_human_review_insurance(extracted_data: dict) -> Tuple[bool, str]:
    """Determine if insurance needs human review"""
    if not extracted_data:
        return True, "No data extracted"
    
    vital_fields = ["policy_number", "insured_name", "insurance_company"]
    missing_vital = [
        field for field in vital_fields
        if extracted_data.get(field) in (None, "", [])
    ]
    
    if missing_vital:
        missing_str = ", ".join(missing_vital)
        return True, f"Missing vital field(s): {missing_str}"
    
    field_count = count_non_null_fields(extracted_data)
    if field_count < 6:
        return True, f"Insufficient data extracted ({field_count}/6 required fields)"
    
    return False, ""

# ==================== ROOT ENDPOINT ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    frontend_links = ""
    if ENABLE_FRONTEND:
        frontend_links = """
                <div style="margin-top: 30px;">
                    <h3>Frontend Tools:</h3>
                    <ul style="line-height: 2;">
                        <li><a href="/test" style="color: #0066cc;">🧪 Test Interface</a> - Upload and test documents</li>
                    </ul>
                </div>
        """
    
    return f"""
    <html>
        <head><title>Unified Document Extractor API</title></head>
        <body style="font-family: Arial; padding: 40px; background: #f5f5f5;">
            <div style="max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h1 style="color: #333;">Unified Document Extractor API</h1>
                <p style="color: #666;">Powered by Azure Document Intelligence & Azure OpenAI</p>
                
                <div style="margin-top: 30px;">
                    <h3>Available Endpoints:</h3>
                    <ul style="line-height: 2;">
                        <li><a href="/docs" style="color: #0066cc;">📚 API Documentation</a></li>
                        <li><a href="/health" style="color: #0066cc;">❤️ Health Check</a></li>
                    </ul>
                </div>
                
                {frontend_links}
                
                <div style="margin-top: 30px;">
                    <h3>Extraction Endpoints:</h3>
                    <ul style="line-height: 2;">
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/invoice</code> - Invoice extraction</li>
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/poi</code> - Proof of Income extraction</li>
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/id</code> - ID Document extraction</li>
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/insurance</code> - Insurance Document extraction</li>
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/bulkinvoice</code> - Bulk Invoice extraction</li>
                    </ul>
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: #f0f8ff; border-left: 4px solid #0066cc;">
                    <h3>Supported Documents:</h3>
                    <ul>
                        <li><strong>Invoice:</strong> Vendor invoices, bills, receipts</li>
                        <li><strong>POI:</strong> Paystubs, employer letters, W-2, 1099, bank statements</li>
                        <li><strong>ID:</strong> Driver licenses, passports, national IDs, visas</li>
                        <li><strong>Insurance:</strong> Home insurance policies, certificates</li>
                        <li><strong>Bulk Invoice:</strong> Multiple vendor invoices in a single request</li>
                    </ul>
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: #fff8dc; border-left: 4px solid #ffa500;">
                    <h3>Authentication:</h3>
                    <p>Include your API key in the request header:</p>
                    <code style="background: #f0f0f0; padding: 10px; display: block; margin-top: 10px;">
                        X-API-Key: your_api_key_here
                    </code>
                </div>
            </div>
        </body>
    </html>
    """

# API Documentation
@app.get("/api-docs", response_class=HTMLResponse)
async def api_documentation():
    """Serve API integration documentation"""
    try:
        with open("api-documentation.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Documentation not found</h1>"
        
# ==================== INVOICE EXTRACTION ====================

@app.post("/extract/invoice")
async def extract_invoice(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_invoice)
):
    """Extract data from invoice document"""
    start_time = datetime.now()
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        try:
            extracted_data = extract_invoice_data(content)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        invoices: List[Dict[str, Any]] = [extracted_data] if isinstance(extracted_data, dict) else (extracted_data or [])
        if not invoices:
            raise HTTPException(status_code=400, detail="No invoice data returned")
        
        needs_review, review_reason = requires_human_review_invoice(invoices)
        field_counts = [count_non_null_fields(i) for i in invoices]
        field_count = min(field_counts) if field_counts else 0
        total_time = (datetime.now() - start_time).total_seconds()
        status = "needs_review" if needs_review else "success"
        
        return {
            "status": status,
            "document_type": "invoice",
            "requires_human_review": needs_review,
            "review_reason": review_reason if needs_review else None,
            "data": invoices,
            "field_count": field_count,
            "processing_time_seconds": round(total_time, 2),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        fname = getattr(file, "filename", None) or "unknown"
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": fname, "traceback": error_trace[:500]})

# ==================== POI EXTRACTION ====================

@app.post("/extract/poi")
async def extract_poi(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_poi)
):
    """Extract proof of income/employment data from document"""
    start_time = datetime.now()
    
    logger.info(f"Received POI file: {file.filename}")
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        content = await file.read()
        logger.info(f"File read: {len(content)} bytes")
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        try:
            logger.info("Starting OCR extraction...")
            extracted_data = extract_employment_data_ocr(content)
            extraction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"OCR extraction completed in {extraction_time:.2f}s")
        except ValueError as ve:
            extraction_time = (datetime.now() - start_time).total_seconds()
            raise HTTPException(status_code=400, detail=str(ve))
        
        logger.info("Applying filter for employment verification applications...")
        extracted_data = filter_verification_applications(extracted_data)
        extracted_data = recompute_average_monthly_income(extracted_data)
        
        if not has_proof_of_employment(extracted_data):
            msg = "The provided document does not appear to be a proof of employment."
            raise HTTPException(status_code=400, detail=msg)
        
        needs_review, review_reason = requires_human_review_poi(extracted_data)
        logger.info(f"Needs review: {needs_review}, Reason: {review_reason}")
        
        vital_fields_status = get_vital_fields_status_poi(extracted_data)
        field_count = count_non_null_fields(extracted_data)
        total_time = (datetime.now() - start_time).total_seconds()
        status = "needs_review" if needs_review else "success"
        
        return {
            "status": status,
            "document_type": "poi",
            "requires_human_review": needs_review,
            "review_reason": review_reason if needs_review else None,
            "data": extracted_data,
            "field_count": field_count,
            "vital_fields_status": vital_fields_status,
            "processing_time_seconds": round(total_time, 2),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        logger.error(f"Error after {total_time:.2f}s: {error_msg}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": file.filename, "traceback": error_trace[:500]})

# ==================== ID EXTRACTION ====================

@app.post("/extract/id")
async def extract_id(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_id)
):
    """Extract data from ID document"""
    start_time = datetime.now()
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        try:
            extracted_data = extract_id_data(content)
            
            if isinstance(extracted_data, dict):
                raw_ids: List[Dict[str, Any]] = [extracted_data]
            else:
                raw_ids = extracted_data
            
            # Filter out invalid IDs (missing both name and ID number)
            ids = filter_valid_ids(raw_ids)
            
            # If all IDs were filtered out, return error
            if not ids:
                msg = "The provided document does not appear to be an ID document."
                raise HTTPException(status_code=400, detail=msg)
                
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        needs_review, review_reason = requires_human_review_id(ids)
        
        all_missing_vitals = all(
            not (
                d.get("id_number") not in (None, "", [])
                or d.get("full_name") not in (None, "", [])
                or d.get("date_of_birth") not in (None, "", [])
            )
            for d in ids
        )
        
        if all_missing_vitals:
            msg = "The provided document does not appear to be an ID document."
            raise HTTPException(status_code=400, detail=msg)
        
        vital_fields = ["id_number", "full_name", "date_of_birth"]
        vital_fields_status = {
            field: all(d.get(field) not in (None, "", []) for d in ids)
            for field in vital_fields
        }
        
        field_counts = [count_non_null_fields_id(d) for d in ids]
        field_count = min(field_counts) if field_counts else 0
        total_time = (datetime.now() - start_time).total_seconds()
        status = "needs_review" if needs_review else "success"
        
        return {
            "status": status,
            "document_type": "id",
            "requires_human_review": needs_review,
            "review_reason": review_reason if needs_review else None,
            "data": ids,
            "field_count": field_count,
            "vital_fields_status": vital_fields_status,
            "processing_time_seconds": round(total_time, 2),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": file.filename, "traceback": error_trace[:500]})

# ==================== INSURANCE EXTRACTION ====================

@app.post("/extract/insurance")
async def extract_insurance(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_insurance)
):
    """Extract data from insurance document"""
    start_time = datetime.now()
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        try:
            extracted_data = extract_insurance_data_ocr(content)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        needs_review, review_reason = requires_human_review_insurance(extracted_data)
        
        if all(
            extracted_data.get(field) in (None, "", [])
            for field in ["policy_number", "insured_name", "insurance_company"]
        ):
            msg = "The provided document does not appear to be an insurance document."
            raise HTTPException(status_code=400, detail=msg)
        
        vital_fields = ["policy_number", "insured_name", "insurance_company"]
        vital_fields_status = {
            field: extracted_data.get(field) not in (None, "", [])
            for field in vital_fields
        }
        
        field_count = count_non_null_fields(extracted_data)
        total_time = (datetime.now() - start_time).total_seconds()
        status = "needs_review" if needs_review else "success"
        
        return {
            "status": status,
            "document_type": "insurance",
            "requires_human_review": needs_review,
            "review_reason": review_reason if needs_review else None,
            "data": extracted_data,
            "field_count": field_count,
            "vital_fields_status": vital_fields_status,
            "processing_time_seconds": round(total_time, 2),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": file.filename, "traceback": error_trace[:500]})

# ==================== BULK INVOICE EXTRACTION ====================

@app.post("/extract/bulkinvoice")
async def extract_bulk_invoice(
    files: List[UploadFile] = File(...),
    authenticated: bool = Depends(verify_api_key_bulkinvoice)
):
    """Extract data from multiple invoice documents"""
    start_time = datetime.now()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Validate all files before processing any
    file_data: List[tuple] = []
    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="One or more files are missing a filename."
            )
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed: '{file.filename}'. "
                       f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"File is empty: '{file.filename}'"
            )
        if len(content) > BULK_MAX_FILE_SIZE:
            size_mb = len(content) / 1024 / 1024
            raise HTTPException(
                status_code=400,
                detail=f"File too large: '{file.filename}' ({size_mb:.1f}MB). "
                       f"Maximum allowed size is 5MB.",
            )
        file_data.append((file.filename, content))

    logger.info(
        f"Processing {len(file_data)} file(s) in parallel",
        extra={"file_count": len(file_data), "filenames": [f for f, _ in file_data]},
    )

    loop = asyncio.get_running_loop()

    async def process_one(filename: str, content: bytes) -> Dict[str, Any]:
        async with _gpt_semaphore:
            return await loop.run_in_executor(
                None, extract_bulk_invoice_data, content, filename
            )

    try:
        raw_results: List[Dict[str, Any]] = await asyncio.gather(
            *[process_one(fname, content) for fname, content in file_data]
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"Bulk invoice extraction failed: {error_msg}",
            exc_info=True,
            extra={"filenames": [f for f, _ in file_data]},
        )
        correlation_match = re.search(r"\[([a-f0-9]{8})\]", error_msg)
        correlation_id = correlation_match.group(1) if correlation_match else None
        raise HTTPException(
            status_code=400,
            detail={
                "error": error_msg,
                "correlation_id": correlation_id,
                "hint": "Use the correlation_id to find full details in server logs.",
            }
        )

    # Assign auto-incrementing IDs across all files
    company_id = 1
    invoice_id = 1
    invoice_item_id = 1
    final_results: List[Dict[str, Any]] = []

    for raw in raw_results:
        invoices: List[Dict[str, Any]] = []

        for prop in raw.get("properties", []):
            current_invoice_id = invoice_id
            invoice_items: List[Dict[str, Any]] = []

            for item in prop.get("invoiceItems", []):
                invoice_items.append({
                    "invoiceItemId": invoice_item_id,
                    "invoiceId": current_invoice_id,
                    "description": item.get("description"),
                    "quantity": item.get("quantity"),
                    "price": item.get("price"),
                    "overhead": item.get("overhead"),
                    "freight": item.get("freight"),
                    "tax": item.get("tax"),
                    "discount": item.get("discount"),
                    "totalPrice": item.get("totalPrice"),
                })
                invoice_item_id += 1

            invoices.append({
                "invoiceId": current_invoice_id,
                "propertyName": prop.get("propertyName"),
                "vendorName": raw.get("vendorName"),
                "invoiceNumber": raw.get("invoiceNumber"),
                "invoiceDate": raw.get("invoiceDate"),
                "notes": prop.get("notes"),
                "uploadInvoice": raw.get("uploadInvoice"),
                "invoiceItems": invoice_items,
            })
            invoice_id += 1

        final_results.append({
            "companyId": company_id,
            "companyName": raw.get("companyName"),
            "invoices": invoices,
        })
        company_id += 1

    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Completed {len(file_data)} file(s) in {total_time:.2f}s",
        extra={
            "file_count": len(file_data),
            "duration_seconds": round(total_time, 2),
            "total_invoices": sum(len(r.get("invoices", [])) for r in final_results),
        },
    )

    return {
        "status": "success",
        "document_type": "bulkinvoice",
        "data": final_results,
        "processing_time_seconds": round(total_time, 2),
    }

# ==================== FRONTEND ROUTES ====================

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """Serve test page for all document types"""
    if not ENABLE_FRONTEND:
        raise HTTPException(status_code=404, detail="Frontend is disabled")
    
    try:
        with open("../frontend/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Test page not found</h1><p>Make sure frontend/index.html exists</p>"

# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "invoice": "available",
            "poi": "available",
            "id": "available",
            "insurance": "available",
            "bulkinvoice": "available"
        }
    }

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)