import os 
import json
import logging
from typing import Dict, Any, List
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("azure.ai").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def extract_employment_data_ocr(file_content: bytes) -> Dict[str, Any]:
    """
    Extract employment data using Azure Document Intelligence prebuilt-read (OCR)
    to get raw text, then format with GPT.
    """
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not key:
        raise ValueError("Azure Document Intelligence credentials are not configured")
    
    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body=file_content,
        pages="1-8"
    )

    result = poller.result()

    full_text = result.content if hasattr(result, "content") else ""
    
    logger.debug(f"Text preview: {full_text[:1000]}")

    pages_text: List[str] = []
    if hasattr(result, "pages") and result.pages:
        for page in result.pages:
            page_text = ""
            if hasattr(page, "lines"):
                for line in page.lines:
                    if hasattr(line, "content"):
                        page_text += line.content + "\n"
            pages_text.append(page_text)
    
    logger.info(f"Extracted text length: {len(full_text)} characters")
    logger.info(f"Extracted {len(pages_text)} pages")

    if pages_text:
        logger.debug(f"First page preview: {pages_text[0][:1000]}")

    if not full_text and not pages_text:
        raise ValueError(
            "No text could be extracted from document. "
            "It may be corrupted, image-only, or not a valid document."
        )
    
    structured_data = enhance_with_gpt(full_text, pages_text)

    return structured_data


def enhance_with_gpt(full_text: str, pages_text: list) -> Dict[str, Any]:
    """
    Use GPT to extract and structure employment information from OCR text.
    """
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") 

    if not openai_endpoint or not openai_key or not deployment_name:
        raise ValueError(
            "Azure OpenAI credentials not configured. Please set "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME."
        )

    client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_key=openai_key,
        api_version="2024-05-01-preview"
    )

    if full_text:
        text_sample = full_text
    else:
        text_sample = "\n\n".join(pages_text) 
    
    logger.info("Sending request to GPT for data structuring")

    prompt = f"""
You are given OCR text extracted from documents submitted by a prospective tenant as proof of income. 
Your task is to extract structured employment and income information.

CONTEXT
- The proof of income may include one or more of the following document types (possibly mixed in a single file):
  - Employer letter
  - Paystub
  - Employment verification application (A form provided by the property to be filled. This may include Texas Apartment Association (TAA),
    Florida Apartment Association (FAA), or Ohio Apartment Association (OAA))
  - Paycheck
  - Bank statements
  - Employer records (e.g., Walmart employment verification packet / employee records)
  - Other documents related to income or employment
- A single file may contain:
  - Multiple document types
  - Multiple applicants (e.g., husband and wife)
  - Multiple jobs per applicant (e.g., sales job + rideshare driving)
- If only bank statements are provided, you must still look for employers and income sources
  (e.g., repeated deposits labeled CASHAPP, ZELLE, VENMO, or similar digital payments, or deposits from a taco truck business, etc.)
- The goal is to determine financial eligibility for renting by extracting all relevant income/employment info

OCR_TEXT:
{text_sample}

EXTRACTION RULES

For each applicant and each of their jobs, extract:

1) applicant
   - Name of the applicant as provided in the documents.
   - If not clearly stated, infer the most likely name from signatures, letter headers, paystubs, or bank account owners.
   - If it is a joint bank account with multiple applicant names i.e more than one name is mentioned in the same bank statement, return only the first applicant name.
   - If it is a tax documnent with multiple applicant names i.e more than one name is mentioned in the same tax document, return only the first applicant name. Only return
     multuple applicants if taxes file separately or have separate tax forms or records.

2) document_type
   - One of:
     - "employer letter"
     - "paystub"
     - "employment verification application"
     - "paycheck"
     - "bank statements"
     - "employer record"
     - "w-2"
     - "f-1 visa"
     - "1099"
     - "tax document"
     - "other document" (if none of the above clearly apply)
   - If multiple document types exist for the same job, choose the dominant or most informative one.
   - If "employment verification application" of an applicant is supported with "employer letter", "paystub", "paycheck", "bank statements",
     "employer record", "w-2", "1099", "tax document" or other valid document, change document_type to type of the supported documents for that applicant only.
     For example, if there is an employment verification application for employees 'A' and 'B', but there is only supported documents(e.g W-2) to prove 
     income of applicant 'A', then document_type for 'A' -> "w-2" and for applicant 'B', return document_type -> "employment verification application"
   - Any employment affidavit is considered as "employer letter".

3) employer
   - Name of the company or business (e.g., "Walmart", "Uber", "Joe's Taco Truck").
   - If income comes from retirement, pension, or annuity payments, return the name of the pension or retirement fund 
     (e.g., "Fidelity Pension Services", "TRS Retirement System", "CalPERS").
   - If income comes from government benefits, return the relevant agency:
       - Social Security income → "Social Security Administration"
       - Disability income (SSDI) → "Social Security Administration"
       - Supplemental Security Income (SSI) → "Social Security Administration"
       - Veterans benefits → "U.S. Department of Veterans Affairs"
       - Unemployment benefits → "State Unemployment Agency" (include state name if visible)
   - If income comes from child support deposits, return "Child Support Deposits" (or the agency name if specified).
   - If the income source is a financial institution that issues recurring payments (e.g., annuities, structured settlements),
     return the institution name (e.g., "MetLife", "New York Life").
   - If the income is self-employment, the employer name should be the business name or, if no business name appears, 
     use a descriptive label such as "Self-employed — rideshare driver" or "Self-employed — landscaping services".
   - Return institution/school name if it is a F-1 visa.
   - If no employer or source name can be determined, return null.

4) position/title
   - Job title or role as written in the document (e.g., "Sales Associate", "Manager").
   - If the applicant appears to own the business, return "owner".
   - If self-employed (e.g., rideshare driver, freelance), return "self-employed" plus any explicit title if provided.
   - If the applicant provides income through vendors (e.g., taxi driver, food delivery), return "vendor" or "contractor".
   - If no position is mentioned or can not be determined, return null. 

5) employer_address
   - Employer's address if mentioned.
   - If multiple addresses appear, pick the one clearly associated with the employer.
   - Address of the source of income e.g social security administration, U.S. Department of Veterans Affairs, New York Life, and so on.
   - If not found, return null.

6) supervisor_name
   - Name of the supervisor or manager if present.
   - If not found, return null.

7) three_consecutive_checks
   - Identify up to 3 most recent paychecks or income deposits for that job.
   - Include date and GROSS amount for each (not NET), formatted as: "MM-DD-YY: amount". No currency sign. Get NET amount if GROSS is not available.
   - If it is 1040 or tax document, get the GROSS AMOUNT.
   - IMPORTANT: Always use GROSS pay (before deductions), NOT net pay (take-home).
   - Prefer checks/paystubs; if not available, use recurring bank statement deposits clearly tied to that employer/source.
   - If fewer than 3 checks/deposits are visible, return only those available.
   - If the income comes as benefits (e.g., Social Security, retirement plan, financial institutions, child support, and similar):
     - Only consider the recurring income amounts mentioned in the document.
   - If it is an F-1 visa, return the annual funding from the document.
   - If it is an employer letter or employee record, return the amount mentioned. Ignore invoices or bank statements.
   - For bank statements (when paystubs or paychecks are missing or incomplete):

    - Identify deposit transactions that appear to be income based on descriptors such as:
      PAYROLL, DIRECT DEP, DIRECT DEPOSIT, ADP, GUSTO, SQUARE, STRIPE, CLOVER, TOAST,
      PAYPAL BUSINESS, VENMO BUSINESS, CASH APP (business), employer names, or recurring
      business deposits.

    - Exclude the following from consideration as income:
      - Opening / Beginning / Ending balances
      - Internal transfers between the applicant’s own accounts
      - Chargebacks or reversals
      - Refunds or adjustments
      - Zelle deposits
      - Loan transfers or credit card payments
      - Any obvious non-income items even though it is recurring.
      - If it is a joint bank account with multiple applicant names i. e more than one name is mentioned in the same bank statement, return only the first applicant name.

    - If bank statements contain daily income or multiple income deposits from any jobs per day (for example, per-job, per-trip, or gig-type earnings), follow this procedure:
      - For example, uber driver, lyft driver, barder, mobile mechanic, and so on. The name of the income source should be mentioned in the bank statement.
      - If the bank statement is provided after paycheck, paystub, tax document or any other document but do not have employer name or not a different employer ignore the bank statement completely.
      - Aggregate daily totals:
        - For each day, sum all qualifying income deposits to compute that day's total gross income.
      - Group daily totals into pay periods by examining the spacing and apply any one of these:
        - Clusters every ~7 days → weekly
        - Clusters every ~14 days → bi-weekly
        - Clusters around the 1st and 15th → semi-monthly
        - Clusters roughly every 25–32 days → monthly
      - Create “period checks”:
        - For each weekly, bi-weekly, semi-monthly, or monthly group, sum all daily totals within that group.
        - Select the three most recent period totals as the three_consecutive_checks.
        - Format each as "MM-DD-YY: amount"
        - Use the last date of the pay period.
        - Use the gross period total.
      - If fewer than three periods exist:
        - Return only the available ones.
      - If employer or business name is not mentioned, return null. Do not return any check deposits without name of employer or source of income.

    - If deposits represent a traditional pay frequency like weekly, bi-weekly, semi-monthly or monthly from employers:
      - Find possible pay_frequency from the given information.
      - Return MM-DD-YY: amount for each deposits.
      - If employer or business name is not mentioned, return null.
    
    - If the bank statement of an applicant is provided after paycheck, paystub, or taxes but do not have employer name or not contains income from
        a different employer, ignore the bank statement completely.

    - If bank statements do not reflect any income or shows random transfers, return null.

    - For business bank accounts:
      - Ignore opening or beginning balances.
      - Identify deposits clearly tied to business activity (customer payments, payment processors, etc.).
      - Withdrawals or expenses do not affect which deposits are chosen.
      - Select the three most recent qualifying income deposits and format as "MM-DD-YY: amount".

  - For certificates of account balance:
    - Return the DATE and AMOUNT as a single three_consecutive_checks entry.

  - Set three_consecutive_checks to null if none of the above applies.

8) hourly_rate
   - Hourly pay rate if explicitly mentioned, e.g., "15.50".
   - Do not mention, overtime payrate.
   - If there is hourly tip mentioned, add it to the hourly rate and return the value. 
   - If not present or not clear, return null.

9) yearly_salary
   - Annual salary if explicitly stated, e.g., "52,000".
   - For a single paycheck, consider "monthly" if not mentioned.
   - If not present or not clear, return null.

10) pay_frequency
   - Determine the payment frequency from the document
   - Look at dates between paychecks to determine frequency:
     - 7 days apart = "weekly"
     - 14 days apart = "bi-weekly"
     - ~15 days apart = "semi-monthly"
     - 20+ days apart = "monthly"
   - Tax documents are mostly contains yearly records.
   - For a bank statement, find the dates of the checks and use best estimate to return "weekly", "bi-weekly", "semi-monthly" or "monthly".
   - If it is a certificate of account balance, return yearly.
   - If only ONE paycheck is present and no frequency is mentioned, assume "monthly".
   - If only overtime pay is mentioned with no regular pay rate and checks are more than 2 weeks apart, consider it monthly.
   - If only hours is used to calculate pay in the document, it is either "weekly" or "bi-weekly".
   - If three checks are from the same month and year with different dates, it is "weekly".
   - Possible values: "weekly", "bi-weekly", "semi-monthly", "monthly".
   - If frequency cannot be determined, return null.
   
11) average_monthly_income
   - Estimated GROSS average monthly income for this job.
   - If multiple checks/deposits are available, compute a reasonable monthly average based on them.
   - If only a yearly salary is provided, compute monthly as yearly_salary / 12 and return as a dollar amount string (e.g., "4333.33").
   - Always attempt a best-effort estimate if enough information exists; otherwise return null.

12) income_calculation_details
   - Explain how the average_monthly_income was calculated with the actual math shown.
   - Include:
     - The source data used (e.g., "Based on 3 paychecks" or "Based on yearly salary")
     - The pay frequency (e.g., "bi-weekly", "semi-monthly", "monthly", "weekly")
     - The calculation steps (e.g., "(500 + 520 + 510) / 3 paychecks = 510 average per paycheck. Bi-weekly pay: 510 × 26 pay periods / 12 months = 1105/month")
     - For multiple monthly incomes, return the average monthly income.
   - Examples:
     - "Based on yearly salary: 52000 / 12 months = 4333.33/month"
     - "Based on 3 bi-weekly paychecks: (1200 + 1250 + 1200) / 3 = 1216.67 average. 1216.67 × 26 pay periods / 12 months = 2636.12/month"
     - "Based on 3 semi-monthly paychecks: (2000 + 2100 + 2050) / 3 = 2050 average. Semi-monthly (24 pay periods): 2050 × 2 = 4100/month"
     - "Based on hourly rate: 15.50/hour × 40 hours/week × 52 weeks / 12 months = 2686.67/month"
     - "Based on 3 weekly paychecks: (1092 + 1092 + 1092) / 3 = 1092 average. Weekly pay: 1092 × 52 weeks / 12 months = 4732/month"
     - If calculation cannot be done, return null.

 13) bank_balances [FOR BANK STATEMENTS ONLY]
     - Return the bank balances or ending balances and net/total deposits provided in each of the bank statements in format MM-DD-YY: net/total deposit, ending balance. 
     - Net/total deposits: You should not add the deposits. It should be mentioned in the document.
     - There can be multiple bank statements from different accounts or time periods or banks.
       For example, a documents primary/saving and checking accounts. Both have net deposits of $3000 and $6000 and ending balances of $1000 and $4000 respectively.
     - If it is a joint bank account, then return bank_balances once in the first applicant but keep the other fields and calculations for individual applicants.
     - If there are multiple applicants with multiple bank statements, bank_balances should be added only to the respective applicants with matching names.
     - If net/total deposit or ending balance do not exist, return null. 

 14) currency
     - Return the currency name mentioned in the document. For US dollar, return "USD". 

CRITICAL CONSISTENCY RULE:
- The final number in "income_calculation_details" MUST EXACTLY MATCH the value in "average_monthly_income"
- Leave three_consecutive_checks empty if there are no names of the employer in bank statements.
- Example: If calculation shows "4732/month", then average_monthly_income must be "4732"
- Double-check your math before outputting the JSON
- DO NOT divide or multiply the final result by any additional factors

OUTPUT FORMAT (STRICT)

Return ONLY a valid JSON object with this structure (no additional text):

{{
  "applicants": [
    {{
      "applicant": "Applicant Name",
      "jobs": [
        {{
          "document_type": "paystub",
          "employer": "Employer Name",
          "position/title": "Job Title or 'owner' or 'self-employed'",
          "employer_address": "Employer address or null",
          "supervisor_name": "Supervisor name or null",
          "three_consecutive_checks": [
            "11-02-25: 500",
            "11-16-25: 500",
            "11-30-25: 500"
          ],
          "hourly_rate": "15.50",
          "yearly_salary": "52000",
          "pay_frequency": "bi-weekly",
          "average_monthly_income": "4333.33",
          "income_calculation_details": "Based on yearly salary: 52000 / 12 months = 4333.33/month",
          "bank_balances": [
            "10-10-24: 5000, 1000",
            "04-16-25: 15000, 6000",
            "11-30-25: 500, 8000"
          ],
          "currency":"USD"
        }}
      ]
    }}
  ]
}}

ADDITIONAL RULES
- If there are multiple applicants, include each one as a separate object in the "applicants" array.
- If an applicant has multiple jobs, include each job as a separate object in that applicant's "jobs" array.
- Use null for any field that cannot be confidently determined.
- Normalize all currency values as strings with no currency sign and comma but decimals where appropriate.
- Normalize dates to "MM-DD-YY" format whenever possible.
- Do NOT include any explanation outside the JSON.
- ALWAYS show the complete calculation in income_calculation_details when average_monthly_income is provided.
- CRITICAL: Always use GROSS income (before deductions), not NET income (after deductions).
"""

    try:
        response = client.chat.completions.create(  
            model=deployment_name,  
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting employment and income data from documents. Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=4000,
        )
        
        if isinstance(response, str):
            logger.error(f"Azure OpenAI returned error string: {response}")
            raise ValueError(f"Azure OpenAI API error: {response}")
        
        if not hasattr(response, 'choices'):
            logger.error(f"Response has no 'choices' attribute. Response: {response}")
            raise ValueError(f"Invalid Azure OpenAI response format: {type(response)}")
        
        if not response.choices:
            logger.error(f"Response choices is empty: {response}")
            raise ValueError("Azure OpenAI returned empty choices array")
        
        result_text = response.choices[0].message.content.strip()
        logger.info("Successfully received GPT response")
        
    except AttributeError as ae:
        logger.error(f"AttributeError accessing response: {ae}")
        logger.error(f"Response object: {response if 'response' in locals() else 'No response variable'}")
        raise ValueError(f"Azure OpenAI response format error: {str(ae)}")
        
    except Exception as e:
        logger.error(f"Error calling Azure OpenAI: {e}")
        logger.error(f"Error type: {type(e)}")
        raise ValueError(f"Failed to get structured data from GPT: {str(e)}")

    try:
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

        extracted_data = json.loads(result_text)
        logger.info("Successfully parsed GPT response")
        return extracted_data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GPT response as JSON: {e}")
        logger.error(f"Raw response (first 1000 chars): {result_text[:1000]}")
        raise ValueError(f"GPT returned invalid JSON: {str(e)}")