#This script, and all other contents of this repo, is NOT associated with SAM.gov or SBIR.gov or grants.gov. It is a privately made enterprise solution! 

import streamlit as st
import requests
import re
import time
import fitz  # PyMuPDF
from info import openai_api_key
import openai
import json
from bs4 import BeautifulSoup
import io
import urllib.request
import gspread
from google.oauth2.service_account import Credentials

# ------------------------------------------------------------
# Google Sheets Export Function
# ------------------------------------------------------------
def export_to_gsheet(data, sheet_name="Government Opportunities", worksheet_name="SAM.gov"):
    # Authenticate with the service account
    creds = Credentials.from_service_account_file(
        "service_account.json",  # path to your downloaded JSON
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    gc = gspread.authorize(creds)
    
    try:
        sh = gc.open(sheet_name)
    except Exception as e:
        raise Exception(f"Could not open sheet '{sheet_name}': {e}")
    
    try:
        worksheet = sh.worksheet(worksheet_name)
    except Exception as e:
        # List available worksheets to help debug
        available_worksheets = [ws.title for ws in sh.worksheets()]
        raise Exception(f"Could not find worksheet '{worksheet_name}'. Available worksheets: {available_worksheets}")

    if not data:
        return
    
    headers = list(data[0].keys())
    
    # Check if worksheet has any data
    try:
        existing_data = worksheet.get_all_values()
        if not existing_data:
            # Completely empty sheet - add headers and data
            worksheet.append_row(headers)
            for row in data:
                row_values = [row.get(h, "") for h in headers]
                worksheet.append_row(row_values)
        else:
            # Sheet has data - check if headers match
            existing_headers = existing_data[0] if existing_data else []
            if existing_headers != headers:
                # Headers don't match - this might be a different data structure
                # Find the first completely empty row to add our headers
                empty_row_idx = len(existing_data) + 2  # Add some space
                end_col = chr(ord('A') + len(headers) - 1)
                worksheet.update(f"A{empty_row_idx}:{end_col}{empty_row_idx}", [headers])
            
            # Append all new data rows regardless of header match
            for row in data:
                row_values = [row.get(h, "") for h in headers]
                worksheet.append_row(row_values)
                
    except Exception as e:
        # If we can't read existing data, just append (safest option)
        for row in data:
            row_values = [row.get(h, "") for h in headers]
            worksheet.append_row(row_values)

# ------------------------------------------------------------
# DATA PARSER FOR GRANTS.GOV 
# ------------------------------------------------------------

if "export_rows_samgov" not in st.session_state:
    st.session_state["export_rows_samgov"] = []

if "export_rows_sbirgov" not in st.session_state:
    st.session_state["export_rows_sbirgov"] = []

if "export_rows_grantsgov" not in st.session_state:
    st.session_state["export_rows_grantsgov"] = []

if "export_rows_random" not in st.session_state:
    st.session_state["export_rows_random"] = []

def _s(x):
    """Return a string no matter what the raw value is."""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return "" if x is None else str(x)

def compose_text_blob(d: dict) -> str:
    """Normalise Grants.gov JSON → plain text blob for GrantBuddy."""
    # CFDA normalisation
    cfda_raw = d.get("cfdaList") or d.get("cfda")
    if isinstance(cfda_raw, list):
        cfda = ", ".join(_s(item.get("cfdaNumber", item)) if isinstance(item, dict) else _s(item)
                         for item in cfda_raw)
    else:
        cfda = _s(cfda_raw)

    # Synopsis normalisation
    syn_raw = d.get("synopsis") or d.get("synopsisDesc") or d.get("synopsisText")
    syn = _s(syn_raw)

    lines = [
        f"Title: {_s(d.get('opportunityTitle'))}",
        f"Opportunity Number: {_s(d.get('opportunityNumber'))}",
        f"Agency: {_s(d.get('agencyName'))}",
        f"CFDA: {cfda}",
        f"Close Date: {_s(d.get('closeDate'))}",
        "",
        "Synopsis:",
        syn
    ]
    return "\n".join(lines)
# ------------------------------------------------------------
# 🧠 GPT Summarization and Extraction Logic (No Changes Here)
# ------------------------------------------------------------

def summarize_text(text):
    """Always send a *string* to OpenAI and return the summary text."""
    openai.api_key = openai_api_key
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)

    # Preprocess text to reduce tokens and avoid rate limits
    processed_text = preprocess_text_for_gpt(text, max_chars=3500)

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that summarizes federal grant solicitations "
                        "or RFIs in 200 words. Focus on clearly explaining WHAT the government wants "
                        "companies to provide or develop, using plain language that a non-technical "
                        "person could understand. Include:\n"
                        "1. What specific products, services, or capabilities they're seeking\n"
                        "2. The purpose or problem they're trying to solve\n"
                        "3. Key requirements or deliverables expected\n"
                        "4. Target applications or use cases\n\n"
                        "Refer to {info.py} for company context and add a line on suitability for our company. "
                        "- Unusual geographic or demographic limitations\n"
                        "Do NOT warn about obvious things like government approval processes or standard procurement rules."
                    ),
                },
                {"role": "user", "content": processed_text},
            ],
            temperature=0.4,
            max_tokens=500,
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Summarization failed: {e}"

# ------------------------------------------------------------
# 📝 Text Preprocessing for Token Reduction
# ------------------------------------------------------------

def preprocess_text_for_gpt(text: str, max_chars: int = 4000) -> str:
    """
    Intelligently extract relevant sections from long documents to reduce tokens.
    More conservative approach - keeps more content but removes obvious fluff.
    """
    if len(text) <= max_chars:
        return text
    
    # For very long documents, use a smarter strategy
    # First, try to find and prioritize key sections
    lines = text.split('\n')
    
    # Very obvious things to skip (web page artifacts)
    skip_patterns = [
        'cookie', 'javascript', 'privacy policy', 'terms of service',
        'accessibility', 'site map', 'navigation', 'footer', 'header',
        'click here', 'login', 'sign up', 'subscribe', 'download pdf'
    ]
    
    # Priority content indicators
    priority_indicators = [
        'naics', 'solicitation', 'opportunity', 'contract', 'cso', 'rfi', 'rfp',
        'deadline', 'due date', 'submission', 'requirement', 'scope', 'award',
        'agency', 'department', 'air force', 'army', 'navy', 'marine', 'coast guard',
        'fa8600', 'w912', 'n00024', 'project', 'title', 'description', 'contact',
        'evaluation', 'criteria', 'funding', 'budget', 'deliverable', 'performance',
        'solution brief', 'technical', 'management', 'cover page', 'format', 'font',
        'page limit', 'ms word', 'times new roman', 'section', 'submittal',
        'sla', 'service level', 'agreement', 'offeror', 'contractor', 'metrics',
        'performance management', 'auditing', 'remedies', 'monitoring'
    ]
    
    # Keep track of content sections
    high_priority = []
    medium_priority = []
    
    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()
        
        # Skip obviously irrelevant lines
        if any(skip in line_lower for skip in skip_patterns):
            continue
            
        # Skip very short lines that are likely navigation
        if len(line_clean) < 10 and not any(char.isdigit() for char in line_clean):
            continue
            
        # High priority: lines with multiple indicators or key patterns
        priority_count = sum(1 for indicator in priority_indicators if indicator in line_lower)
        
        # Specific high-value patterns
        high_value_patterns = [
            'commercial solutions opening number', 'commercial solutions opening title',
            'solicitation number', 'opportunity number', 'contract number', 
            'cso title', 'project title', 'titled "', 'naics code',
            'section 11', 'solution brief', 'cover page', 'technical/management',
            '11.1', '11.2', '11.3', '11.4', 'format', 'font shall be',
            'solicitation no', 'a.3', 'performance management', 'service level agreement',
            'offeror shall', 'contractor will', 'requirements', 'deliverables'
        ]
        
        if priority_count >= 2 or any(pattern in line_lower for pattern in high_value_patterns):
            high_priority.append(line_clean)
        elif priority_count >= 1 or len(line_clean) > 50:  # Longer lines are usually content
            medium_priority.append(line_clean)
    
    # Build result prioritizing high-value content
    result_lines = []
    
    # Always include high priority content
    for line in high_priority:
        result_lines.append(line)
        if len('\n'.join(result_lines)) > max_chars * 0.6:  # Use 60% for high priority
            break
    
    # Fill remaining space with medium priority content
    remaining_chars = max_chars - len('\n'.join(result_lines))
    for line in medium_priority:
        if len('\n'.join(result_lines + [line])) <= max_chars:
            result_lines.append(line)
        else:
            break
    
    # If we still don't have enough content, just take the beginning
    if len(result_lines) < 10:  # If we filtered too aggressively
        return text[:max_chars] + "\n\n[Text truncated to fit token limits]"
    
    result = '\n'.join(result_lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n[Text truncated to fit token limits]"
    
    return result

# ------------------------------------------------------------
# 📋 Application Checklist Extraction with GPT
# ------------------------------------------------------------

def extract_application_checklist(text: str) -> str:
    """
    Extract application requirements and create a comprehensive checklist from opportunity text.
    Uses GPT to thoroughly analyze and prioritize requirements.
    """
    openai.api_key = openai_api_key
    
    # Preprocess text to reduce tokens, but allow more for detailed analysis
    processed_text = preprocess_text_for_gpt(text, max_chars=6000)
    
    prompt = (
        "Extract the specific requirements from this government solicitation document.\n"
        "Focus on what the grants person needs to know to prepare a response.\n"
        "Only include what's actually stated in the document.\n\n"
        "Organize into simple sections:\n\n"
        "**WHAT TO SUBMIT:**\n"
        "- List the specific documents/sections required\n"
        "- Include page limits, format requirements, content requirements\n\n"
        "**ADMINISTRATIVE REQUIREMENTS:**\n"
        "- Registration requirements (SAM.gov, CAGE, etc.)\n"
        "- Required forms and certifications\n"
        "- Submission deadlines and methods\n\n"
        "**EVALUATION CRITERIA:**\n"
        "- How proposals will be scored/evaluated\n"
        "- Key factors they're looking for\n\n"
        "For each item, include the section/page reference if available.\n"
        "If a section has no requirements, write 'N/A' for that section.\n"
        "Be concise and practical - this is for someone preparing a proposal.\n\n"
        f"Document text:\n{processed_text}"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )["choices"][0]["message"]["content"].strip()
        
        if not resp or resp.lower() in ["no requirements found", "no specific requirements mentioned"]:
            # Try a second approach if first fails
            fallback_prompt = (
                "Extract the key requirements from this government document.\n"
                "What does someone need to submit and how?\n\n"
                "Look for:\n"
                "- Required documents (solution brief, proposals, forms)\n"
                "- Format requirements (page limits, font, file type)\n"
                "- Submission process and deadlines\n"
                "- Any registration or certification requirements\n\n"
                "Keep it simple and practical. Include section references if available.\n\n"
                f"Document text:\n{processed_text[:3000]}"
            )
            
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": fallback_prompt}],
                temperature=0.2,
                max_tokens=300,
            )["choices"][0]["message"]["content"].strip()
        
        return resp if resp else "• Review solicitation for specific submission requirements\n• Ensure SAM.gov registration is current\n• Prepare technical and cost proposals as specified"
    except Exception:
        return "• Review solicitation for specific submission requirements\n• Ensure SAM.gov registration is current\n• Prepare technical and cost proposals as specified"

# ------------------------------------------------------------
# 🎯 Dynamic Title Extraction with GPT
# ------------------------------------------------------------

def extract_title_with_gpt(text: str) -> str:
    """
    Use GPT to dynamically extract the project title from text.
    More reliable than structured extraction for complex documents.
    """
    openai.api_key = openai_api_key
    
    # Preprocess text to reduce tokens
    processed_text = preprocess_text_for_gpt(text, max_chars=3000)
    
    prompt = (
        "Extract the project title from this government solicitation text.\n"
        "Look specifically for:\n"
        "- Text in quotes after 'titled' (e.g. titled \"Project Taj Majal – CSO Call 7\")\n"
        "- Project names mentioned after 'Commercial Solutions Opening Title:'\n"
        "- Main opportunity or project names at the top of the document\n"
        "- Titles near 'Solicitation No.' or document headers\n"
        "- Any prominent project/opportunity names in the first few paragraphs\n\n"
        "Return ONLY the title text, nothing else. If you find 'titled \"Project Taj Majal – CSO Call 7\"', return exactly: Project Taj Majal – CSO Call 7\n\n"
        f"Text to analyze:\n{processed_text}"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )["choices"][0]["message"]["content"].strip()
        
        # Clean up common GPT responses
        if resp.lower() in ["not found", "not specified", "n/a", "", "none", "no title found"]:
            return "N/A"
        
        return resp
    except Exception:
        return "N/A"

# ------------------------------------------------------------
# 💰 General Funding Amount Extraction with GPT (for SAM.gov and SBIR.gov)
# ------------------------------------------------------------

def extract_funding_amount(text: str) -> str:
    """
    Use GPT to extract funding/award amounts from opportunity text.
    Returns a clean string like "$500K - $2M" or "Not specified"
    """
    openai.api_key = openai_api_key
    prompt = (
        "You are a funding amount extraction engine.\n"
        "Extract the funding/award amount from this opportunity text.\n"
        "Look for terms like: award amount, funding, budget, contract value, maximum award, etc.\n"
        "Return ONLY the funding amount in a clean format like '$500K', '$1M - $5M', or 'Not specified'.\n"
        "Do not include any other text or explanation.\n\n"
        f"Text to analyze:\n{text[:4000]}"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )["choices"][0]["message"]["content"].strip()
        
        # Clean up common GPT responses
        if resp.lower() in ["not specified", "not mentioned", "n/a", "", "none"]:
            return "Not specified"
        
        return resp
    except Exception:
        return "Not specified"

# ------------------------------------------------------------
# 🧠 Metadata Extraction Helpers (SAM.gov-aware)
# ------------------------------------------------------------
import re, ast, openai
from datetime import datetime

# ------------------------------------------------------------
# 🧾 SAM.gov metadata helpers  ➜  REPLACE the two functions
# ------------------------------------------------------------
import ast, json, re, openai
from datetime import datetime

SERVICE_HINTS = (
    r"Air\s*Force|Army|Navy|Marine\s*Corps|Coast\s*Guard|Space\s*Force|"
    r"Department\s+of\s+the\s+Air\s*Force|Department\s+of\s+the\s+Army|"
    r"Department\s+of\s+the\s+Navy|Department\s+of\s+Defense"
)

# ------------------------------------------------------------
# ⚡ NEW – "rough_sam_regex_extract"  (replace the old one)
# ------------------------------------------------------------
def rough_sam_regex_extract(text: str) -> dict:
    """
    Ultra-fast regex fallback for SAM-gov-style notices.
    Tries to pull Opportunity-ID, Branch, Due-Date, and Title even if
    the GPT extractor fails or you're offline.
    """
    out = {"Opportunity ID": "N/A", "Branch": "N/A", "Due Date": "N/A", "Title": "N/A"}

    # ---------- Opportunity / Notice ID ---------------------------------
    # First try Commercial Solutions Opening Number
    m = re.search(r"Commercial\s+Solutions\s+Opening\s+Number:\s*([A-Z0-9\-_]+)", text, re.I)
    if m:
        out["Opportunity ID"] = m.group(1).strip()
    else:
        # Try Solicitation No.
        m = re.search(r"Solicitation\s+No\.?\s*:?\s*([A-Z0-9\-_]+)", text, re.I)
        if m:
            out["Opportunity ID"] = m.group(1).strip()
        else:
            # Fall back to other patterns
            m = re.search(r"(?:Notice|Solicitation)\s*ID:\s*([A-Z0-9\-_]+)", text, re.I)
            if m:
                out["Opportunity ID"] = m.group(1).strip()

    # ---------- Title extraction ---------------------------------
    # Step 1: Find "Opening Title" line
    opening_title_match = re.search(r'.*Opening Title.*', text, re.I)
    if opening_title_match:
        opening_title_line = opening_title_match.group(0)
        # Step 2: Look for "titled "..."" in that line or nearby text
        titled_match = re.search(r'titled\s*["\']([^"\']+)["\']', opening_title_line, re.I)
        if titled_match:
            out["Title"] = titled_match.group(1).strip()
        else:
            # Look in a broader context around the Opening Title line
            start_pos = max(0, opening_title_match.start() - 200)
            end_pos = min(len(text), opening_title_match.end() + 200)
            context = text[start_pos:end_pos]
            titled_match = re.search(r'titled\s*["\']([^"\']+)["\']', context, re.I)
            if titled_match:
                out["Title"] = titled_match.group(1).strip()
    else:
        # Fallback: search anywhere for "titled "...""
        titled_match = re.search(r'titled\s*["\']([^"\']+)["\']', text, re.I)
        if titled_match:
            out["Title"] = titled_match.group(1).strip()

    # ---------- Branch (try label ➜ frequency fallback) -----------------
    for pat in [
        r"Major Command:\s*(.+)",                       # e.g. "AIR FORCE …"
        r"Department/(?:Ind\.\s*)?Agency:\s*(.+)",      # generic SAM label
        r"Issued by:\s*(.+)",                           # legacy label
    ]:
        m = re.search(pat, text, re.I)
        if m:
            out["Branch"] = m.group(1).strip()
            break

    # ✨ NEW fallback: pick the service mentioned most often
    if out["Branch"] in {"N/A", ""}:
        CANDIDATES = [
            "Air Force", "Army", "Navy", "Marine Corps",
            "Space Force", "Coast Guard", "Department of Defense",
            "Department of the Air Force", "Department of the Army",
        ]
        counts = {
            b: len(re.findall(rf"\b{re.escape(b)}\b", text, re.I))
            for b in CANDIDATES
        }
        best, best_cnt = max(counts.items(), key=lambda kv: kv[1])
        if best_cnt:                           # at least one hit
            out["Branch"] = best

    # ---------- Due / Response / Closing Date ---------------------------
    m = re.search(
        r"(?:Original|Updated)?\s*"
        r"(Response(?:\s+Due)?|Close|Closing|Offers\s+Due)\s*(?:Date)?:\s*"
        r"([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        text,
        re.I,
    )
    if m:
        out["Due Date"] = m.group(2).strip()
    else:
        # -- Fallback: first "Month DD, YYYY" found anywhere
        m = re.search(
            r"(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},\s+\d{4}",
            text,
        )
        if m:
            out["Due Date"] = m.group(0)

    return out


def extract_fields_with_gpt(text: str) -> dict:
    """
    Ask GPT-4 for the three fields + merge any missing ones from the rough regex
    so the cards are always filled if *either* path succeeds.
    """
    openai.api_key = openai_api_key
    # Preprocess text to reduce tokens
    processed_text = preprocess_text_for_gpt(text, max_chars=2500)
    
    prompt = (
        "You are a metadata extraction engine.\n"
        "Return ONLY a Python dict (no prose) inside a ``` code-block.\n"
        "Fields:\n"
        "- 'Opportunity ID': Look for 'Commercial Solutions Opening Number:', 'Solicitation Number:', 'Solicitation No.:', 'Notice ID:', 'Contract Number:', or similar labels followed by alphanumeric codes (e.g. 'FA8600-23-S-C056').\n"
        "- 'Branch': Look for military branches (Air Force, Army, Navy, etc.) or government agencies. Check for patterns like 'Department of Air Force', 'AFLCMC', etc.\n"
        "- 'Due Date': Look for deadlines, response dates, or closing dates in any format.\n"
        "- 'Title': CRITICAL - Extract the exact project title from quoted text. Look for:\n"
        "  * 'titled \"Project Taj Majal – CSO Call 7\"' → return exactly 'Project Taj Majal – CSO Call 7'\n"
        "  * Search the ENTIRE text for any quoted project names or titles\n"
        "  * Look after 'Commercial Solutions Opening Title:' for quoted content\n"
        "  * Return the most specific quoted title, not generic descriptions\n"
        "  * If no quotes found, extract the main project name mentioned\n\n"
        "Text to analyse:\n"
        f"{processed_text}"
    )

    gpt_meta = {}
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=350,
        )["choices"][0]["message"]["content"]

        m = re.search(r"```(?:json)?\s*\n(.*?)\n```", resp, re.S)
        if m:
            resp = m.group(1)
        gpt_meta = ast.literal_eval(resp)
    except Exception:
        gpt_meta = {}

    # --- merge with regex fallback ---------------------------------------
    regex_meta = rough_sam_regex_extract(text)
    for k in ("Opportunity ID", "Branch", "Due Date"):
        v = gpt_meta.get(k, "") if isinstance(gpt_meta, dict) else ""
        if not v or v in {"N/A", ""}:
            gpt_meta[k] = regex_meta.get(k, "N/A")
    
    # For Title, use dedicated dynamic GPT extraction as primary method
    title_from_dynamic_gpt = extract_title_with_gpt(text)
    title_from_structured_gpt = gpt_meta.get("Title", "") if isinstance(gpt_meta, dict) else ""
    title_from_regex = regex_meta.get("Title", "N/A")
    
    if title_from_dynamic_gpt and title_from_dynamic_gpt != "N/A":
        gpt_meta["Title"] = title_from_dynamic_gpt
    elif title_from_regex and title_from_regex != "N/A":
        gpt_meta["Title"] = title_from_regex
    elif title_from_structured_gpt and title_from_structured_gpt not in {"N/A", ""}:
        gpt_meta["Title"] = title_from_structured_gpt
    else:
        gpt_meta["Title"] = "N/A"

    # final safety net
    return {k: gpt_meta.get(k, "N/A") for k in ("Opportunity ID", "Branch", "Due Date", "Title")}

# ------------------------------------------------------------
# 📎 PDF Text Extraction (No Changes Here)
# ------------------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    try:
        # For uploaded files, the stream is already in bytes
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        return f"PDF extraction failed: {e}"

# Helper – pull first Month DD YYYY date out of plain text
def first_long_date(txt: str) -> str:
    """"withdraw from text. responseDateStr": "2025-09-29-00-00-00" or ''."""
    pat = (r"(January|February|March|April|May|June|July|August|"
           r"September|October|November|December)\s+\d{1,2},?\s+\d{4}")
    m = re.search(pat, txt)
    return m.group(0) if m else ""

def parse_sam_structured_text(text: str) -> dict:
    """
    Parse structured SAM.gov text from Tampermonkey script format:
    Title: [title]
    Notice ID: [id]
    Agency: [agency]
    Response Date: [date]
    Source URL: [url]
    Description: [description]
    """
    result = {
        "Title": "N/A",
        "Opportunity ID": "N/A", 
        "Agency": "N/A",
        "Due Date": "N/A",
        "Source URL": "N/A"
    }
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("Title:"):
            result["Title"] = line.replace("Title:", "").strip()
        elif line.startswith("Notice ID:"):
            result["Opportunity ID"] = line.replace("Notice ID:", "").strip()
        elif line.startswith("Agency:"):
            result["Agency"] = line.replace("Agency:", "").strip()
        elif line.startswith("Response Date:"):
            result["Due Date"] = line.replace("Response Date:", "").strip()
        elif line.startswith("Source URL:"):
            result["Source URL"] = line.replace("Source URL:", "").strip()
    
    return result

# ============================================================
# FULL REPLACEMENT — display_analysis_results()
# ============================================================
from bs4 import BeautifulSoup
import json, textwrap, re

from bs4 import BeautifulSoup
import textwrap, json, re

def display_analysis_results(blob):
    """
    Accepts either Grants.gov JSON or a plain-text blob.
    Shows Branch, Due Date, Title, and a GPT summary.
    """
    if blob is None or (isinstance(blob, str) and not blob.strip()):
        st.warning("Nothing to analyze."); return

    if isinstance(blob, dict):
        # -------- Branch (name + code with sensible fall-backs) ----------
        name = next(filter(None, [
            blob.get("agencyName"),
            blob.get("agencyContactName"),
            (blob.get("topAgencyDetails") or {}).get("agencyName")
        ]), "N/A").strip()

        code = next(filter(None, [
            blob.get("agencyCode"),
            (blob.get("topAgencyDetails") or {}).get("agencyCode")
        ]), "").strip().upper()

        branch = f"{name} ({code})" if code else name

        # -------- Synopsis (plain text) ----------------------------------
        raw_syn   = blob.get("synopsis") or blob.get("synopsisDesc") or ""
        syn_plain = BeautifulSoup(str(raw_syn), "html.parser").get_text(" ", strip=True)

        # ---------- Due date ---------------------------------------------
        due_for_card = blob.get("closeDate") or blob.get("responseDate") or blob.get("responseDateStr") or "N/A"
        # Try to humanize if possible
        try:
            due_for_card = human_date(due_for_card)
        except Exception:
            pass

        # ---------- Four-column info card -------------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.caption("🧾 Opportunity #"); c1.code(blob.get("opportunityNumber", "N/A"))
        c2.caption("🏛️ Branch");        c2.code(branch)
        c3.caption("📅 Due Date");       c3.code(due_for_card)
        c4.caption("📝 Title");          c4.code(blob.get("opportunityTitle", "N/A"))

        # ---------- Budget / eligibility cues ------------------------------
        cues = []
        lo, hi = blob.get("awardFloor"), blob.get("awardCeiling")
        if lo and hi:
            cues.append(f"NOTE: awards roughly ${int(lo):,} – ${int(hi):,}.")
        elig_desc = (
            blob.get("applicantEligibilityDesc", "") +
            json.dumps(blob.get("applicantTypes", []))
        ).lower()
        if not any(k in elig_desc for k in ["small business", "for-profit", "startup"]):
            cues.append("FIT-ALERT: Eligibility omits small businesses / startups.")

        # ---------- build the prompt for GPT --------------------------
        synopsis_for_gpt = (
            f"The deadline for applications is {due_for_card}.\n\n{syn_plain}"
        )
        full_for_gpt = (
            ("\n".join(cues) + "\n\n") if cues else ""
        ) + synopsis_for_gpt

        with st.spinner("Summarizing…"):
            summary = summarize_text(full_for_gpt)

        st.markdown("### 🧠 AI Summary")
        st.info(re.sub(r"\s+", " ", summary).strip())

        # Extract and display checklist in collapsible section
        checklist_text = extract_application_checklist(full_for_gpt)
        with st.expander("📋 Application Checklist"):
            st.write(checklist_text)

    elif isinstance(blob, str):
        # Handle plain text: extract metadata and show summary
        meta = extract_fields_with_gpt(blob)
        col1, col2, col3 = st.columns(3)
        col1.caption("🧾 Opportunity ID"); col1.text(meta.get("Opportunity ID", "N/A"))
        col2.caption("🏛️ Branch");        col2.text(meta.get("Branch", "N/A"))
        col3.caption("📅 Due Date");       col3.text(meta.get("Due Date", "N/A"))

        st.markdown("### 🧠 AI Summary")
        st.info(summarize_text(blob))

        # Extract and display checklist in collapsible section
        checklist_text = extract_application_checklist(blob)
        with st.expander("📋 Application Checklist"):
            st.write(checklist_text)

# ============================================================

st.set_page_config(page_title="GrantBuddy – Federal Funding Analyzer")

# Custom CSS to set font to Be Vietnam Pro
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif !important;
}

.stApp {
    font-family: 'Be Vietnam Pro', sans-serif !important;
}

/* Ensure all text elements use the font */
h1, h2, h3, h4, h5, h6, p, div, span, label, button, input, textarea, select {
    font-family: 'Be Vietnam Pro', sans-serif !important;
}

/* Streamlit specific elements */
.stMarkdown, .stText, .stCaption, .stCode, .stButton, .stTextInput, .stTextArea, .stSelectbox {
    font-family: 'Be Vietnam Pro', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 GrantBuddy – Federal Funding Analyzer")

# Get source and text from URL for Tampermonkey integration
query_params = st.query_params
source = query_params.get("source", "").lower()
prefilled_text = query_params.get("text", "")

# Create the tabs
tab_map = {
    "samgov": "📄 SAM.gov",
    "sbirgov": "🔬 SBIR.gov",
    "grantsgov": "💰 Grants.gov",
    "random": "📁 Random"
}
default_tab = st.query_params.get("tab", "samgov").lower()
selected_label = tab_map.get(default_tab, "📄 SAM.gov")

selected_tab = st.radio("Select a tab:", list(tab_map.values()), horizontal=True, index=list(tab_map.values()).index(selected_label))

# --- SAM.gov Tab ---
# --- 📄 SAM.gov TAB --------------------------------------------------------
if selected_tab == "📄 SAM.gov":
    st.header("📄 SAM.gov Opportunity Analyzer")
    
    # Instructions for users
    st.info("💡 **How to use:** To analyze opportunities, click the link below and search your desired keyword. Once you have clicked on a suitable opportunity, click the \"Send to GrantBuddy\" button in the bottom left corner.")
    
    st.link_button("🔗 Open SAM.gov", "https://sam.gov/")

    # ─────────────────────────────────────────────────────────────────────
    # 1️⃣ Optional ?pdf= URL parameter (for Tampermonkey-style deep links)
    # ─────────────────────────────────────────────────────────────────────
    pdf_url = st.query_params.get("pdf")
    if pdf_url:
        try:
            with st.spinner("📥 Downloading PDF from link…"):
                pdf_bytes = urllib.request.urlopen(pdf_url, timeout=15).read()
                text_from_url_pdf = extract_text_from_pdf(io.BytesIO(pdf_bytes))
            st.success("✅ PDF downloaded & text extracted.")
            st.text_area("📄 Extracted PDF Text", text_from_url_pdf, height=300)

            meta = extract_fields_with_gpt(text_from_url_pdf)
            col1, col2, col3 = st.columns(3)
            col1.caption("🧾 Opportunity ID"); col1.text(meta.get("Opportunity ID", "N/A"))
            col2.caption("🏛️ Branch");        col2.text(meta.get("Branch", "N/A"))
            col3.caption("📅 Due Date");       col3.text(meta.get("Due Date", "N/A"))

            st.markdown("### 🧠 AI Summary")
            st.info(summarize_text(text_from_url_pdf))

            # Extract and display checklist in collapsible section
            checklist_text = extract_application_checklist(text_from_url_pdf)
            with st.expander("📋 Application Checklist"):
                st.write(checklist_text)
        except Exception as e:
            st.error(f"PDF download or extraction failed: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # 2️⃣ Manual text-paste workflow
    #    (prefill via ?text= query-param when coming from a userscript)
    # ─────────────────────────────────────────────────────────────────────
    sam_prefill = st.query_params.get("text", "") if source == "samgov" else ""
    sam_text = st.text_area("Paste SAM.gov solicitation text here:",
                            value=sam_prefill, height=300, key="sam_text")

    if st.button("🔍 Analyze SAM Text", key="sam_analyze"):
        if sam_text.strip():
            # Check if this is structured text from Tampermonkey or raw text
            if "Title:" in sam_text and "Notice ID:" in sam_text and "Agency:" in sam_text:
                # Parse structured format from Tampermonkey
                meta = parse_sam_structured_text(sam_text)
            else:
                # Fall back to GPT extraction for raw text
                meta = extract_fields_with_gpt(sam_text)
                # Map "Branch" to "Agency" for consistency
                if "Branch" in meta and "Agency" not in meta:
                    meta["Agency"] = meta["Branch"]
            
            summary = summarize_text(sam_text)
            funding_amount = extract_funding_amount(sam_text)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.caption("🧾 Opportunity ID"); col1.text(meta.get("Opportunity ID", "N/A"))
            col2.caption("🏛️ Agency");        col2.text(meta.get("Agency", "N/A"))
            col3.caption("📅 Due Date");       col3.text(meta.get("Due Date", "N/A"))
            col4.caption("📝 Title");          col4.text(meta.get("Title", "N/A"))
            col5.caption("💰 Funding");        col5.text(funding_amount)

            st.markdown("### 🧠 AI Summary")
            st.info(summary)

            # Extract and display checklist in collapsible section
            checklist_text = extract_application_checklist(sam_text)
            with st.expander("📋 Application Checklist"):
                st.write(checklist_text)

            # Source URL card (collapsible)
            with st.expander("🔗 Source Information"):
                source_url = meta.get("Source URL", "Not provided")
                st.text(f"Source URL: {source_url}")

            st.session_state["export_rows_samgov"].append({
                "Opportunity ID": meta.get("Opportunity ID", "N/A"),
                "Due Date": meta.get("Due Date", "N/A"),
                "Agency": meta.get("Agency", "N/A"),
                "Title": meta.get("Title", "N/A"),
                "Funding Amount": funding_amount,
                "Summary": summary,
                "Application Checklist": checklist_text,
                "Source URL": meta.get("Source URL", "Not provided")
            })
        else:
            st.warning("Please paste some text first.")

    # ─────────────────────────────────────────────────────────────────────
    # 3️⃣ Local PDF upload (auto-collapse if there's text)
    # ─────────────────────────────────────────────────────────────────────
    has_text = sam_text.strip() != ""
    with st.expander("📎 Upload SAM.gov PDF", expanded=not has_text):
        sam_pdf = st.file_uploader("Upload a SAM.gov PDF file", type=["pdf"], key="sam_pdf")
    if sam_pdf:
        with st.spinner("🔍 Extracting text from uploaded PDF…"):
            pdf_text = extract_text_from_pdf(sam_pdf)
        if pdf_text.startswith("PDF extraction failed"):
            st.error(pdf_text)
        else:
            st.success("✅ PDF text extracted.")
            st.text_area("📄 Extracted PDF Text", pdf_text, height=300)

            # Check if this is structured text from Tampermonkey or raw text
            if "Title:" in pdf_text and "Notice ID:" in pdf_text and "Agency:" in pdf_text:
                # Parse structured format from Tampermonkey
                meta = parse_sam_structured_text(pdf_text)
            else:
                # Fall back to GPT extraction for raw text
                meta = extract_fields_with_gpt(pdf_text)
                # Map "Branch" to "Agency" for consistency
                if "Branch" in meta and "Agency" not in meta:
                    meta["Agency"] = meta["Branch"]
            
            summary = summarize_text(pdf_text)
            funding_amount = extract_funding_amount(pdf_text)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.caption("🧾 Opportunity ID"); col1.text(meta.get("Opportunity ID", "N/A"))
            col2.caption("🏛️ Agency");        col2.text(meta.get("Agency", "N/A"))
            col3.caption("📅 Due Date");       col3.text(meta.get("Due Date", "N/A"))
            col4.caption("📝 Title");          col4.text(meta.get("Title", "N/A"))
            col5.caption("💰 Funding");        col5.text(funding_amount)

            st.markdown("### 🧠 AI Summary")
            st.info(summary)

            # Extract and display checklist in collapsible section
            checklist_text = extract_application_checklist(pdf_text)
            with st.expander("📋 Application Checklist"):
                st.write(checklist_text)

            # Source URL card (collapsible)
            with st.expander("🔗 Source Information"):
                source_url = meta.get("Source URL", "Not provided")
                st.text(f"Source URL: {source_url}")

            st.session_state["export_rows_samgov"].append({
                "Opportunity ID": meta.get("Opportunity ID", "N/A"),
                "Due Date": meta.get("Due Date", "N/A"),
                "Agency": meta.get("Agency", "N/A"),
                "Title": meta.get("Title", "N/A"),
                "Funding Amount": funding_amount,
                "Summary": summary,
                "Application Checklist": checklist_text,
                "Source URL": meta.get("Source URL", "Not provided")
            })

    # Export and sheet buttons for SAM.gov
    if st.button("Export SAM.gov results to Google Sheets"):
        sheet_name = st.text_input("Google Sheet Name:", value="Government Opportunities", key="sheet_name_input")
        if sheet_name:
            try:
                export_to_gsheet(st.session_state["export_rows_samgov"], sheet_name=sheet_name)
                st.session_state["export_rows_samgov"] = []  # Optional: clear after export
                st.success(f"Exported to Google Sheets: {sheet_name}!")
            except Exception as e:
                st.error(f"Export failed: {e}")
                st.info("Make sure the sheet exists and is shared with your service account.")
        else:
            st.warning("Please enter a sheet name.")

    # Add "See Your Sheet" button for SAM.gov
    st.link_button("📊 See Your Sheet", "//")

# --------------------------------------------------------------------------
# SBIR.GOV TAB BEGINNING
# --------------------------------------------------------------------------

def extract_sbir_meta(text: str) -> dict:
    """
    Extract SBIR.gov Opportunity ID (e.g., A254-P026), Due Date (second MM/DD/YYYY date), Branch, and Title from text.
    The ID comes first, then the title is the line immediately below it.
    """
    out = {"Opportunity ID": "N/A", "Due Date": "N/A", "Branch": "N/A", "Title": "N/A"}
    
    # Find SBIR ID pattern: letter + 3 digits + dash + letter/P + 3 digits (e.g., A254-P026, A254-019)
    m = re.search(r"\b[A-Z]\d{3}-[A-Z]?\d{3}\b", text)
    if m:
        out["Opportunity ID"] = m.group(0)
    
    # Find all MM/DD/YYYY dates
    dates = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text)
    if len(dates) >= 2:
        out["Due Date"] = dates[1]
    elif dates:
        out["Due Date"] = dates[0]
    
    # Branch extraction: look for known branches
    branches = ["Army", "Navy", "Air Force", "Space Force", "Marine Corps", "Coast Guard", "Department of Defense"]
    for branch in branches:
        if re.search(rf"\b{re.escape(branch)}\b", text, re.I):
            out["Branch"] = branch
            break
    
    # Title extraction: find the line immediately after the ID
    lines = [line.strip() for line in text.splitlines()]
    if out["Opportunity ID"] != "N/A":
        for i, line in enumerate(lines):
            if out["Opportunity ID"] in line:
                # Title is all consecutive non-empty lines after the ID until a blank line
                title_parts = []
                for j in range(i + 1, len(lines)):
                    current_line = lines[j].strip()
                    # Stop if we hit a blank line or a date line
                    if not current_line or re.match(r"\b\d{2}/\d{2}/\d{4}\b", current_line):
                        break
                    title_parts.append(current_line)
                
                if title_parts:
                    out["Title"] = " ".join(title_parts)
                break
    
    # If title is still N/A, try GPT extraction as fallback
    if out["Title"] == "N/A":
        try:
            openai.api_key = openai_api_key
            prompt = (
                "Extract the SBIR opportunity title from this text. Look for:\n"
                "- Solicitation names (e.g. 'xTechPacific Open Topic')\n"
                "- Program titles or topic names\n"
                "- Main opportunity descriptors\n\n"
                "Return ONLY the title, nothing else.\n\n"
                f"Text:\n{text[:2000]}"
            )
            
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
            )["choices"][0]["message"]["content"].strip()
            
            # Clean up the response
            if resp and resp.lower() not in ["not found", "n/a", "none", "not specified"]:
                out["Title"] = resp
        except Exception:
            pass  # Keep N/A if GPT fails
    
    return out

# --- 🔬 SBIR.gov TAB -------------------------------------------------------
if selected_tab == "🔬 SBIR.gov":
    st.header("🔬 SBIR.gov Opportunity Analyzer")
    
    # Instructions for users
    st.info("💡 **How to use:** Click the link below, highlight all the text of your desired opportunity (including title and ID number) and then click the button in the bottom right corner.")
    
    st.link_button(
        "🔗 Open SBIR.gov (DoD portal)",
        "https://www.dodsbirsttr.mil/topics-app/"
    )

    # ── 1️⃣ Manual text input (prefill if coming from a userscript) ────────
    sbir_prefill = prefilled_text if source == "sbirgov" else ""
    sbir_text    = st.text_area(
        "Paste SBIR.gov solicitation text here:",
        value=sbir_prefill,
        height=300,
        key="sbir_text",
    )

    if st.button("🔍 Analyze SBIR Text", key="sbir_run"):
        if sbir_text.strip():
            # SBIR-specific extraction
            sbir_meta = extract_sbir_meta(sbir_text)
            funding_amount = extract_funding_amount(sbir_text)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.caption("🧾 Opportunity ID"); c1.text(sbir_meta.get("Opportunity ID", "N/A"))
            c2.caption("📅 Due Date");       c2.text(sbir_meta.get("Due Date", "N/A"))
            c3.caption("🏛️ Branch");         c3.text(sbir_meta.get("Branch", "N/A"))
            c4.caption("📝 Title");           c4.text(sbir_meta.get("Title", "N/A"))
            c5.caption("💰 Funding");         c5.text(funding_amount)
            summary = summarize_text(sbir_text)
            st.markdown("### 🧠 AI Summary")
            st.info(summary)
            
                        # Extract and display checklist in collapsible section
            checklist_text = extract_application_checklist(sbir_text)
            with st.expander("📋 Application Checklist"):
                st.write(checklist_text)
            
            # Collect for export
            st.session_state["export_rows_sbirgov"].append({
                "Opportunity ID": sbir_meta.get("Opportunity ID", "N/A"),
                "Due Date": sbir_meta.get("Due Date", "N/A"),
                "Branch": sbir_meta.get("Branch", "N/A"),
                "Title": sbir_meta.get("Title", "N/A"),
                "Funding Amount": funding_amount,
                "Summary": summary,
                "Application Checklist": checklist_text,
                "Source URL": "https://www.dodsbirsttr.mil/topics-app/"
            })
        else:
            st.warning("Please paste some text to analyze.")

    # ── 2️⃣ Local PDF upload (optional - auto-collapse if there's text) ───
    has_sbir_text = sbir_text.strip() != ""
    with st.expander("📎 Upload SBIR.gov PDF", expanded=not has_sbir_text):
        sbir_pdf = st.file_uploader(
            "Upload an SBIR.gov PDF file",
            type=["pdf"],
            key="sbir_pdf"
        )
    if sbir_pdf:
        with st.spinner("🔍 Extracting text from PDF…"):
            pdf_text = extract_text_from_pdf(sbir_pdf)

        if pdf_text.startswith("PDF extraction failed"):
            st.error(pdf_text)                        # show PyMuPDF error, if any
        else:
            st.success("✅ PDF text extracted.")
            sbir_meta = extract_sbir_meta(pdf_text)
            funding_amount = extract_funding_amount(pdf_text)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.caption("🧾 Opportunity ID"); c1.text(sbir_meta.get("Opportunity ID", "N/A"))
            c2.caption("📅 Due Date");       c2.text(sbir_meta.get("Due Date", "N/A"))
            c3.caption("🏛️ Branch");         c3.text(sbir_meta.get("Branch", "N/A"))
            c4.caption("📝 Title");           c4.text(sbir_meta.get("Title", "N/A"))
            c5.caption("💰 Funding");         c5.text(funding_amount)
            summary = summarize_text(pdf_text)
            st.markdown("### 🧠 AI Summary")
            st.info(summary)
            
                        # Extract and display checklist in collapsible section
            checklist_text = extract_application_checklist(pdf_text)
            with st.expander("📋 Application Checklist"):
                st.write(checklist_text)
            
            # Collect for export
            st.session_state["export_rows_sbirgov"].append({
                "Opportunity ID": sbir_meta.get("Opportunity ID", "N/A"),
                "Due Date": sbir_meta.get("Due Date", "N/A"),
                "Branch": sbir_meta.get("Branch", "N/A"),
                "Title": sbir_meta.get("Title", "N/A"),
                "Funding Amount": funding_amount,
                "Summary": summary,
                "Application Checklist": checklist_text,
                "Source URL": "https://www.dodsbirsttr.mil/topics-app/"
            })

    # Export button for SBIR.gov
    if st.button("Export SBIR.gov results to Google Sheets"):
        sheet_name = st.text_input("Google Sheet Name:", value="Government Opportunities", key="sbir_sheet_name_input")
        if sheet_name:
            try:
                export_to_gsheet(st.session_state["export_rows_sbirgov"], sheet_name=sheet_name, worksheet_name="SBIR.gov")
                st.session_state["export_rows_sbirgov"] = []  # Optional: clear after export
                st.success(f"Exported to Google Sheets: {sheet_name} (SBIR.gov tab)!")
            except Exception as e:
                st.error(f"Export failed: {e}")
                st.info("Make sure the sheet exists and is shared with your service account.")
        else:
            st.warning("Please enter a sheet name.")

    # Add "See Your Sheet" button for SBIR.gov
    st.link_button("📊 See Your Sheet", "//")

# -------------------------------------------------------------------------

#google sheets integration system 
import re, requests, json
import streamlit as st                      # already at top of your file

# ─── Grants.gov helpers ────────────────────────────────────────────
import requests, streamlit as st, json, re
from urllib.parse import urlparse, parse_qs

def parse_opp_id(url:str)->str|None:
    u = urlparse(url)
    qs = parse_qs(u.query)
    if qs.get("oppId"):                       # …/view-opportunity.html?oppId=###
        return qs["oppId"][0]
    m = re.search(r"/search-results-detail/(\d+)", u.path, re.I)
    return m.group(1) if m else None

import requests, json

def safe_json(resp):
    """Return {} if the body is not valid JSON (HTML, CAPTCHA, etc.)."""
    try:
        return resp.json()
    except ValueError:
        return {}

def fetch_grants_json(opp_id: str, timeout: int = 12) -> dict | None:
    """
    1) Hit GET /search2  (no key, fast).
    2) If that body isn't JSON or is empty, fall back to
       POST /v1/api/fetchOpportunity.
    Never throws JSONDecodeError; returns None on hard miss.
    """
    # ❶ open GET
    r = requests.get(
        "https://www.grants.gov/api/opportunities/v1/search2",
        params={"oppId": opp_id, "maxRecords": 1},
        headers={"Accept": "application/json"},       # a hint for some proxies
        timeout=timeout,
    )
    data = safe_json(r).get("opportunitiesData", [])
    if data:
        return data[0]

    # ❷ fallback POST
    r = requests.post(
        "https://api.grants.gov/v1/api/fetchOpportunity",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"opportunityId": int(opp_id)}),
        timeout=timeout,
    )
    payload = safe_json(r)
    return payload.get("data") or None

# ✂︎ REPLACE the old get_due_date() with ALL of this  ✂︎
from datetime import datetime

DATE_PATTERNS = [
    # 1) ISO 2025-09-29 or 2025-09-29-00-00-00
    (r"\d{4}-\d{2}-\d{2}(?:-\d{2}-\d{2}-\d{2})?", "%Y-%m-%d"),
    # 2) long month  September 29 2025   /   September 29, 2025
    (r"(January|February|March|April|May|June|July|August|"
     r"September|October|November|December)\s+\d{1,2},?\s+\d{4}", "%B %d %Y"),
    # 3) abbreviated month  Sep 29 2025   /   Sep 29, 2025
    (r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
     r"\.?[\s-]+\d{1,2},?\s+\d{4}", "%b %d %Y"),
    # 4) purely numeric  09/29/2025  or  9-29-25
    (r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", None),
]

def _clean_iso(s: str) -> str:
    """Trim trailing -00-00-00 etc."""
    return s[:10] if len(s) >= 10 and s[4] == "-" else s

def get_due_date(j: dict, fallback_text: str = "") -> str:
    """
    Return a human-readable due date from Grants.gov JSON or N/A.

    Priority: closeDate ▸ responseDate ▸ responseDateStr ▸ regex in text.
    All common formats are accepted.
    """
    # ❶ direct JSON fields first
    for key in ("closeDate", "responseDate", "responseDateStr"):
        v = (j.get(key) or "").strip()
        if v:
            return _clean_iso(v)

    # ❷ scan the plain text once and stop at the first match
    for pat, strptime_fmt in DATE_PATTERNS:
        m = re.search(pat, fallback_text, flags=re.I)
        if not m:
            continue
        raw = m.group(0)
        if not strptime_fmt:                    # numeric mm/dd/yy
            return raw
        try:                                    # normalise to YYYY-MM-DD
            dt = datetime.strptime(re.sub(r"[^\w\s]", " ", raw), strptime_fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return raw                           # leave as-is on failure
    return "N/A"

# ─── helper to pretty-print whatever get_due_date() gave us ─────────
from datetime import datetime

def human_date(due_raw: str) -> str:
    """Return 'September 29, 2025' style if we can parse it; else raw."""
    # ISO → Month DD, YYYY
    if re.match(r"\d{4}-\d{2}-\d{2}", due_raw):
        try:
            return datetime.strptime(due_raw[:10], "%Y-%m-%d").strftime("%B %d, %Y")
        except ValueError:
            pass
    # already like "Sep 29, 2025" → normalise long month name
    try:
        return datetime.strptime(due_raw, "%b %d, %Y").strftime("%B %d, %Y")
    except ValueError:
        try:
            return datetime.strptime(due_raw, "%B %d, %Y").strftime("%B %d, %Y")
        except ValueError:
            return due_raw  # fall back: show whatever we got

# ------------------------------------------------------------
# 💰 General Funding Amount Extraction with GPT (for SAM.gov and SBIR.gov)
# ------------------------------------------------------------

def extract_funding_amount(text: str) -> str:
    """
    Use GPT to extract funding/award amounts from opportunity text.
    Returns a clean string like "$500K - $2M" or "Not specified"
    """
    openai.api_key = openai_api_key
    prompt = (
        "You are a funding amount extraction engine.\n"
        "Extract the funding/award amount from this opportunity text.\n"
        "Look for terms like: award amount, funding, budget, contract value, maximum award, etc.\n"
        "Return ONLY the funding amount in a clean format like '$500K', '$1M - $5M', or 'Not specified'.\n"
        "Do not include any other text or explanation.\n\n"
        f"Text to analyze:\n{text[:4000]}"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )["choices"][0]["message"]["content"].strip()
        
        # Clean up common GPT responses
        if resp.lower() in ["not specified", "not mentioned", "n/a", "", "none"]:
            return "Not specified"
        
        return resp
    except Exception:
        return "Not specified"

# ------------------------------------------------------------
# 💰 Grants.gov Specific Funding Amount Extraction with GPT
# ------------------------------------------------------------

def extract_grants_funding_amount(text: str, grants_json: dict = None) -> str:
    """
    Use GPT to extract INDIVIDUAL award amounts for Grants.gov, not total program funding.
    Returns a clean string like "$500K per award" or "Not specified"
    """
    openai.api_key = openai_api_key
    
    # First check if we have structured JSON data with award floor/ceiling
    if grants_json:
        lo, hi = grants_json.get("awardFloor"), grants_json.get("awardCeiling")
        if lo and hi:
            try:
                lo_formatted = f"${int(lo):,}"
                hi_formatted = f"${int(hi):,}"
                if lo == hi:
                    return f"{lo_formatted} per award"
                else:
                    return f"{lo_formatted} - {hi_formatted} per award"
            except (ValueError, TypeError):
                pass
        elif lo:
            try:
                return f"Up to ${int(lo):,} per award"
            except (ValueError, TypeError):
                pass
        elif hi:
            try:
                return f"Up to ${int(hi):,} per award"
            except (ValueError, TypeError):
                pass
    
    # If no structured data, use GPT with specific instructions
    prompt = (
        "You are a funding amount extraction engine for grant opportunities.\n"
        "Extract the INDIVIDUAL AWARD AMOUNT that would go to each applicant/organization, NOT the total program funding.\n"
        "Look specifically for:\n"
        "- Award amount per applicant\n"
        "- Individual grant size\n"
        "- Maximum award per organization\n"
        "- Award range per recipient\n"
        "- Per-project funding\n\n"
        "IGNORE:\n"
        "- Total program funding\n"
        "- Overall budget for the entire program\n"
        "- Total funds available across all awards\n\n"
        "Return ONLY the individual award amount in a clean format like '$500K per award', '$1M - $5M per recipient', or 'Not specified'.\n"
        "Do not include any other text or explanation.\n\n"
        f"Text to analyze:\n{text[:4000]}"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )["choices"][0]["message"]["content"].strip()
        
        # Clean up common GPT responses
        if resp.lower() in ["not specified", "not mentioned", "n/a", "", "none"]:
            return "Not specified"
        
        return resp
    except Exception:
        return "Not specified"

# ------------------------------------------------------------
# ---------------- Grants.gov TAB START -------------------------------------
if selected_tab == "💰 Grants.gov":
    st.header("💰 Grants.gov Opportunity Analyzer")
    
    # Instructions for users
    st.info("💡 **How to use:** Click the below link and then paste in the URL of your desired opportunity below.")
    
    st.link_button("🔗 Open Grants.gov", "https://www.grants.gov/search-grants")

    # session defaults …
    for k, v in {"gjson": None, "gblob": "", "g_show_full": False}.items():
        st.session_state.setdefault(k, v)

    # 1️⃣ URL input + fetch --------------------------------------------------
    url = st.text_input("Paste a Grants.gov opportunity URL:")
    if st.button("Fetch"):
        opp = parse_opp_id(url.strip())
        if not opp:
            st.error("Couldn't find oppId in that URL.")
        else:
            data = fetch_grants_json(opp)
            if data:
                st.session_state["gjson"] = data
                st.session_state["gblob"] = compose_text_blob(data)
                st.session_state["g_show_full"] = False
                st.success(f"Fetched opportunity {opp}.")
            else:
                st.error("Grants.gov didn't return JSON.")

    # 🔑 shorthand references –- ALWAYS place these **inside** the tab block
    gjson      = st.session_state.get("gjson")
    gblob      = st.session_state.get("gblob", "")
    show_full  = st.session_state.get("g_show_full", False)

    # 2️⃣ Metadata + AI summary --------------------------------------------
    if gjson:
        # ---------- Branch -------------------------------------------------
        name  = next(filter(None, [
            gjson.get("agencyName"),
            gjson.get("agencyContactName"),
            (gjson.get("topAgencyDetails") or {}).get("agencyName")
        ]), "N/A").strip()
        code  = next(filter(None, [
            gjson.get("agencyCode"),
            (gjson.get("topAgencyDetails") or {}).get("agencyCode")
        ]), "").strip().upper()
        branch = f"{name} ({code})" if code else name

        # ---------- Synopsis plain-text ------------------------------------
        raw_syn   = gjson.get("synopsis") or gjson.get("synopsisDesc") or ""
        syn_plain = BeautifulSoup(str(raw_syn), "html.parser").get_text(" ", strip=True)

        # ---------- Due date -----------------------------------------------
        due_for_card = get_due_date(gjson, syn_plain)   # <— single, reliable call
        due_display   = human_date(due_for_card)

        # -------- Four-column "Basic Info" card ------------------------------
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.caption("🧾 Opportunity #")
        c1.text(gjson.get("opportunityNumber", "N/A"))

        c2.caption("🏛️ Branch")
        c2.text(branch)                        # wraps nicely

        c3.caption("📅 Due Date")
        c3.text(due_display)                   # e.g. September 29, 2025

        c4.caption("📝 Title")
        c4.text(gjson.get("opportunityTitle", "N/A"))   # full title visible

        # Extract and show funding amount
        funding_amount = extract_grants_funding_amount(gblob or syn_plain, gjson)
        c5.caption("💰 Funding")
        c5.text(funding_amount)

        # ---------- cues (budget + eligibility) ----------------------------
        cues = []
        lo, hi = gjson.get("awardFloor"), gjson.get("awardCeiling")
        if lo and hi:
            cues.append(f"NOTE: awards roughly ${int(lo):,} – ${int(hi):,}.")
        elig_desc = (gjson.get("applicantEligibilityDesc","") +
                     json.dumps(gjson.get("applicantTypes",[]))).lower()
        if not any(k in elig_desc for k in ["small business","for-profit","startup"]):
            cues.append("FIT-ALERT: Eligibility omits small businesses / startups.")

        # ---------- GPT summary --------------------------------------------
        syn_for_gpt  = f"The deadline for applications is {due_for_card}.\n\n{syn_plain}"
        full_for_gpt = ("\n".join(cues) + "\n\n" if cues else "") + syn_for_gpt
        with st.spinner("Summarizing…"):
            summary = summarize_text(full_for_gpt)

        summary = f"**Deadline → {due_for_card}**\n\n" + summary
        st.markdown("### 🧠 AI Summary")
        st.info(re.sub(r"\s+", " ", summary).strip())
        
                # Extract and display checklist in collapsible section
        checklist_text = extract_application_checklist(full_for_gpt)
        with st.expander("📋 Application Checklist"):
            st.write(checklist_text)
        
        # Collect for export
        st.session_state["export_rows_grantsgov"].append({
            "Opportunity ID": gjson.get("opportunityNumber", "N/A"),
            "Due Date": due_display,
            "Branch": branch,
            "Title": gjson.get("opportunityTitle", "N/A"),
            "Funding Amount": funding_amount,
            "Summary": summary,
            "Application Checklist": checklist_text,
            "Source URL": "Grants.gov"
        })

    # 3️⃣ Show full text toggle --------------------------------------------
    if gblob and not show_full:
        if st.button("📄 Show full text"):
            st.session_state["g_show_full"] = True
            show_full = True

    # 4️⃣ Text-area / manual paste -----------------------------------------
    if show_full or not gjson:
        grants_text = st.text_area("Paste Grants.gov solicitation text here:",
                                   height=300, key="grants_text",
                                   value=gblob if show_full else "")
        if st.button("🔍 Analyze Grants.gov Text"):
            # Use display_analysis_results but also collect for export
            display_analysis_results(grants_text)
            # Note: display_analysis_results for string input doesn't provide structured data,
            # so we'll use a simplified collection here
            meta = extract_fields_with_gpt(grants_text)
            funding_amount = extract_grants_funding_amount(grants_text, gjson)
            summary = summarize_text(grants_text)
            
            # Extract and display checklist in collapsible section
            checklist_text = extract_application_checklist(grants_text)
            with st.expander("📋 Application Checklist"):
                st.write(checklist_text)
            
            st.session_state["export_rows_grantsgov"].append({
                "Opportunity ID": meta.get("Opportunity ID", "N/A"),
                "Due Date": meta.get("Due Date", "N/A"),
                "Branch": meta.get("Branch", "N/A"),
                "Title": "",
                "Funding Amount": funding_amount,
                "Summary": summary,
                "Application Checklist": checklist_text,
                "Source URL": "Grants.gov"
            })

    # 5️⃣ PDF upload (auto-collapse if there's text) -----------------------
    grants_text_value = st.session_state.get("grants_text", "")
    has_grants_text = grants_text_value.strip() != "" or gblob.strip() != ""
    with st.expander("📎 Upload Grants.gov PDF", expanded=not has_grants_text):
        grants_pdf = st.file_uploader("Upload a Grants.gov PDF file", type=["pdf"],
                                      key="grants_pdf")
    if grants_pdf:
        pdf_text = extract_text_from_pdf(grants_pdf)
        if pdf_text.startswith("PDF extraction failed"):
            st.error(pdf_text)
        else:
            st.success("✅ PDF text extracted.")
            display_analysis_results(pdf_text)
            # Collect for export
            meta = extract_fields_with_gpt(pdf_text)
            funding_amount = extract_grants_funding_amount(pdf_text, gjson)
            summary = summarize_text(pdf_text)
            
            # Extract and display checklist in collapsible section
            checklist_text = extract_application_checklist(pdf_text)
            with st.expander("📋 Application Checklist"):
                st.write(checklist_text)
            
            st.session_state["export_rows_grantsgov"].append({
                "Opportunity ID": meta.get("Opportunity ID", "N/A"),
                "Due Date": meta.get("Due Date", "N/A"),
                "Branch": meta.get("Branch", "N/A"),
                "Title": "",
                "Funding Amount": funding_amount,
                "Summary": summary,
                "Application Checklist": checklist_text,
                "Source URL": "Grants.gov"
            })
    
    # Export button for Grants.gov
    if st.button("Export Grants.gov results to Google Sheets"):
        sheet_name = st.text_input("Google Sheet Name:", value="Government Opportunities", key="grants_sheet_name_input")
        if sheet_name:
            try:
                export_to_gsheet(st.session_state["export_rows_grantsgov"], sheet_name=sheet_name, worksheet_name="Grants.gov")
                st.session_state["export_rows_grantsgov"] = []  # Optional: clear after export
                st.success(f"Exported to Google Sheets: {sheet_name} (Grants.gov tab)!")
            except Exception as e:
                st.error(f"Export failed: {e}")
                st.info("Make sure the sheet exists and is shared with your service account.")
        else:
            st.warning("Please enter a sheet name.")

    # Add "See Your Sheet" button for Grants.gov
    st.link_button("📊 See Your Sheet", "//")

# --- Grants.gov TAB END -----------------------------------------------------

# --- 📁 Random TAB (Catch-all for file uploads) ----------------------------
if selected_tab == "📁 Random":
    st.header("📁 Random Document Analyzer")
    
    # Instructions for users
    st.info("💡 **How to use:** Not sure where to go? Paste in the PDF summary of any solicitation here for full information. If there is no PDF, locate the domain name and switch to the corresponding tab for instructions.")
    
    st.write("Upload any grant, contract, or opportunity document for analysis.")

    # ── File upload (always expanded) ─────────────────────────────────────
    st.subheader("📎 Upload Document")
    random_file = st.file_uploader(
        "Upload any PDF, Word, or text file",
        type=["pdf", "doc", "docx", "txt"],
        key="random_file"
    )
    
    if random_file:
        with st.spinner("🔍 Extracting text from file..."):
            # Handle different file types
            if random_file.type == "application/pdf":
                file_text = extract_text_from_pdf(random_file)
            elif random_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                # For Word docs, try to extract as text (basic support)
                file_text = "Word document processing not fully implemented. Please convert to PDF."
            elif random_file.type == "text/plain":
                # For text files
                file_text = str(random_file.read(), "utf-8")
            else:
                file_text = "Unsupported file type. Please use PDF, Word, or text files."

        if file_text.startswith("PDF extraction failed") or file_text.startswith("Word document processing") or file_text.startswith("Unsupported file type"):
            st.error(file_text)
        else:
            st.success("✅ File text extracted.")

            # Use general metadata extraction
            meta = extract_fields_with_gpt(file_text)
            funding_amount = extract_funding_amount(file_text)
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.caption("🧾 Opportunity ID"); c1.text(meta.get("Opportunity ID", "N/A"))
            c2.caption("📅 Due Date");       c2.text(meta.get("Due Date", "N/A"))
            c3.caption("🏛️ Branch");         c3.text(meta.get("Branch", "N/A"))
            c4.caption("📝 Title");           c4.text(meta.get("Title", "N/A"))
            c5.caption("💰 Funding");         c5.text(funding_amount)
            
            summary = summarize_text(file_text)
            st.markdown("### 🧠 AI Summary")
            st.info(summary)
            
            # Extract and display checklist in collapsible section
            checklist_text = extract_application_checklist(file_text)
            with st.expander("📋 Application Checklist"):
                st.write(checklist_text)
            
            # Collect for export
            st.session_state["export_rows_random"].append({
                "Opportunity ID": meta.get("Opportunity ID", "N/A"),
                "Due Date": meta.get("Due Date", "N/A"),
                "Branch": meta.get("Branch", "N/A"),
                "Title": meta.get("Title", "N/A"),
                "Funding Amount": funding_amount,
                "Summary": summary,
                "Application Checklist": checklist_text,
                "Source URL": f"File: {random_file.name}"
            })

    # Export button for Random
    if st.button("Export Random results to Google Sheets"):
        sheet_name = st.text_input("Google Sheet Name:", value="Government Opportunities", key="random_sheet_name_input")
        if sheet_name:
            try:
                export_to_gsheet(st.session_state["export_rows_random"], sheet_name=sheet_name, worksheet_name="Random")
                st.session_state["export_rows_random"] = []  # Optional: clear after export
                st.success(f"Exported to Google Sheets: {sheet_name} (Random tab)!")
            except Exception as e:
                st.error(f"Export failed: {e}")
                st.info("Make sure the sheet exists and is shared with your service account.")
        else:
            st.warning("Please enter a sheet name.")

    # Add "See Your Sheet" button for Random (will need actual GID when created)
    st.link_button("📊 See Your Sheet", "//")

# --- Random TAB END --------------------------------------------------------



