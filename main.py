
import json
import os
import tempfile
from decimal import Decimal, InvalidOperation

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCR

app = FastAPI(title="Invoice AI")

# Initialize OCR once when the application starts.
ocr = PaddleOCR(use_angle_cls=True, lang="en")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


def extract_text(image_path: str) -> str:
    """Extract text and remove exact consecutive duplicates."""
    result = ocr.ocr(image_path, cls=True)
    lines = []

    if result and result[0]:
        for item in result[0]:
            text = item[1][0].strip()

            if not text:
                continue

            # Remove only exact consecutive duplicates.
            if lines and text == lines[-1]:
                continue

            lines.append(text)

    return "\n".join(lines)


def extract_invoice_fields(raw_text: str) -> dict:
    """Convert OCR text into structured invoice JSON."""

    prompt = f"""
Extract invoice information from the OCR text below.

Return one JSON object containing:
- invoice_number: string or null
- invoice_date: string or null
- vendor: string or null
- invoice_type: purchase, sales, or unknown
- currency: ISO currency code such as USD or INR, or null
- items: array of objects with:
    - name: string
    - quantity: number or null
    - unit_price: number or null
    - amount: number or null
- subtotal: number or null
- tax: number or null
- total: number or null

Rules:
1. Never invent missing values. Use null.
2. Do not interpret payment methods such as DEBIT as currency.
3. Do not treat phone numbers, barcodes, or card details
   as invoice numbers or products.
4. OCR may contain duplicate or slightly different readings
   of the same printed lines. Do not duplicate an item solely
   because its text appears more than once.
5. Do not merge distinct purchased products.
6. Preserve the printed subtotal, tax, and total.
7. Include purchased products only, not promotional text.
8. Use the printed quantity when identifiable. If unclear,
   use null rather than assuming a quantity.
9. If the currency cannot be identified, use null.
10. Do not calculate missing totals by guessing.
11. If the invoice type is unclear, use "unknown".
12. Return valid JSON only.

OCR text:
{raw_text}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You extract structured invoice data. "
                            "Follow the schema and return JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "format": "json",
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not reach Ollama. Check that it is running.",
        ) from exc

    try:
        content = response.json()["message"]["content"]
        invoice = json.loads(content)

        required_fields = {
            "invoice_number",
            "invoice_date",
            "vendor",
            "invoice_type",
            "currency",
            "items",
            "subtotal",
            "tax",
            "total",
        }

        if not isinstance(invoice, dict):
            raise ValueError("Expected a JSON object.")

        if not required_fields.issubset(invoice.keys()):
            raise ValueError("Required invoice fields are missing.")

        if not isinstance(invoice["items"], list):
            raise ValueError("Invoice items must be a list.")

        for item in invoice["items"]:
            if not isinstance(item, dict):
                raise ValueError("Each item must be a JSON object.")

            if "name" not in item:
                raise ValueError("An item is missing its name.")

        return invoice

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="The model returned invalid invoice JSON.",
        ) from exc


def to_decimal(value):
    """Convert a numeric value to Decimal, or return None."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = Decimal(str(value))

        if not number.is_finite():
            return None

        return number

    except (InvalidOperation, ValueError, TypeError):
        return None


def validate_invoice(invoice: dict) -> dict:
    """Check arithmetic consistency without changing model values."""
    issues = []
    items = invoice.get("items") or []

    subtotal = to_decimal(invoice.get("subtotal"))
    tax = to_decimal(invoice.get("tax"))
    total = to_decimal(invoice.get("total"))

    item_amounts = [
        to_decimal(item.get("amount"))
        for item in items
    ]

    # Check item sum only when all item amounts are available.
    if items and all(amount is not None for amount in item_amounts):
        items_sum = sum(item_amounts, Decimal("0"))

        if subtotal is not None:
            difference = abs(items_sum - subtotal)

            if difference > Decimal("0.02"):
                issues.append(
                    "Line-item amounts do not match the subtotal."
                )
        else:
            issues.append(
                "Cannot compare line items because subtotal is missing."
            )
    else:
        items_sum = None
        if items:
            issues.append(
                "Some item amounts are missing or invalid."
            )

    # Check subtotal + tax against the total.
    if subtotal is not None and tax is not None and total is not None:
        difference = abs(subtotal + tax - total)

        if difference > Decimal("0.02"):
            issues.append(
                "Subtotal plus tax does not match the total."
            )
    else:
        issues.append(
            "Cannot fully validate subtotal, tax, and total."
        )

    return {
        "items_sum": (
            float(items_sum) if items_sum is not None else None
        ),
        "subtotal_matches_items": (
            abs(items_sum - subtotal) <= Decimal("0.02")
            if items_sum is not None and subtotal is not None
            else None
        ),
        "total_matches_subtotal_plus_tax": (
            abs(subtotal + tax - total) <= Decimal("0.02")
            if subtotal is not None
            and tax is not None
            and total is not None
            else None
        ),
        "is_consistent": len(issues) == 0,
        "issues": issues,
        "requires_review": len(issues) > 0,
    }


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Invoice AI",
    }


@app.post("/extract-invoice")
async def extract_invoice(file: UploadFile = File(...)):
    """Upload an invoice image and extract structured information."""

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WebP image.",
        )

    image_bytes = await file.read(MAX_IMAGE_SIZE + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB.",
        )

    suffixes = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    suffix = suffixes[file.content_type]
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp:
            temp.write(image_bytes)
            temp_path = temp.name

        raw_text = extract_text(temp_path)

        if not raw_text.strip():
            raise HTTPException(
                status_code=422,
                detail="No readable text found in the image.",
            )

        invoice = extract_invoice_fields(raw_text)
        validation = validate_invoice(invoice)

        return {
            "ocr_text": raw_text,
            "invoice": invoice,
            "validation": validation,
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
