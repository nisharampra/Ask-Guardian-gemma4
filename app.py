import hashlib
import io
import os
import re
import csv
import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

try:
    from chromadb.api.types import EmbeddingFunction
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except Exception:  # pragma: no cover - keeps import failures visible in the UI
    EmbeddingFunction = object
    SentenceTransformerEmbeddingFunction = None


APP_DIR = Path(__file__).resolve().parent
CHROMA_DIR = APP_DIR / "guardian_chroma"
MEMORY_PATH = APP_DIR / "guardian_user_memory.json"
COLLECTION_NAME = "guardian_statement_rows"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GEMMA_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-3-4b-it")
COUNTRY = os.getenv("STATEMENT_COUNTRY", "Uploaded statement")
CURRENCY_CODE = os.getenv("STATEMENT_CURRENCY", "SGD")
CURRENCY_SYMBOL = os.getenv("STATEMENT_CURRENCY_SYMBOL", "S$")
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")
CSV_SEPARATORS = (None, ",", ";", "\t", "|")


CATEGORY_RULES = {
    "Food": [
        "restaurant",
        "cafe",
        "coffee",
        "grabfood",
        "foodpanda",
        "deliveroo",
        "mcdonald",
        "mcdonalds",
        "burger",
        "pizza",
        "bakery",
        "kopitiam",
        "koufu",
        "toast box",
        "ya kun",
        "hawker",
        "food court",
        "old chang kee",
        "dining",
        "eatery",
        "takeaway",
        "takeout",
        "kitchen",
    ],
    "Transport": [
        "grab",
        "gojek",
        "tada",
        "ryde",
        "comfort",
        "comfortdelgro",
        "taxi",
        "mrt",
        "smrt",
        "sbs",
        "simplygo",
        "ezlink",
        "ez-link",
        "bus",
        "fuel",
        "petrol",
        "parking",
        "erp",
        "uber",
        "lyft",
        "train",
        "rail",
        "metro",
        "subway",
        "transit",
    ],
    "Groceries": [
        "grocery",
        "supermarket",
        "mart",
        "fairprice",
        "ntuc",
        "cold storage",
        "sheng siong",
        "giant",
        "redmart",
        "don don donki",
        "cs fresh",
        "prime supermarket",
        "walmart",
        "target",
        "costco",
        "aldi",
        "tesco",
        "carrefour",
        "hypermarket",
    ],
    "Shopping": [
        "amazon",
        "shop",
        "store",
        "mall",
        "retail",
        "lazada",
        "shopee",
        "qoo10",
        "uniqlo",
        "courts",
        "challenger",
        "takashimaya",
        "capitastar",
        "marketplace",
        "department store",
        "apparel",
        "clothing",
    ],
    "Bills": [
        "sp services",
        "singtel",
        "starhub",
        "m1",
        "simba",
        "circles.life",
        "utility",
        "water",
        "electricity",
        "internet",
        "phone",
        "mobile",
        "bill",
        "town council",
        "hdb",
        "insurance",
        "gas",
        "broadband",
        "council",
        "maintenance",
        "service charge",
    ],
    "Rent": ["rent", "apartment rent", "house rent", "room rent", "landlord", "lease"],
    "Subscriptions": [
        "netflix",
        "spotify",
        "prime",
        "icloud",
        "google",
        "microsoft",
        "subscription",
        "disney",
        "youtube",
        "adobe",
        "apple.com/bill",
    ],
    "Income": ["salary", "payroll", "interest", "refund", "cashback", "deposit", "giro salary", "cpf"],
    "Transfers": [
        "transfer",
        "paynow",
        "paylah",
        "dbs paylah",
        "pay lah",
        "pay anyone",
        "payanyone",
        "nets",
        "fast",
        "giro",
        "meps",
        "atm",
        "withdrawal",
        "fund transfer",
    ],
    "Healthcare": ["guardian", "watsons", "unity", "pharmacy", "doctor", "hospital", "clinic", "medical"],
    "Travel": ["hotel", "airline", "flight", "booking", "airbnb", "scoot", "sia", "jetstar", "travel", "railway", "resort"],
}

SCAM_RED_FLAGS = {
    "Requests OTP, national ID login, bank login, card CVV, or password": [
        "otp",
        "one time password",
        "identity login",
        "national id",
        "password",
        "cvv",
        "pin",
        "login details",
        "banking credentials",
    ],
    "Creates urgency or threatens account closure, police, tax, parcel, or legal action": [
        "urgent",
        "immediately",
        "account suspended",
        "account locked",
        "police",
        "warrant",
        "iras",
        "customs",
        "parcel",
        "delivery failed",
        "final warning",
    ],
    "Asks for payment, transfer, crypto, gift card, or PayNow to a personal account": [
        "paynow",
        "transfer now",
        "bank transfer",
        "crypto",
        "usdt",
        "gift card",
        "deposit",
        "processing fee",
    ],
    "Uses a suspicious shortened or non-official link": [
        "bit.ly",
        "tinyurl",
        "t.co",
        "is.gd",
        "rebrand.ly",
        "singpass-",
        "dbs-",
        "posb-",
        "ocbc-",
        "uob-",
    ],
    "Promises unrealistic investment returns or guaranteed profit": [
        "guaranteed profit",
        "risk free",
        "double your money",
        "daily profit",
        "fixed return",
        "capital guaranteed",
        "limited slots",
    ],
}

FINANCE_SIGNALS = {
    "Mentions financial regulation or licensing": [
        "regulated",
        "licensed",
        "financial conduct authority",
        "fca",
        "sec",
        "finra",
        "mas",
        "monetary authority",
        "capital markets services",
    ],
    "Mentions company registration identifiers": ["company registration", "tax id", "uen", "acra", "registered company"],
    "Mentions common regulated finance products": [
        "investment",
        "fund",
        "insurance",
        "loan",
        "trading",
        "forex",
        "crypto",
        "digital payment token",
        "robo advisor",
    ],
}


class HashEmbeddingFunction(EmbeddingFunction):
    """Deterministic fallback embeddings when sentence-transformers is unavailable."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:  # Chroma expects this name
        vectors = []
        for text in input:
            vector = [0.0] * self.dimensions
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class TextOnlyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self.skip = tag in {"script", "style", "noscript", "svg"}

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip = False

    def handle_data(self, data: str):
        if not self.skip and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


@dataclass
class RetrievedTransaction:
    document: str
    metadata: dict[str, Any]


@dataclass
class SafetyAssessment:
    verdict: str
    risk_score: int
    red_flags: list[str]
    reassuring_signals: list[str]
    recommendation: str
    extracted_text: str


def default_memory() -> dict[str, Any]:
    return {"facts": [], "category_rules": {}, "exclude_terms": []}


def load_user_memory() -> dict[str, Any]:
    if not MEMORY_PATH.exists():
        return default_memory()
    try:
        with MEMORY_PATH.open("r", encoding="utf-8") as handle:
            memory = json.load(handle)
    except Exception:
        return default_memory()
    base = default_memory()
    if isinstance(memory, dict):
        base.update({key: value for key, value in memory.items() if key in base})
    if not isinstance(base["facts"], list):
        base["facts"] = []
    if not isinstance(base["category_rules"], dict):
        base["category_rules"] = {}
    if not isinstance(base["exclude_terms"], list):
        base["exclude_terms"] = []
    return base


def save_user_memory(memory: dict[str, Any]):
    MEMORY_PATH.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")


def remember_user_text(text: str) -> str | None:
    normalized = text.strip()
    match = re.match(r"^(remember|learn|note|teach|correction)[:\s]+(.+)$", normalized, flags=re.I | re.S)
    if not match:
        return None

    lesson = re.sub(r"\s+", " ", match.group(2)).strip()
    if not lesson:
        return None

    memory = load_user_memory()
    if lesson not in memory["facts"]:
        memory["facts"].append(lesson)

    category_match = re.search(
        r"(.+?)\s+(?:is|are|should be|belongs to)\s+(income|food|transport|groceries|shopping|bills|subscriptions|transfers|healthcare|travel|rent|savings|other)\b",
        lesson,
        flags=re.I,
    )
    if category_match:
        term = re.sub(r"[^a-z0-9 ]+", " ", category_match.group(1).lower()).strip()
        category = category_match.group(2).title()
        if term:
            memory["category_rules"][term] = category

    exclude_match = re.search(r"(?:ignore|exclude|do not count)\s+(.+?)\s+(?:as spending|from spending|in spending)", lesson, flags=re.I)
    if exclude_match:
        term = re.sub(r"[^a-z0-9 ]+", " ", exclude_match.group(1).lower()).strip()
        if term and term not in memory["exclude_terms"]:
            memory["exclude_terms"].append(term)

    save_user_memory(memory)
    return lesson


def memory_to_text(memory: dict[str, Any]) -> str:
    parts = []
    if memory.get("facts"):
        parts.append("Learned user facts:\n" + "\n".join(f"- {fact}" for fact in memory["facts"][-12:]))
    if memory.get("category_rules"):
        rules = [f"- Treat '{term}' as {category}" for term, category in memory["category_rules"].items()]
        parts.append("Learned category rules:\n" + "\n".join(rules[-12:]))
    if memory.get("exclude_terms"):
        excludes = [f"- Exclude '{term}' from spending when relevant" for term in memory["exclude_terms"]]
        parts.append("Learned exclusions:\n" + "\n".join(excludes[-12:]))
    return "\n\n".join(parts) or "No learned user memory yet."


def clean_column_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def read_statement(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        raw = read_csv_robust(uploaded_file)
    elif suffix in {".xls", ".xlsx"}:
        raw = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Please upload a CSV, XLS, or XLSX bank statement.")

    raw = raw.dropna(how="all")
    raw.columns = [clean_column_name(column) for column in raw.columns]
    return normalize_statement(raw)


def read_many_statements(uploaded_files: list[Any]) -> pd.DataFrame:
    frames = []
    errors = []
    for uploaded_file in uploaded_files:
        try:
            frame = read_statement(uploaded_file)
            if not frame.empty:
                frame["source_file"] = getattr(uploaded_file, "name", "uploaded statement")
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{getattr(uploaded_file, 'name', 'statement')}: {exc}")

    if not frames:
        detail = " ".join(errors) if errors else "No usable transactions were found."
        raise ValueError(f"No usable transactions were found across the uploaded statements. {detail}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = dedupe_statement_rows(combined)
    combined = combined.sort_values(["date", "description", "amount"], ascending=[True, True, True]).reset_index(drop=True)
    return combined


def dedupe_statement_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dedupe_columns = [column for column in ["date", "description", "merchant", "amount"] if column in df.columns]
    if not dedupe_columns:
        return df
    return df.drop_duplicates(subset=dedupe_columns, keep="first")


def read_csv_robust(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if isinstance(content, str):
        content = content.encode("utf-8")

    errors: list[str] = []
    for encoding in CSV_ENCODINGS:
        sample = content[:8192].decode(encoding, errors="replace")
        sniffed_sep = sniff_csv_separator(sample)
        separators = [sniffed_sep] if sniffed_sep else []
        separators.extend(sep for sep in CSV_SEPARATORS if sep not in separators)

        for separator in separators:
            try:
                kwargs: dict[str, Any] = {
                    "encoding": encoding,
                    "skip_blank_lines": True,
                    "thousands": ",",
                    "on_bad_lines": "skip",
                }
                if separator is None:
                    kwargs.update({"sep": None, "engine": "python"})
                else:
                    kwargs["sep"] = separator
                candidate = pd.read_csv(io.BytesIO(content), **kwargs)
                candidate = drop_empty_unnamed_columns(candidate)
                if looks_like_statement(candidate):
                    return candidate
            except Exception as exc:
                errors.append(f"{encoding}/{separator or 'auto'}: {exc}")

    detail = "; ".join(errors[:3])
    raise ValueError(
        "Could not read this CSV as a bank statement. Try exporting with a header row containing date, "
        f"description, and amount/debit/credit columns. Parser details: {detail}"
    )


def sniff_csv_separator(sample: str) -> str | None:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return None


def drop_empty_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [column for column in df.columns if str(column).lower().startswith("unnamed")]
    empty_unnamed = [column for column in unnamed if df[column].isna().all()]
    return df.drop(columns=empty_unnamed)


def looks_like_statement(df: pd.DataFrame) -> bool:
    if df.empty or len(df.columns) < 2:
        return False
    columns = [clean_column_name(column) for column in df.columns]
    has_date = first_existing(
        columns,
        ["date", "transaction_date", "posted_date", "value_date", "txn_date", "posting_date", "entry_date"],
    )
    has_description = first_existing(
        columns,
        [
            "description",
            "narration",
            "details",
            "transaction_details",
            "transaction_description",
            "merchant",
            "payee",
            "particulars",
            "reference",
            "transaction_ref",
        ],
    )
    has_amount = first_existing(
        columns,
        [
            "amount",
            "transaction_amount",
            "amt",
            "value",
            "sgd_amount",
            "debit",
            "debit_amount",
            "withdrawal",
            "withdrawals",
            "withdrawal_amount",
            "paid_out",
            "outflow",
            "credit",
            "credit_amount",
            "deposit",
            "deposits",
            "deposit_amount",
            "paid_in",
            "inflow",
        ],
    )
    return bool(has_date and has_description and has_amount)


def first_existing(columns: list[str], options: list[str]) -> str | None:
    for option in options:
        if option in columns:
            return option
    return None


def normalize_statement(raw: pd.DataFrame) -> pd.DataFrame:
    raw = clean_malformed_datetime_values(raw)
    columns = list(raw.columns)
    date_col = first_existing(
        columns,
        ["date", "transaction_date", "posted_date", "value_date", "txn_date", "posting_date", "entry_date"],
    )
    desc_col = first_existing(
        columns,
        [
            "description",
            "narration",
            "details",
            "transaction_details",
            "transaction_description",
            "merchant",
            "payee",
            "counterparty",
            "beneficiary",
            "payer",
            "particulars",
            "reference",
            "transaction_ref",
            "memo",
            "remarks",
            "transaction_narrative",
        ],
    )
    merchant_col = first_existing(columns, ["merchant", "payee", "counterparty", "beneficiary", "payer"])
    amount_col = first_existing(
        columns,
        ["amount", "transaction_amount", "amt", "value", "sgd_amount", "signed_amount", "transaction_value"],
    )
    debit_col = first_existing(
        columns,
        [
            "debit",
            "debit_amount",
            "withdrawal",
            "withdrawals",
            "withdrawal_amount",
            "paid_out",
            "outflow",
            "money_out",
            "payments",
        ],
    )
    credit_col = first_existing(
        columns,
        [
            "credit",
            "credit_amount",
            "deposit",
            "deposits",
            "deposit_amount",
            "paid_in",
            "inflow",
            "money_in",
            "receipts",
        ],
    )
    direction_col = first_existing(columns, ["type", "direction", "dr_cr", "debit_credit", "transaction_type"])
    balance_col = first_existing(columns, ["balance", "running_balance", "available_balance", "closing_balance"])

    if not date_col or not desc_col:
        raise ValueError("Could not find required date and description columns in the statement.")
    if not amount_col and not (debit_col or credit_col):
        raise ValueError("Could not find an amount column, or debit/credit columns, in the statement.")

    df = pd.DataFrame()
    df["date"] = parse_dates_robust(raw[date_col])
    df["description"] = combine_text_columns(raw, [desc_col, "remarks", "memo", "reference", "transaction_ref"])
    df["merchant"] = (
        raw[merchant_col].fillna("").astype(str).str.strip()
        if merchant_col
        else df["description"].apply(extract_merchant)
    )

    if amount_col:
        df["amount"] = raw[amount_col].apply(parse_amount)
        if direction_col:
            df["amount"] = [
                apply_direction(amount, direction)
                for amount, direction in zip(df["amount"], raw[direction_col].fillna("").astype(str))
            ]
    else:
        debit = raw[debit_col].apply(parse_amount) if debit_col else 0.0
        credit = raw[credit_col].apply(parse_amount) if credit_col else 0.0
        df["amount"] = credit.fillna(0.0) - debit.fillna(0.0)

    df = df.dropna(subset=["date"])
    df = df[df["description"].str.len() > 0]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["type"] = df["amount"].apply(lambda value: "Income" if value > 0 else "Expense")
    df["spend"] = df["amount"].apply(lambda value: abs(value) if value < 0 else 0.0)
    df["category"] = df.apply(categorize_transaction, axis=1)
    df["category"] = apply_learned_categories(df, load_user_memory())
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    if balance_col:
        df["balance"] = raw.loc[df.index, balance_col].apply(parse_amount) if len(raw) == len(df) else None
    df = df.reset_index(drop=True)
    return df


def clean_malformed_datetime_values(raw: pd.DataFrame) -> pd.DataFrame:
    cleaned = raw.copy()
    for column in cleaned.columns:
        column_name = str(column).lower()
        if "date" in column_name or "time" in column_name:
            cleaned[column] = (
                cleaned[column]
                .astype(str)
                .str.replace(r"(\d{4}-\d{2}-\d{2})(\d{6,})", r"\1", regex=True)
                .str.replace(r"(\d{1,2}/\d{1,2}/\d{4})(\d{6,})", r"\1", regex=True)
                .str.replace(r"(\d{1,2}-\d{1,2}-\d{4})(\d{6,})", r"\1", regex=True)
            )
    return cleaned


def combine_text_columns(raw: pd.DataFrame, candidates: list[str]) -> pd.Series:
    existing = []
    seen = set()
    for column in candidates:
        if column in raw.columns and column not in seen:
            existing.append(column)
            seen.add(column)
    if not existing:
        return pd.Series([""] * len(raw), index=raw.index)

    text = raw[existing].fillna("").astype(str)
    return text.apply(
        lambda row: " | ".join(part.strip() for part in row if part and part.strip() and part.strip().lower() != "nan"),
        axis=1,
    )


def parse_dates_robust(values: pd.Series) -> pd.Series:
    text_values = values.astype(str).str.strip()
    iso_mask = text_values.str.match(r"^\d{4}-\d{1,2}-\d{1,2}(?:\b|[ T])", na=False)
    iso_dates = pd.to_datetime(text_values.where(iso_mask), errors="coerce", yearfirst=True)
    day_first = pd.to_datetime(text_values.where(~iso_mask), errors="coerce", dayfirst=True)
    month_first = pd.to_datetime(text_values.where(~iso_mask), errors="coerce", dayfirst=False)
    parsed = iso_dates.fillna(day_first).fillna(month_first)
    return parsed.dt.date


def parse_amount(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in {"", "-"}:
        return 0.0
    if looks_like_datetime_value(text):
        return 0.0
    is_negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned in {"", "-", "."}:
        return 0.0
    try:
        amount = float(cleaned)
    except ValueError:
        return 0.0
    return -abs(amount) if is_negative else amount


def looks_like_datetime_value(text: str) -> bool:
    normalized = text.strip()
    datetime_patterns = [
        r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$",
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$",
        r"^\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?$",
        r"^\d{4}-\d{2}-\d{2}\d{6,}$",
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{4}\d{6,}$",
    ]
    return any(re.search(pattern, normalized, flags=re.I) for pattern in datetime_patterns)


def apply_direction(amount: float, direction: str) -> float:
    normalized = direction.lower().strip()
    if re.search(r"\b(dr|debit|withdrawal|paid|out|expense)\b", normalized):
        return -abs(float(amount))
    if re.search(r"\b(cr|credit|deposit|in|income|receipt)\b", normalized):
        return abs(float(amount))
    return float(amount)


def extract_merchant(description: str) -> str:
    cleaned = re.sub(
        r"\b(pos|debit|credit|card|payment|purchase|transfer|nets|giro|paynow|paylah|visa|mastercard)\b",
        "",
        description,
        flags=re.I,
    )
    cleaned = re.sub(r"[^A-Za-z0-9 &.-]+", " ", cleaned)
    words = cleaned.strip().split()
    return " ".join(words[:4]) if words else "Unknown merchant"


def categorize_transaction(row: pd.Series) -> str:
    text = f"{row.get('merchant', '')} {row.get('description', '')}".lower()
    if row.get("amount", 0) > 0:
        return "Income"
    for category, terms in CATEGORY_RULES.items():
        if any(term in text for term in terms):
            return category
    return "Other"


def apply_learned_categories(df: pd.DataFrame, memory: dict[str, Any]) -> pd.Series:
    categories = df["category"].copy()
    learned_rules = memory.get("category_rules", {})
    if not learned_rules:
        return categories

    searchable = (
        df["merchant"].fillna("").astype(str).str.lower()
        + " "
        + df["description"].fillna("").astype(str).str.lower()
    )
    for term, category in learned_rules.items():
        clean_term = str(term).lower().strip()
        if clean_term:
            categories = categories.mask(searchable.str.contains(re.escape(clean_term), na=False), str(category))
    return categories


def transaction_to_document(row: pd.Series) -> str:
    amount = float(row["amount"])
    direction = "credit" if amount > 0 else "debit"
    balance = ""
    if "balance" in row and pd.notna(row.get("balance")):
        balance = f" Balance after transaction: {format_sgd(float(row['balance']))}."
    return (
        f"Date: {row['date']}. Merchant: {row['merchant']}. "
        f"Description: {row['description']}. Category: {row['category']}. "
        f"Amount: {format_sgd(amount)}. Currency: {CURRENCY_CODE}. "
        f"Type: {row['type']}. Direction: {direction}. Month: {row['month']}. Country: {COUNTRY}.{balance}"
    )


def format_sgd(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}{CURRENCY_SYMBOL}{abs(amount):,.2f}"


@st.cache_resource(show_spinner=False)
def get_embedding_function():
    if SentenceTransformerEmbeddingFunction:
        try:
            return SentenceTransformerEmbeddingFunction(model_name=DEFAULT_EMBEDDING_MODEL)
        except Exception:
            return HashEmbeddingFunction()
    return HashEmbeddingFunction()


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    CHROMA_DIR.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def reset_collection():
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def index_statement(df: pd.DataFrame):
    collection = reset_collection()
    docs = [transaction_to_document(row) for _, row in df.iterrows()]
    ids = [f"txn-{idx}" for idx in range(len(docs))]
    metadatas = [
        {
            "date": str(row["date"]),
            "merchant": str(row["merchant"]),
            "description": str(row["description"]),
            "category": str(row["category"]),
            "amount": float(row["amount"]),
            "spend": float(row["spend"]),
            "type": str(row["type"]),
            "month": str(row["month"]),
        }
        for _, row in df.iterrows()
    ]
    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metadatas)
    return collection


def retrieve_transactions(question: str, n_results: int = 12) -> list[RetrievedTransaction]:
    collection = get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        return []

    if is_overview_question(question):
        rows = collection_rows(collection)
        return prioritize_overview_rows(rows, n_results)

    structured_matches = structured_retrieval(collection, question)
    if structured_matches:
        return structured_matches[:n_results]

    result = collection.query(query_texts=[question], n_results=min(n_results, collection.count()))
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    retrieved = [RetrievedTransaction(document=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]
    return dedupe_retrieved(retrieved)[:n_results]


def is_overview_question(question: str) -> bool:
    normalized = question.lower()
    return any(
        term in normalized
        for term in [
            "overview",
            "summary",
            "summarize",
            "analyse",
            "analyze",
            "insight",
            "budget",
            "pattern",
            "overall",
            "how am i doing",
            "spending habit",
        ]
    )


def prioritize_overview_rows(rows: list[RetrievedTransaction], n_results: int) -> list[RetrievedTransaction]:
    if not rows:
        return []
    unusual = find_unusual_rows(rows)
    recurring = find_recurring_rows(rows)
    largest_income = sorted(
        [row for row in rows if float(row.metadata.get("amount", 0.0)) > 0],
        key=lambda row: float(row.metadata.get("amount", 0.0)),
        reverse=True,
    )
    by_spend = sorted(rows, key=lambda row: float(row.metadata.get("spend", 0.0)), reverse=True)
    return dedupe_retrieved(unusual + recurring + largest_income + by_spend)[:n_results]


def collection_rows(collection) -> list[RetrievedTransaction]:
    result = collection.get(include=["documents", "metadatas"])
    return [
        RetrievedTransaction(document=doc, metadata=meta)
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
    ]


def structured_retrieval(collection, question: str) -> list[RetrievedTransaction]:
    lower_question = question.lower()
    rows = collection_rows(collection)
    selected: list[RetrievedTransaction] = []
    merchant_selected: list[RetrievedTransaction] = []

    for category in CATEGORY_RULES:
        if category.lower() in lower_question:
            selected.extend(row for row in rows if str(row.metadata.get("category", "")).lower() == category.lower())

    if "food" in lower_question or "restaurant" in lower_question or "cafe" in lower_question:
        selected.extend(row for row in rows if row.metadata.get("category") == "Food")

    if any(term in lower_question for term in ["subscription", "repeat", "recurring", "monthly"]):
        selected.extend(find_recurring_rows(rows))

    month = extract_month_filter(lower_question, rows)
    if month:
        selected.extend(row for row in rows if row.metadata.get("month") == month)

    merchant_terms = extract_possible_merchants(lower_question)
    if merchant_terms:
        for row in rows:
            merchant = str(row.metadata.get("merchant", "")).lower()
            description = str(row.metadata.get("description", "")).lower()
            if any(term in merchant or term in description for term in merchant_terms):
                merchant_selected.append(row)

    if any(term in lower_question for term in ["unusual", "large", "high", "highest", "spike", "outlier"]):
        unusual_scope = merchant_selected or selected or rows
        unusual_rows = find_unusual_rows(unusual_scope)
        selected.extend(unusual_rows or unusual_scope)

    selected.extend(merchant_selected)

    return dedupe_retrieved(selected)


def dedupe_retrieved(rows: list[RetrievedTransaction]) -> list[RetrievedTransaction]:
    seen: set[tuple[str, str, float]] = set()
    deduped = []
    for row in rows:
        key = (
            str(row.metadata.get("date", "")),
            str(row.metadata.get("description", "")),
            float(row.metadata.get("amount", 0.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def find_recurring_rows(rows: list[RetrievedTransaction]) -> list[RetrievedTransaction]:
    groups: dict[tuple[str, float], list[RetrievedTransaction]] = {}
    for row in rows:
        if row.metadata.get("type") != "Expense":
            continue
        merchant = re.sub(r"[^a-z0-9]+", " ", str(row.metadata.get("merchant", "")).lower()).strip()
        amount = round(abs(float(row.metadata.get("amount", 0.0))), 2)
        if not merchant or amount == 0:
            continue
        groups.setdefault((merchant, amount), []).append(row)

    recurring = []
    for matches in groups.values():
        months = {match.metadata.get("month") for match in matches}
        if len(matches) >= 2 and len(months) >= 2:
            recurring.extend(matches)
    return recurring


def find_unusual_rows(rows: list[RetrievedTransaction]) -> list[RetrievedTransaction]:
    expenses = [row for row in rows if float(row.metadata.get("spend", 0.0)) > 0]
    if not expenses:
        return []
    spends = pd.Series([float(row.metadata.get("spend", 0.0)) for row in expenses])
    threshold = max(spends.quantile(0.85), spends.mean() + spends.std(ddof=0))
    return sorted(
        [row for row in expenses if float(row.metadata.get("spend", 0.0)) >= threshold],
        key=lambda row: float(row.metadata.get("spend", 0.0)),
        reverse=True,
    )


def extract_month_filter(question: str, rows: list[RetrievedTransaction]) -> str | None:
    month_names = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }
    months_in_data = sorted({str(row.metadata.get("month")) for row in rows if row.metadata.get("month")})
    for name, number in month_names.items():
        if name in question:
            matching = [month for month in months_in_data if month.endswith(f"-{number}")]
            return matching[-1] if matching else None
    explicit = re.search(r"\b(20\d{2})[-/](0[1-9]|1[0-2])\b", question)
    if explicit:
        return f"{explicit.group(1)}-{explicit.group(2)}"
    return None


def extract_possible_merchants(question: str) -> list[str]:
    stopwords = {
        "show",
        "what",
        "did",
        "spend",
        "spent",
        "on",
        "for",
        "any",
        "why",
        "was",
        "my",
        "the",
        "in",
        "high",
        "payments",
        "payment",
        "food",
        "subscriptions",
        "repeat",
        "monthly",
        "unusual",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]{3,}", question.lower())
        if token not in stopwords and not token.startswith("20")
    ]


def build_statement_summary(retrieved: list[RetrievedTransaction]) -> str:
    if not retrieved:
        return "No statement rows were retrieved."

    total_spend = sum(float(item.metadata.get("spend", 0.0)) for item in retrieved)
    total_income = sum(
        float(item.metadata.get("amount", 0.0))
        for item in retrieved
        if float(item.metadata.get("amount", 0.0)) > 0
    )
    categories: dict[str, float] = {}
    months = sorted({str(item.metadata.get("month", "")) for item in retrieved if item.metadata.get("month")})
    for item in retrieved:
        category = str(item.metadata.get("category", "Other"))
        categories[category] = categories.get(category, 0.0) + float(item.metadata.get("spend", 0.0))
    top_categories = sorted(categories.items(), key=lambda pair: pair[1], reverse=True)[:5]
    top_category_text = ", ".join(f"{category}: {format_sgd(value)}" for category, value in top_categories if value)
    return (
        f"Retrieved rows: {len(retrieved)}. Months represented: {', '.join(months) or 'unknown'}. "
        f"Retrieved expense total: {format_sgd(total_spend)}. Retrieved income total: {format_sgd(total_income)}. "
        f"Top retrieved categories: {top_category_text or 'none'}."
    )


def build_dataframe_summary(df: pd.DataFrame | None, memory: dict[str, Any] | None = None) -> str:
    if df is None or df.empty:
        return "No uploaded statement dataframe is available."

    working = apply_memory_exclusions(df, memory or load_user_memory())
    expenses = working[working["amount"] < 0].copy()
    income = working[working["amount"] > 0].copy()
    total_spend = expenses["spend"].sum()
    total_income = income["amount"].sum()
    net = total_income - total_spend
    months = sorted(working["month"].dropna().astype(str).unique())
    top_categories = expenses.groupby("category")["spend"].sum().sort_values(ascending=False).head(8)
    recurring = recurring_expense_summary(expenses).head(8)
    unusual = expenses.sort_values("spend", ascending=False).head(8)

    lines = [
        f"Rows: {len(working)}. Months: {', '.join(months) or 'unknown'}.",
        f"Income: {format_sgd(total_income)}. Spending: {format_sgd(total_spend)}. Net: {format_sgd(net)}.",
        "Top spending categories: "
        + (", ".join(f"{category}: {format_sgd(value)}" for category, value in top_categories.items()) or "none"),
    ]
    if not recurring.empty:
        lines.append(
            "Likely recurring expenses: "
            + ", ".join(
                f"{row.merchant} {format_sgd(row.monthly_average)}/month"
                for row in recurring.itertuples(index=False)
            )
        )
    if not unusual.empty:
        lines.append(
            "Largest transactions: "
            + ", ".join(
                f"{row.date} {row.merchant} {format_sgd(row.amount)}"
                for row in unusual.itertuples(index=False)
            )
        )
    return "\n".join(lines)


def statement_period(df: pd.DataFrame) -> str:
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return "the uploaded statement"
    start = dates.min().date()
    end = dates.max().date()
    return str(start) if start == end else f"{start} to {end}"


def extract_month_filter_from_df(question: str, df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "month" not in df.columns:
        return None
    lower_question = question.lower()
    months_in_data = sorted(str(month) for month in df["month"].dropna().astype(str).unique())

    explicit = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", lower_question)
    if explicit:
        return f"{explicit.group(1)}-{int(explicit.group(2)):02d}"

    if re.search(r"\blatest month\b|\bmost recent month\b|\bcurrent month\b", lower_question):
        return months_in_data[-1] if months_in_data else None
    if re.search(r"\bprevious month\b|\blast month\b", lower_question):
        return months_in_data[-2] if len(months_in_data) >= 2 else (months_in_data[-1] if months_in_data else None)

    month_names = {
        "january": "01",
        "jan": "01",
        "february": "02",
        "feb": "02",
        "march": "03",
        "mar": "03",
        "april": "04",
        "apr": "04",
        "may": "05",
        "june": "06",
        "jun": "06",
        "july": "07",
        "jul": "07",
        "august": "08",
        "aug": "08",
        "september": "09",
        "sep": "09",
        "october": "10",
        "oct": "10",
        "november": "11",
        "nov": "11",
        "december": "12",
        "dec": "12",
    }
    for name, number in month_names.items():
        if re.search(rf"\b{name}\b", lower_question):
            matching = [month for month in months_in_data if month.endswith(f"-{number}")]
            return matching[-1] if matching else None
    return None


def apply_question_month_filter(question: str, df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, str | None, str | None]:
    if df is None or df.empty:
        return df, None, None
    month_filter = extract_month_filter_from_df(question, df)
    if not month_filter:
        return df, None, None
    filtered = df[df["month"].astype(str) == month_filter].copy()
    if filtered.empty:
        return filtered, month_filter, f"I could not find transactions for **{month_filter}** in the uploaded statements."
    return filtered, month_filter, None


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def apply_memory_exclusions(df: pd.DataFrame, memory: dict[str, Any]) -> pd.DataFrame:
    if df.empty or not memory.get("exclude_terms"):
        return df.copy()
    working = df.copy()
    searchable = (
        working["merchant"].fillna("").astype(str).str.lower()
        + " "
        + working["description"].fillna("").astype(str).str.lower()
    )
    mask = pd.Series(False, index=working.index)
    for term in memory.get("exclude_terms", []):
        clean_term = str(term).lower().strip()
        if clean_term:
            mask = mask | searchable.str.contains(re.escape(clean_term), na=False)
    return working[~mask].copy()


def recurring_expense_summary(expenses: pd.DataFrame) -> pd.DataFrame:
    if expenses.empty:
        return pd.DataFrame(columns=["merchant", "category", "count", "months", "total", "monthly_average"])
    grouped = (
        expenses.assign(merchant_key=expenses["merchant"].fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
        .groupby(["merchant_key", "category"], as_index=False)
        .agg(
            merchant=("merchant", "first"),
            count=("spend", "size"),
            months=("month", "nunique"),
            total=("spend", "sum"),
        )
    )
    grouped = grouped[(grouped["count"] >= 2) | (grouped["months"] >= 2)].copy()
    if grouped.empty:
        return grouped
    grouped["monthly_average"] = grouped["total"] / grouped["months"].clip(lower=1)
    return grouped.sort_values(["monthly_average", "total"], ascending=False)


def is_coaching_question(question: str) -> bool:
    normalized = question.lower()
    return any(
        term in normalized
        for term in [
            "save",
            "saving",
            "budget",
            "advice",
            "what should i do",
            "how can i",
            "reduce",
            "cut down",
            "spend less",
            "plan",
            "improve",
            "help me",
            "recommend",
        ]
    )


def dataframe_coach_answer(
    question: str,
    df: pd.DataFrame | None,
    user_context: str = "",
    memory: dict[str, Any] | None = None,
) -> str | None:
    if df is None or df.empty or not is_coaching_question(question):
        return None

    memory = memory or load_user_memory()
    working = apply_memory_exclusions(df, memory)
    expenses = working[working["amount"] < 0].copy()
    income = working[working["amount"] > 0].copy()
    if expenses.empty:
        return None

    total_spend = expenses["spend"].sum()
    total_income = income["amount"].sum()
    net = total_income - total_spend
    months_count = max(1, expenses["month"].nunique())
    monthly_spend = total_spend / months_count
    monthly_income = total_income / max(1, income["month"].nunique()) if not income.empty else 0.0
    savings_rate = (net / total_income * 100) if total_income else 0.0

    category_totals = expenses.groupby("category")["spend"].sum().sort_values(ascending=False)
    fixed_categories = {"Income", "Bills", "Rent"}
    flexible = category_totals[~category_totals.index.isin(fixed_categories)].head(4)
    recurring = recurring_expense_summary(expenses).head(5)
    largest = expenses.sort_values("spend", ascending=False).head(5)

    target_rows = []
    for category, value in flexible.items():
        reduction = value * 0.15
        if reduction > 0:
            target_rows.append((category, value, reduction))

    possible_monthly_saving = sum(item[2] for item in target_rows) / months_count if target_rows else 0.0

    period = statement_period(working)
    lines = [
        "**Short answer:** You have room to save by tightening flexible categories first.",
        "",
        f"For **{period}**, the statement shows **{format_sgd(total_income)} income**, **{format_sgd(total_spend)} spending**, and **{format_sgd(net)} net left over**. "
        f"Average spending is **{format_sgd(monthly_spend)} per month**"
        + (f", with a **{savings_rate:.1f}% savings rate**." if total_income else "."),
        "",
        "**Best next moves**",
    ]

    if target_rows:
        for category, value, reduction in target_rows[:3]:
            lines.append(
                f"- Put a cap on **{category}**: current spend is {format_sgd(value)}. A 15% cut saves about **{format_sgd(reduction / months_count)}/month**."
            )
    else:
        lines.append("- Label fixed costs, transfers, and rent first, then set targets only on flexible spending.")

    if not recurring.empty:
        lines.append(
            f"- Review repeated charges: **{recurring.iloc[0]['merchant']}** is about "
            f"**{format_sgd(float(recurring.iloc[0]['monthly_average']))}/month**."
        )

    if not largest.empty:
        biggest = largest.iloc[0]
        lines.append(
            f"- Review the largest item: **{biggest['merchant']}** on {biggest['date']} was **{format_sgd(abs(float(biggest['amount'])))}**. "
            "If it was a one-off or fixed cost, do not use it as the main savings target."
        )

    if possible_monthly_saving:
        lines.append(
            f"- A realistic first savings target is about **{format_sgd(possible_monthly_saving)}/month**."
        )

    lines.extend(["", "**Evidence used**"])
    for row in largest.itertuples(index=False):
        lines.append(f"- {row.date} | {row.merchant} | {row.category} | {format_sgd(abs(float(row.amount)))}")

    if user_context.strip() or memory.get("facts"):
        lines.extend(["", "**Personal context applied**"])
        if user_context.strip():
            lines.append(f"- Current context: {user_context.strip()[:500]}")
        for fact in memory.get("facts", [])[-4:]:
            lines.append(f"- {fact}")

    lines.extend(["", "_To personalize this, type something like `remember apartment rent is rent`._"])
    return "\n".join(lines)


def dataframe_direct_answer(
    question: str,
    df: pd.DataFrame | None,
    memory: dict[str, Any] | None = None,
) -> str | None:
    if df is None or df.empty:
        return None

    normalized = question.lower()
    memory = memory or load_user_memory()
    working = apply_memory_exclusions(df, memory)
    if any(
        term in normalized
        for term in [
            "compare",
            "month wise",
            "month-wise",
            "by month",
            "monthly breakdown",
            "monthly summary",
            "which month",
            "highest month",
            "most expensive month",
            "spending by month",
        ]
    ):
        monthly_answer = dataframe_monthly_summary_answer(working)
        if monthly_answer:
            return monthly_answer

    working, month_filter, month_error = apply_question_month_filter(question, working)
    if month_error:
        return month_error

    expenses = working[working["amount"] < 0].copy()
    income = working[working["amount"] > 0].copy()
    if expenses.empty and any(term in normalized for term in ["spend", "spent", "expense", "money"]):
        return "I do not see any expense rows in the uploaded statement."

    asks_top_spend = any(
        term in normalized
        for term in [
            "spend money most",
            "spend most",
            "spent most",
            "where do i spend",
            "where i spend",
            "highest spend",
            "top spend",
            "most money",
            "biggest expense",
        ]
    )
    asks_total = any(term in normalized for term in ["total spend", "total expense", "how much did i spend"])
    asks_income = any(term in normalized for term in ["income", "salary", "earned", "credit"])
    asks_subscription = any(term in normalized for term in ["subscription", "recurring", "monthly"])
    asks_transactions = any(
        term in normalized
        for term in ["show transactions", "list transactions", "all transactions", "show statement", "list statement"]
    )

    if asks_top_spend:
        category_totals = expenses.groupby("category")["spend"].sum().sort_values(ascending=False)
        merchant_totals = (
            expenses.groupby("merchant")["spend"]
            .agg(["sum", "count"])
            .sort_values(["sum", "count"], ascending=False)
            .head(5)
        )
        largest = expenses.sort_values("spend", ascending=False).head(5)
        top_category = category_totals.index[0]
        top_value = float(category_totals.iloc[0])

        period = statement_period(working)
        lines = [
            f"**Short answer:** You spend the most on **{top_category}**: **{format_sgd(top_value)}**.",
            f"This is based on the uploaded statement period **{period}**.",
            "",
            "**Spending breakdown**",
        ]
        for category, value in category_totals.head(6).items():
            share = (float(value) / expenses["spend"].sum() * 100) if expenses["spend"].sum() else 0.0
            lines.append(f"- **{category}**: {format_sgd(float(value))} ({share:.1f}% of spending)")

        lines.extend(["", "**Top merchants**"])
        for merchant, row in merchant_totals.iterrows():
            count = int(row["count"])
            lines.append(f"- **{merchant}**: {format_sgd(float(row['sum']))} across {count} {pluralize(count, 'transaction')}")

        lines.extend(["", "**Largest transactions checked**"])
        for row in largest.itertuples(index=False):
            lines.append(f"- {row.date} | {row.merchant} | {row.category} | {format_sgd(abs(float(row.amount)))}")
        return "\n".join(lines)

    if asks_total:
        total = expenses["spend"].sum()
        months = ", ".join(sorted(expenses["month"].dropna().astype(str).unique()))
        return (
            f"**Total spending:** {format_sgd(total)}\n\n"
            f"I counted **{len(expenses)} expense {pluralize(len(expenses), 'transaction')}** in the uploaded statement. "
            f"Months covered: **{months or 'unknown'}**."
        )

    if asks_income:
        total_income = income["amount"].sum()
        top_income = income.sort_values("amount", ascending=False).head(5)
        lines = [f"**Total income/credits:** {format_sgd(total_income)}", ""]
        if not top_income.empty:
            lines.append("**Income rows checked**")
            for row in top_income.itertuples(index=False):
                lines.append(f"- {row.date} | {row.merchant} | {format_sgd(float(row.amount))}")
        return "\n".join(lines)

    if asks_subscription:
        recurring = recurring_expense_summary(expenses).head(8)
        if recurring.empty:
            return "I did not find clear recurring expenses in the uploaded statement."
        lines = ["**Likely recurring expenses**", ""]
        for row in recurring.itertuples(index=False):
            lines.append(
                f"- **{row.merchant}**: {format_sgd(float(row.total))} total, about **{format_sgd(float(row.monthly_average))}/month**"
            )
        return "\n".join(lines)

    if asks_transactions or (month_filter and any(term in normalized for term in ["show", "list", "details", "statement"])):
        return dataframe_transaction_list_answer(working, month_filter)

    return None


def dataframe_transaction_list_answer(df: pd.DataFrame, month_filter: str | None = None) -> str:
    if df is None or df.empty:
        return "I could not find matching transactions in the uploaded statements."

    period = month_filter or statement_period(df)
    sorted_df = df.sort_values("date", ascending=True)
    total_spend = sorted_df.loc[sorted_df["amount"] < 0, "spend"].sum()
    total_income = sorted_df.loc[sorted_df["amount"] > 0, "amount"].sum()
    lines = [
        f"**Short answer:** I found **{len(sorted_df)} {pluralize(len(sorted_df), 'transaction')}** for **{period}**.",
        f"Income is **{format_sgd(float(total_income))}** and spending is **{format_sgd(float(total_spend))}**.",
        "",
        "**Transactions**",
    ]
    for row in sorted_df.head(25).itertuples(index=False):
        amount = abs(float(row.amount)) if float(row.amount) < 0 else float(row.amount)
        direction = "credit" if float(row.amount) > 0 else "debit"
        lines.append(f"- {row.date} | {row.merchant} | {row.category} | {direction} | {format_sgd(amount)}")
    if len(sorted_df) > 25:
        lines.append(f"- Showing first 25 of {len(sorted_df)} transactions. Use the Dashboard month filter to view every row.")
    return "\n".join(lines)


def dataframe_monthly_summary_answer(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "month" not in df.columns:
        return None
    rows = []
    for month, group in df.groupby("month"):
        expenses = group[group["amount"] < 0]
        income = group[group["amount"] > 0]
        spend = float(expenses["spend"].sum()) if not expenses.empty else 0.0
        income_total = float(income["amount"].sum()) if not income.empty else 0.0
        top_category = "None"
        if not expenses.empty:
            category_totals = expenses.groupby("category")["spend"].sum().sort_values(ascending=False)
            if not category_totals.empty:
                top_category = f"{category_totals.index[0]} ({format_sgd(float(category_totals.iloc[0]))})"
        rows.append(
            {
                "month": str(month),
                "income": income_total,
                "spend": spend,
                "net": income_total - spend,
                "transactions": len(group),
                "top_category": top_category,
            }
        )
    if not rows:
        return None
    summary = pd.DataFrame(rows).sort_values("month")
    highest = summary.sort_values("spend", ascending=False).iloc[0]
    lines = [
        f"**Short answer:** Highest spending was in **{highest['month']}** at **{format_sgd(float(highest['spend']))}**.",
        "",
        "**Monthly breakdown**",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- **{row.month}**: income {format_sgd(float(row.income))}, spending {format_sgd(float(row.spend))}, "
            f"net {format_sgd(float(row.net))}, top category {row.top_category}, {row.transactions} {pluralize(int(row.transactions), 'transaction')}"
        )
    return "\n".join(lines)


def build_chat_history(messages: list[dict[str, str]] | None, limit: int = 6) -> str:
    if not messages:
        return "No prior chat messages."
    recent = messages[-limit:]
    lines = []
    for message in recent:
        role = message.get("role", "message")
        content = re.sub(r"\s+", " ", message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:600]}")
    return "\n".join(lines) or "No prior chat messages."


def build_prompt(
    question: str,
    retrieved: list[RetrievedTransaction],
    user_context: str = "",
    messages: list[dict[str, str]] | None = None,
    df: pd.DataFrame | None = None,
    memory: dict[str, Any] | None = None,
) -> str:
    context = "\n".join(f"{idx + 1}. {item.document}" for idx, item in enumerate(retrieved))
    statement_summary = build_statement_summary(retrieved)
    dataframe_summary = build_dataframe_summary(df, memory)
    learned_memory = memory_to_text(memory or load_user_memory())
    user_context_text = user_context.strip() or "No extra user-provided context."
    chat_history = build_chat_history(messages)
    return f"""
You are Ask Guardian, a bank-statement assistant.

Rules:
1. Use ONLY the uploaded statement evidence.
2. Give a direct answer first.
3. Do NOT say vague things if exact totals are available.
4. Always include numbers, categories, merchants, and dates when possible.
5. If the dataframe summary gives stronger evidence than retrieved rows, use the dataframe summary.
6. Never mention ChromaDB, RAG, retrieval, model, or fallback.
7. Use this format:

**Short answer:** ...
**Breakdown**
- ...
**Evidence checked**
- date | merchant | category | amount

User question:
{question}

Whole uploaded statement summary:
{dataframe_summary}

Retrieved rows:
{context}

Recent chat:
{chat_history}

Answer:
"""

def ask_gemma(prompt: str) -> str | None:
    provider = os.getenv("GEMMA_PROVIDER", "transformers").lower()
    model = os.getenv("GEMMA_MODEL", DEFAULT_GEMMA_MODEL)

    if provider == "auto" or (provider == "transformers" and ":" in model):
        ollama_answer = ask_gemma_ollama(prompt, model)
        if ollama_answer:
            return ollama_answer
        if ":" in model:
            return None
        if provider == "auto":
            provider = "transformers"

    if provider == "ollama":
        return ask_gemma_ollama(prompt, model)

    try:
        from transformers import pipeline

        generator = pipeline("text-generation", model=model, device_map="auto")
        output = generator(prompt, max_new_tokens=500, do_sample=False, temperature=0.1)
        text = output[0]["generated_text"]
        return text.split("Answer:", 1)[-1].strip()
    except Exception as exc:
        st.warning(f"Gemma via transformers is unavailable: {exc}")
        return None


# def ask_gemma_ollama(prompt: str, model: str) -> str | None:
#     try:
#         response = requests.post(
#             os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
#             json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
#             timeout=120,
#         )
#         response.raise_for_status()
#         return response.json().get("response", "").strip()
#     except Exception as exc:
#         if os.getenv("GEMMA_PROVIDER", "transformers").lower() == "ollama":
#             st.warning(f"Gemma via Ollama is unavailable: {exc}")
#         return None


# def looks_like_weak_answer(answer: str) -> bool:
#     normalized = answer.lower()
#     if len(answer.strip()) < 120:
#         return True
#     return (
#         "based only on the retrieved statement rows" in normalized
#         and "what i would do first" not in normalized
#         and "practical read" not in normalized
#     )
def ask_gemma_ollama(prompt: str, model: str) -> str | None:
    try:
        response = requests.post(
            os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "num_ctx": 8192,
                    "num_predict": 700,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return clean_llm_answer(answer)
    except Exception as exc:
        st.warning(f"Gemma via Ollama is unavailable: {exc}")
        return None


def clean_llm_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^Answer:\s*", "", answer, flags=re.I)
    banned = ["ChromaDB", "retrieval", "vector database", "language model", "LLM"]
    for word in banned:
        answer = answer.replace(word, "uploaded statement")
    return answer


def looks_like_weak_answer(answer: str) -> bool:
    if not answer or len(answer.strip()) < 180:
        return True

    normalized = answer.lower()

    weak_phrases = [
        "i don't have enough information",
        "not enough detail",
        "cannot determine",
        "based only on the retrieved",
        "uploaded statement data does not contain enough evidence",
    ]

    has_money = bool(re.search(r"\$|s\$|\d+\.\d{2}", answer.lower()))
    has_evidence = any(word in normalized for word in ["date", "merchant", "category", "amount", "evidence"])

    return any(p in normalized for p in weak_phrases) and not (has_money and has_evidence)

def question_needs_followups(question: str, retrieved: list[RetrievedTransaction]) -> bool:
    normalized = question.lower()
    if len(normalized.split()) <= 5:
        return True
    if is_overview_question(question):
        return True
    if not retrieved:
        return True
    return any(term in normalized for term in ["help", "advice", "why", "better", "reduce", "save", "budget"])


def generate_follow_up_questions(question: str, retrieved: list[RetrievedTransaction], user_context: str = "") -> list[str]:
    rows_text = " ".join(
        f"{item.metadata.get('merchant', '')} {item.metadata.get('category', '')} {item.metadata.get('month', '')}"
        for item in retrieved
    ).lower()
    followups = []
    if not user_context.strip():
        followups.append("Should I treat transfers and shared expenses as real spending or exclude them?")
    if not retrieved:
        followups.append("Which date range, merchant, category, or amount should I focus on?")
        return followups
    if question_needs_followups(question, retrieved):
        followups.append("Should I focus next on reducing spending, finding unusual transactions, or explaining monthly patterns?")
    if "income" not in rows_text:
        followups.append("Is all income included in this statement, or is income split across another account?")
    if any(term in rows_text for term in ["transfer", "paynow", "paylah", "atm"]):
        followups.append("Are transfer or cash-withdrawal rows personal spending, account movement, or payments to someone else?")
    return followups[:1]


def append_followups(answer: str, question: str, retrieved: list[RetrievedTransaction], user_context: str = "") -> str:
    questions = generate_follow_up_questions(question, retrieved, user_context)
    if not questions:
        return answer
    followup_text = "\n".join(f"- {item}" for item in questions)
    return f"{answer.rstrip()}\n\n**One thing to confirm**\n{followup_text}"


def evidence_only_answer(question: str, retrieved: list[RetrievedTransaction], user_context: str = "") -> str:
    if not retrieved:
        return append_followups(
            "I could not find matching transactions in the uploaded statement for that question.",
            question,
            retrieved,
            user_context,
        )

    total_spend = sum(float(item.metadata.get("spend", 0.0)) for item in retrieved)
    lines = [
        "**Short answer:** I found related transactions, but not enough detail for a fuller conclusion.",
        "",
        "**Closest matches**",
    ]
    for item in retrieved[:8]:
        meta = item.metadata
        amount = float(meta.get("amount", 0.0))
        lines.append(
            f"- {meta.get('date')} | {meta.get('merchant')} | {meta.get('category')} | {format_sgd(amount)}"
        )
    if total_spend:
        lines.extend(["", f"Retrieved expense total: {format_sgd(total_spend)}."])
    if user_context.strip():
        lines.extend(["", "I also have your extra context, but the transaction evidence above is what the statement proves."])
    lines.append("Only uploaded statement rows were used.")
    return append_followups("\n".join(lines), question, retrieved, user_context)


def answer_question(
    question: str,
    user_context: str = "",
    messages: list[dict[str, str]] | None = None,
    df: pd.DataFrame | None = None,
) -> tuple[str, list[RetrievedTransaction]]:
    learned = remember_user_text(question)
    memory = load_user_memory()
    if learned:
        return (
            "Got it. I learned this and will use it in future answers:\n\n"
            f"- {learned}\n\n"
            "Ask your next question and I will apply it to the uploaded statement.",
            [],
        )

    filtered_df, month_filter, month_error = apply_question_month_filter(question, df)
    if month_error:
        return month_error, []
    answer_df = filtered_df if filtered_df is not None else df

    direct_answer = dataframe_direct_answer(question, df, memory=memory)
    if direct_answer:
        return direct_answer, []

    coach_answer = dataframe_coach_answer(question, answer_df, user_context=user_context, memory=memory)

    retrieved = retrieve_transactions(question)
    if month_filter:
        retrieved = [
            row
            for row in retrieved
            if str(row.metadata.get("month", "")) == month_filter
        ]

    if not retrieved:
        if coach_answer:
            return coach_answer, retrieved
        return evidence_only_answer(question, retrieved, user_context), retrieved

    prompt = build_prompt(
        question,
        retrieved,
        user_context=user_context,
        messages=messages,
        df=answer_df,
        memory=memory,
    )
    answer = ask_gemma(prompt)
    if not answer or (coach_answer and looks_like_weak_answer(answer)):
        answer = coach_answer or evidence_only_answer(question, retrieved, user_context)
    else:
        answer = append_followups(answer, question, retrieved, user_context)
    return answer, retrieved


@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False)


def extract_text_from_image(uploaded_image) -> str:
    import numpy as np

    image = Image.open(uploaded_image).convert("RGB")
    reader = get_ocr_reader()
    results = reader.readtext(np.array(image))
    return "\n".join(text for _, text, confidence in results if confidence >= 0.25).strip()


def fetch_url_text(url: str) -> str:
    if not url:
        return ""
    if not re.match(r"^https?://", url, flags=re.I):
        url = f"https://{url}"
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "AskGuardian/1.0"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text" not in content_type and "html" not in content_type:
        return f"Fetched URL but content type is {content_type or 'unknown'}, not readable text."
    parser = TextOnlyHTMLParser()
    parser.feed(response.text[:250_000])
    return parser.text()[:12_000]


def combine_safety_inputs(text: str, url: str, uploaded_image) -> tuple[str, str]:
    pieces = []
    extracted_text = ""

    if text.strip():
        pieces.append(f"User pasted message/text:\n{text.strip()}")

    if url.strip():
        try:
            url_text = fetch_url_text(url.strip())
            pieces.append(f"Text fetched from URL {url.strip()}:\n{url_text}")
        except Exception as exc:
            pieces.append(f"URL fetch failed for {url.strip()}: {exc}")

    if uploaded_image:
        try:
            extracted_text = extract_text_from_image(uploaded_image)
            pieces.append(f"OCR text extracted from screenshot:\n{extracted_text or '[No readable text found]'}")
        except Exception as exc:
            pieces.append(f"OCR failed: {exc}")

    return "\n\n".join(pieces).strip(), extracted_text


def assess_safety(text: str, mode: str) -> SafetyAssessment:
    normalized = text.lower()
    red_flags = []
    reassuring = []
    score = 0

    for flag, terms in SCAM_RED_FLAGS.items():
        hits = [term for term in terms if term in normalized]
        if hits:
            red_flags.append(f"{flag}: {', '.join(hits[:4])}")
            score += 18 if "OTP" in flag or "unrealistic" in flag else 14

    links = re.findall(r"https?://[^\s)>\"]+|www\.[^\s)>\"]+", text, flags=re.I)
    if links:
        score += 8
        official_links = [
            link
            for link in links
            if re.search(r"\.gov\b|\.gov\.|\.edu\b|\.edu\.|mas\.gov\.sg|sec\.gov|fca\.org\.uk", link, flags=re.I)
        ]
        if official_links:
            reassuring.append("Contains at least one government, education, or regulator-style official domain.")
        suspicious_links = [link for link in links if re.search(r"bit\.ly|tinyurl|login|verify|secure|account", link, flags=re.I)]
        if suspicious_links:
            red_flags.append("Contains link wording commonly used in credential collection.")
            score += 12

    phone_like = re.findall(r"(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){7,14}\d", text)
    if phone_like:
        reassuring.append("Contains a phone number, but this alone does not prove legitimacy.")

    for signal, terms in FINANCE_SIGNALS.items():
        hits = [term for term in terms if term in normalized]
        if hits:
            reassuring.append(f"{signal}: {', '.join(hits[:3])}")

    if any(
        term in normalized
        for term in ["official register", "regulator register", "financial institutions directory", "mas register", "fca register", "sec adviser"]
    ):
        reassuring.append("Mentions checking an official regulator register.")

    if mode == "finance" and any(term in normalized for term in ["guaranteed", "risk free", "daily profit", "crypto", "forex"]):
        score += 18

    if not text.strip():
        return SafetyAssessment(
            verdict="No content to check",
            risk_score=0,
            red_flags=[],
            reassuring_signals=[],
            recommendation="Paste a message, enter a URL, or upload a screenshot.",
            extracted_text="",
        )

    score = min(score, 100)
    if score >= 60:
        verdict = "High risk / likely scam"
        recommendation = (
            "Do not click links, do not reply, and do not share OTP, identity, card, or banking details. "
            "For finance services, verify independently on the relevant official regulator register in your country."
        )
    elif score >= 30:
        verdict = "Suspicious / verify first"
        recommendation = (
            "Pause and verify through the official app, official website typed manually, or a known hotline. "
            "Do not rely on links or phone numbers inside the message."
        )
    else:
        verdict = "No strong scam signals found"
        recommendation = (
            "No strong red flags were found in the submitted content, but this is not proof of legitimacy. "
            "For finance services, still verify licensing and company identity independently."
        )

    return SafetyAssessment(
        verdict=verdict,
        risk_score=score,
        red_flags=red_flags,
        reassuring_signals=reassuring,
        recommendation=recommendation,
        extracted_text=text,
    )


def explain_safety_with_gemma(content: str, assessment: SafetyAssessment, mode: str) -> str:
    prompt = f"""You are Ask Guardian, helping users evaluate scam and finance-service risk.
Use ONLY the submitted content and the rule-based assessment below.
Do not claim that a message or finance service is definitely legitimate.
Do not browse or invent licensing facts.
For finance services, tell the user to verify the company on the relevant official regulator register and avoid sharing OTP, identity, card, or banking details.

Mode: {mode}
Rule verdict: {assessment.verdict}
Risk score: {assessment.risk_score}/100
Red flags: {assessment.red_flags or ['None found']}
Reassuring signals: {assessment.reassuring_signals or ['None found']}

Submitted content:
{content[:8000]}

Answer with:
1. Verdict
2. Why
3. What to do next"""
    answer = ask_gemma(prompt)
    if answer:
        return answer
    return format_safety_answer(assessment)


def format_safety_answer(assessment: SafetyAssessment) -> str:
    red_flags = "\n".join(f"- {flag}" for flag in assessment.red_flags) or "- None found in submitted text."
    reassuring = "\n".join(f"- {signal}" for signal in assessment.reassuring_signals) or "- None strong enough to prove legitimacy."
    return (
        f"**Verdict:** {assessment.verdict}\n\n"
        f"**Risk score:** {assessment.risk_score}/100\n\n"
        f"**Red flags**\n{red_flags}\n\n"
        f"**Signals checked**\n{reassuring}\n\n"
        f"**Next step:** {assessment.recommendation}"
    )


def transaction_option_label(row: pd.Series) -> str:
    return f"{row['date']} | {row['merchant']} | {format_sgd(float(row['amount']))}"


def explain_transaction(row: pd.Series, df: pd.DataFrame) -> str:
    amount = float(row["amount"])
    spend = abs(amount) if amount < 0 else 0.0
    transaction_type = "income or credit" if amount > 0 else "expense"
    category = str(row.get("category", "Other"))
    merchant = str(row.get("merchant", "Unknown merchant"))
    description = str(row.get("description", ""))
    month = str(row.get("month", ""))

    expenses = df[df["amount"] < 0].copy()
    same_category = expenses[expenses["category"] == category]
    same_merchant = df[df["merchant"].astype(str).str.lower() == merchant.lower()]
    total_spend = expenses["spend"].sum()
    month_spend = expenses.loc[expenses["month"].astype(str) == month, "spend"].sum() if month else 0.0
    category_spend = same_category["spend"].sum() if not same_category.empty else 0.0

    lines = [
        f"**Short answer:** **{merchant}** is an **{transaction_type}** for **{format_sgd(abs(amount) if amount < 0 else amount)}** on **{row['date']}**.",
        "",
        "**Details**",
        f"- Statement description: {description}",
        f"- Assigned category: **{category}**",
    ]

    if amount < 0:
        share_total = (spend / total_spend * 100) if total_spend else 0.0
        share_month = (spend / month_spend * 100) if month_spend else 0.0
        share_category = (spend / category_spend * 100) if category_spend else 0.0
        lines.extend(
            [
                f"- It is {share_total:.1f}% of total spending in the uploaded statement.",
                f"- It is {share_month:.1f}% of spending in {month}." if month else "- Month is not available.",
                f"- It is {share_category:.1f}% of your {category} spending.",
            ]
        )

        if not expenses.empty:
            rank = int((expenses["spend"] > spend).sum() + 1)
            lines.append(f"- By amount, it ranks **#{rank} of {len(expenses)}** expenses.")
    else:
        total_income = df.loc[df["amount"] > 0, "amount"].sum()
        share_income = (amount / total_income * 100) if total_income else 0.0
        lines.append(f"- It is {share_income:.1f}% of total income/credits in the uploaded statement.")

    if len(same_merchant) > 1:
        merchant_total = same_merchant["amount"].sum()
        lines.append(
            f"- This merchant/description appears {len(same_merchant)} times, net total {format_sgd(float(merchant_total))}."
        )

    if category == "Other":
        lines.append("- This is marked **Other** because it did not match the current category rules. You can teach a correction in chat, for example `remember this merchant is rent`.")

    return "\n".join(lines)


def detect_unusual_spending(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "amount" not in df.columns:
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "reason", "severity"])

    expenses = df[df["amount"] < 0].copy()
    if expenses.empty:
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "reason", "severity"])

    expenses["spend"] = expenses.get("spend", expenses["amount"].abs())
    category_average = expenses.groupby("category")["spend"].transform("mean")
    merchant_average = expenses.groupby("merchant")["spend"].transform("mean")
    merchant_count = expenses.groupby("merchant")["spend"].transform("size")
    global_threshold = max(float(expenses["spend"].quantile(0.85)), float(expenses["spend"].mean() + expenses["spend"].std(ddof=0)))

    rows = []
    for _, row in expenses.iterrows():
        spend = float(row.get("spend", 0.0))
        if spend <= 0:
            continue

        category_avg = float(category_average.loc[row.name]) if pd.notna(category_average.loc[row.name]) else 0.0
        merchant_avg = float(merchant_average.loc[row.name]) if pd.notna(merchant_average.loc[row.name]) else 0.0
        merchant_rows = int(merchant_count.loc[row.name]) if pd.notna(merchant_count.loc[row.name]) else 0
        category_ratio = spend / category_avg if category_avg > 0 else 0.0
        merchant_ratio = spend / merchant_avg if merchant_avg > 0 and merchant_rows >= 2 else 0.0

        reasons = []
        severity_score = 0
        if category_ratio >= 2.0 and spend >= 20:
            reasons.append(f"about {category_ratio:.1f}x higher than your usual {row.get('category', 'category')} amount")
            severity_score += 2
        if merchant_ratio >= 2.0 and spend >= 20:
            reasons.append(f"about {merchant_ratio:.1f}x higher than your usual {row.get('merchant', 'merchant')} amount")
            severity_score += 2
        if spend >= global_threshold and spend >= 20:
            reasons.append("one of the largest expenses in this statement")
            severity_score += 1

        if reasons:
            severity = "High" if severity_score >= 3 else "Watch"
            rows.append(
                {
                    "date": row.get("date"),
                    "merchant": row.get("merchant", "Unknown"),
                    "category": row.get("category", "Other"),
                    "amount": spend,
                    "reason": f"{row.get('merchant', 'This transaction')} on {row.get('date')} was {format_sgd(spend)}, "
                    + "; ".join(reasons)
                    + ".",
                    "severity": severity,
                }
            )

    unusual = pd.DataFrame(rows)
    if unusual.empty:
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "reason", "severity"])
    return unusual.sort_values(["severity", "amount"], ascending=[True, False]).head(10).reset_index(drop=True)


def calculate_guardian_score(df: pd.DataFrame) -> dict[str, Any]:
    score = 100
    positive_factors = []
    risk_factors = []
    recommended_actions = []

    if df is None or df.empty:
        return {
            "score": 0,
            "rating": "No data",
            "positive_factors": [],
            "risk_factors": ["Upload a statement to calculate the Guardian Score."],
            "recommended_actions": ["Upload a CSV, XLS, or XLSX bank statement."],
        }

    expenses = df[df["amount"] < 0].copy()
    income = df[df["amount"] > 0].copy()
    total_spend = float(expenses.get("spend", expenses["amount"].abs()).sum()) if not expenses.empty else 0.0
    total_income = float(income["amount"].sum()) if not income.empty else 0.0
    savings_rate = ((total_income - total_spend) / total_income) if total_income > 0 else None

    if savings_rate is None:
        score -= 10
        risk_factors.append("No income credits were found, so savings rate cannot be confirmed.")
        recommended_actions.append("Upload the account that receives income, or teach Ask Guardian which credits count as income.")
    elif savings_rate < 0:
        score -= 30
        risk_factors.append("Spending is higher than income in this statement period.")
        recommended_actions.append("Review the largest categories first and separate one-off expenses from recurring commitments.")
    elif savings_rate < 0.10:
        score -= 22
        risk_factors.append(f"Savings rate is low at {savings_rate * 100:.1f}%.")
        recommended_actions.append("Set a small first savings target, such as reducing flexible spending by 10%.")
    elif savings_rate < 0.20:
        score -= 12
        risk_factors.append(f"Savings rate is moderate at {savings_rate * 100:.1f}%.")
    else:
        positive_factors.append(f"Savings rate is healthy at {savings_rate * 100:.1f}%.")

    if total_income > 0:
        spend_ratio = total_spend / total_income
        if spend_ratio >= 1.0:
            score -= 18
            risk_factors.append("Total spending is equal to or above total income.")
        elif spend_ratio >= 0.85:
            score -= 10
            risk_factors.append("Total spending is close to income.")
        else:
            positive_factors.append("Spending is below income for this statement period.")

    flexible_categories = {"Food", "Shopping", "Transport", "Subscriptions"}
    flexible_spend = expenses.loc[expenses["category"].isin(flexible_categories), "spend"].sum() if not expenses.empty else 0.0
    flexible_share = flexible_spend / total_spend if total_spend > 0 else 0.0
    if flexible_share >= 0.45:
        score -= 12
        risk_factors.append(f"Flexible categories make up {flexible_share * 100:.1f}% of spending.")
        recommended_actions.append("Set category caps for Food, Shopping, Transport, and Subscriptions.")
    elif flexible_share > 0:
        positive_factors.append(f"Flexible categories are {flexible_share * 100:.1f}% of spending.")

    recurring = recurring_expense_summary(expenses)
    recurring_count = len(recurring)
    if recurring_count >= 6:
        score -= 10
        risk_factors.append(f"{recurring_count} repeated or recurring expenses were detected.")
        recommended_actions.append("Review recurring charges and cancel anything no longer used.")
    elif recurring_count >= 3:
        score -= 5
        risk_factors.append(f"{recurring_count} repeated expenses were detected.")
    else:
        positive_factors.append("Recurring expense load looks manageable.")

    unusual = detect_unusual_spending(df)
    unusual_count = len(unusual)
    if unusual_count >= 5:
        score -= 12
        risk_factors.append(f"{unusual_count} unusual high-value transactions were detected.")
        recommended_actions.append("Review unusual transactions and mark one-offs so future analysis is cleaner.")
    elif unusual_count > 0:
        score -= 6
        risk_factors.append(f"{unusual_count} unusual high-value {pluralize(unusual_count, 'transaction')} detected.")

    if total_spend == 0:
        positive_factors.append("No expenses were found in this upload.")

    score = max(0, min(100, int(round(score))))
    if score >= 85:
        rating = "Strong"
    elif score >= 70:
        rating = "Good"
    elif score >= 50:
        rating = "Watch"
    else:
        rating = "Needs attention"

    if not recommended_actions:
        recommended_actions.append("Keep monitoring category trends and review unusual transactions each statement cycle.")
    if not positive_factors:
        positive_factors.append("Statement was parsed successfully and is ready for review.")

    return {
        "score": score,
        "rating": rating,
        "positive_factors": positive_factors[:4],
        "risk_factors": risk_factors[:5] or ["No major risk factors detected in this statement."],
        "recommended_actions": recommended_actions[:5],
    }


def generate_behavioral_insights(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "amount" not in df.columns:
        return ["Upload a statement to generate behavioral spending insights."]

    expenses = df[df["amount"] < 0].copy()
    income = df[df["amount"] > 0].copy()
    if expenses.empty:
        return ["No expense transactions were found, so spending pattern insights are limited."]

    insights = []
    expense_dates = pd.to_datetime(expenses["date"], errors="coerce")
    expenses = expenses.assign(parsed_date=expense_dates)
    dated_expenses = expenses.dropna(subset=["parsed_date"]).copy()

    if not dated_expenses.empty:
        dated_expenses["is_weekend"] = dated_expenses["parsed_date"].dt.dayofweek >= 5
        weekend_spend = float(dated_expenses.loc[dated_expenses["is_weekend"], "spend"].sum())
        weekday_spend = float(dated_expenses.loc[~dated_expenses["is_weekend"], "spend"].sum())
        if weekend_spend > 0 and weekday_spend > 0:
            weekend_days = max(1, dated_expenses.loc[dated_expenses["is_weekend"], "parsed_date"].dt.date.nunique())
            weekday_days = max(1, dated_expenses.loc[~dated_expenses["is_weekend"], "parsed_date"].dt.date.nunique())
            weekend_daily = weekend_spend / weekend_days
            weekday_daily = weekday_spend / weekday_days
            if weekend_daily >= weekday_daily * 1.5:
                insights.append("Weekend spending is noticeably higher than weekday spending.")
            elif weekend_spend >= float(dated_expenses["spend"].sum()) * 0.35:
                insights.append("A meaningful share of spending happens on weekends.")

        high_value_threshold = dated_expenses["spend"].quantile(0.80)
        high_value = dated_expenses[dated_expenses["spend"] >= high_value_threshold]
        if not high_value.empty and high_value["is_weekend"].mean() >= 0.5:
            insights.append("Most high-value spending happens on weekends.")

        income_dates = pd.to_datetime(income["date"], errors="coerce").dropna()
        if not income_dates.empty:
            post_income_spend = 0.0
            for income_date in income_dates:
                window = (dated_expenses["parsed_date"] > income_date) & (dated_expenses["parsed_date"] <= income_date + pd.Timedelta(days=3))
                post_income_spend += float(dated_expenses.loc[window, "spend"].sum())
            if post_income_spend >= float(dated_expenses["spend"].sum()) * 0.25:
                insights.append("Spending tends to increase in the few days after income credits.")

    time_text = (
        expenses.get("description", pd.Series(dtype=str)).fillna("").astype(str)
        + " "
        + expenses.get("merchant", pd.Series(dtype=str)).fillna("").astype(str)
    )
    time_matches = time_text.str.extract(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b")
    if not time_matches.dropna().empty:
        hours = pd.to_numeric(time_matches[0], errors="coerce")
        late_night_count = int(((hours >= 22) | (hours <= 4)).sum())
        if late_night_count >= 2:
            insights.append(f"{late_night_count} transactions appear to happen late at night based on available time text.")

    category_counts = expenses.groupby("category").size()
    if int(category_counts.get("Food", 0)) >= 4:
        insights.append("Food-related spending repeats often in this statement.")
    if int(category_counts.get("Shopping", 0)) >= 3:
        insights.append("Shopping appears repeatedly, so it may be useful to set a category cap.")

    delivery_terms = r"grabfood|foodpanda|deliveroo|delivery|takeaway|takeout"
    delivery_count = int(time_text.str.contains(delivery_terms, flags=re.I, regex=True).sum())
    if delivery_count >= 3:
        insights.append(f"Food delivery appears {delivery_count} times in this statement.")

    if not insights:
        insights.append("No strong behavioral pattern stood out; spending looks relatively spread across the statement period.")
    return insights[:5]


def render_guardian_intelligence_report(df: pd.DataFrame):
    st.subheader("Guardian Intelligence Report")
    score = calculate_guardian_score(df)
    unusual = detect_unusual_spending(df)
    insights = generate_behavioral_insights(df)

    score_cols = st.columns([1, 2, 2])
    score_cols[0].metric("Guardian Score", f"{score['score']}/100", score["rating"])
    score_cols[1].markdown("**Positive factors**")
    score_cols[1].markdown("\n".join(f"- {item}" for item in score["positive_factors"]))
    score_cols[2].markdown("**Key risk factors**")
    score_cols[2].markdown("\n".join(f"- {item}" for item in score["risk_factors"]))

    st.markdown("**Unusual transactions**")
    if unusual.empty:
        st.success("No unusual high-value transactions detected.")
    else:
        display_unusual = unusual.copy()
        display_unusual["Amount"] = display_unusual["amount"].apply(format_sgd)
        st.dataframe(
            display_unusual[["date", "merchant", "category", "Amount", "severity", "reason"]],
            use_container_width=True,
            hide_index=True,
        )

    insight_cols = st.columns(2)
    insight_cols[0].markdown("**Behavioral insights**")
    insight_cols[0].markdown("\n".join(f"- {item}" for item in insights))
    insight_cols[1].markdown("**Recommended actions**")
    insight_cols[1].markdown("\n".join(f"- {item}" for item in score["recommended_actions"]))


def render_dashboard(df: pd.DataFrame):
    available_months = sorted(df["month"].dropna().astype(str).unique()) if "month" in df.columns else []
    month_options = ["All months"] + available_months
    selected_month = st.selectbox("Statement month", month_options, key="dashboard_month_filter")
    filtered_df = df.copy()
    if selected_month != "All months":
        filtered_df = df[df["month"].astype(str) == selected_month].copy()

    st.caption(
        f"Showing {len(filtered_df)} transaction {pluralize(len(filtered_df), 'row')} "
        f"for {selected_month.lower() if selected_month == 'All months' else selected_month}."
    )

    expenses = filtered_df[filtered_df["amount"] < 0].copy()
    total_spend = expenses["spend"].sum()
    total_income = filtered_df.loc[filtered_df["amount"] > 0, "amount"].sum()
    txn_count = len(filtered_df)
    top_category = expenses.groupby("category")["spend"].sum().sort_values(ascending=False)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Transactions", f"{txn_count:,}")
    metric_cols[1].metric("Total spend", format_sgd(total_spend))
    metric_cols[2].metric("Income", format_sgd(total_income))
    metric_cols[3].metric("Top category", top_category.index[0] if not top_category.empty else "None")

    render_guardian_intelligence_report(filtered_df)

    chart_cols = st.columns(2)
    if not expenses.empty:
        by_category = expenses.groupby("category", as_index=False)["spend"].sum().sort_values("spend")
        chart_cols[0].plotly_chart(
            px.bar(
                by_category,
                x="spend",
                y="category",
                orientation="h",
                title="Spending by category",
                labels={"spend": f"Spend ({CURRENCY_CODE})", "category": "Category"},
            ),
            use_container_width=True,
        )

        by_month = expenses.groupby("month", as_index=False)["spend"].sum()
        chart_cols[1].plotly_chart(
            px.line(
                by_month,
                x="month",
                y="spend",
                markers=True,
                title="Monthly spending",
                labels={"spend": f"Spend ({CURRENCY_CODE})", "month": "Month"},
            ),
            use_container_width=True,
        )

    st.subheader("All transactions")
    st.caption("Tick one row in the Choose column to explain that transaction.")

    table_columns = ["date", "merchant", "description", "category", "amount", "type"]
    if "source_file" in filtered_df.columns:
        table_columns.append("source_file")

    display_df = filtered_df[table_columns].sort_values("date", ascending=False).copy()
    display_df["_row_id"] = display_df.index
    amount_label = f"Amount ({CURRENCY_CODE})"
    display_df = display_df.assign(
        Choose=False,
        **{amount_label: display_df["amount"].apply(format_sgd)},
    ).drop(columns=["amount"])

    visible_columns = ["Choose", "date", "merchant", "description", "category", amount_label, "type"]
    if "source_file" in display_df.columns:
        visible_columns.append("source_file")

    edited_df = st.data_editor(
        display_df[visible_columns + ["_row_id"]],
        use_container_width=True,
        hide_index=True,
        disabled=[column for column in visible_columns + ["_row_id"] if column != "Choose"],
        column_order=visible_columns,
        column_config={
            "Choose": st.column_config.CheckboxColumn("Choose", help="Select one transaction to explain."),
            "_row_id": None,
        },
        key=f"dashboard_transactions_table_{selected_month}",
    )

    st.subheader("Explain this transaction")
    chosen_rows = edited_df[edited_df["Choose"]]
    if chosen_rows.empty:
        st.info("Choose a transaction from the table above to see the explanation.")
    else:
        if len(chosen_rows) > 1:
            st.caption("Multiple rows are checked. Explaining the first selected transaction.")
        selected_index = int(chosen_rows.iloc[0]["_row_id"])
        st.markdown(explain_transaction(df.loc[selected_index], filtered_df))


def render_chat():
    st.subheader("Ask Guardian")
    st.caption("Ask in one box. The answer is calculated from the uploaded CSV/XLSX, and you can teach corrections by saying `remember ...`.")

    question = st.chat_input("Ask about your uploaded statement")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Reading your uploaded statement..."):
            answer, _ = answer_question(
                question,
                user_context="",
                messages=st.session_state.get("messages", []),
                df=st.session_state.get("statement_df"),
            )
        st.session_state.messages.append({"role": "assistant", "content": answer})

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_message_checker():
    st.subheader("Message Legitimacy Check")
    st.caption("Paste an SMS, email, WhatsApp, Telegram, or Gmail message. You can also upload a screenshot for OCR.")

    message = st.text_area(
        "Message text",
        height=180,
        placeholder="Paste the full message here, including sender name, links, phone numbers, and payment instructions.",
    )
    screenshot = st.file_uploader(
        "Upload message screenshot for OCR",
        type=["png", "jpg", "jpeg", "webp"],
        key="message_ocr_upload",
    )

    if st.button("Check message", type="primary"):
        combined, ocr_text = combine_safety_inputs(message, "", screenshot)
        assessment = assess_safety(combined, mode="message")
        answer = explain_safety_with_gemma(combined, assessment, mode="message")
        st.markdown(answer)
        if ocr_text:
            with st.expander("OCR extracted text"):
                st.text(ocr_text)

        st.warning("Do not share OTP, identity, card, or banking details. Verify through the official app or a website you type manually.")


def render_finance_service_checker():
    st.subheader("Finance Service Check")
    st.caption("Check a finance service, investment offer, loan message, website, or screenshot for scam and licensing risk signals.")

    url = st.text_input("Service link or website", placeholder="https://example.com")
    service_text = st.text_area(
        "Service description or offer text",
        height=160,
        placeholder="Paste the finance service pitch, investment return promise, loan offer, or website text.",
    )
    screenshot = st.file_uploader(
        "Upload service screenshot for OCR",
        type=["png", "jpg", "jpeg", "webp"],
        key="finance_ocr_upload",
    )

    if st.button("Check finance service", type="primary"):
        combined, ocr_text = combine_safety_inputs(service_text, url, screenshot)
        assessment = assess_safety(combined, mode="finance")
        answer = explain_safety_with_gemma(combined, assessment, mode="finance")
        st.markdown(answer)

        st.info(
            "Finance check: independently search the exact company name on the relevant official regulator register, "
            "check the legal name/company ID, and avoid services promising guaranteed or risk-free returns."
        )
        if ocr_text:
            with st.expander("OCR extracted text"):
                st.text(ocr_text)


def main():
    st.set_page_config(page_title="Ask Guardian", page_icon="🛡️", layout="wide")
    st.title("Ask Guardian")
    st.write("Upload a bank statement to build a ChromaDB memory, then ask questions grounded in the uploaded rows.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_context" not in st.session_state:
        st.session_state.user_context = ""

    with st.sidebar:
        st.header("Statement")
        uploaded_files = st.file_uploader(
            "Upload bank statements",
            type=["csv", "xls", "xlsx"],
            accept_multiple_files=True,
        )
        st.caption("Upload one statement or several monthly statements. Ask Guardian combines them into one history.")
        st.divider()
        st.header("Model")
        st.text(f"Provider: {os.getenv('GEMMA_PROVIDER', 'transformers')}")
        st.text(f"Model: {os.getenv('GEMMA_MODEL', DEFAULT_GEMMA_MODEL)}")
        st.text(f"Currency: {CURRENCY_CODE}")
        if st.button("Clear chat context"):
            st.session_state.messages = []
            st.session_state.user_context = ""
        if st.button("Clear learned memory"):
            save_user_memory(default_memory())
            st.session_state.messages = []

    if uploaded_files:
        try:
            df = read_many_statements(uploaded_files)
            if df.empty:
                st.error("No usable transactions were found in the uploaded statements.")
                return
            st.session_state.statement_df = df
            index_statement(df)
            months = ", ".join(sorted(df["month"].dropna().astype(str).unique()))
            file_count = len(uploaded_files)
            st.success(
                f"Indexed {len(df)} transaction rows from {file_count} {pluralize(file_count, 'file')}. "
                f"Statement months: {months or 'unknown'}."
            )
        except Exception as exc:
            st.error(str(exc))
            return

    df = st.session_state.get("statement_df")
    tab_dashboard, tab_chat, tab_message, tab_finance = st.tabs(
        ["Dashboard", "Ask Guardian", "Message Check", "Finance Service Check"]
    )
    with tab_dashboard:
        if df is None:
            st.info("Upload a bank statement to create the dashboard and Ask Guardian retrieval memory.")
        else:
            render_dashboard(df)
    with tab_chat:
        if df is None:
            st.info("Upload a bank statement before asking transaction questions.")
        else:
            render_chat()
    with tab_message:
        render_message_checker()
    with tab_finance:
        render_finance_service_checker()


if __name__ == "__main__":
    main()
