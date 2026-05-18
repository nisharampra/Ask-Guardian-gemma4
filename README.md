# Ask Guardian 🛡️

## Local AI Financial Safety Assistant

A Gemma-powered assistant that helps users understand bank statements, detect suspicious finance messages, and make safer financial decisions locally.

---

## 🎥 Demo Video

YouTube Demo: https://your-youtube-link-here

---

# Overview

Ask Guardian is a local-first AI financial safety assistant designed to help users:

* Understand bank statements
* Track spending behavior
* Detect unusual transactions
* Identify scam-like finance messages
* Analyze suspicious financial offers

The system combines:

* Financial transaction analysis
* Retrieval-grounded AI responses
* Scam detection logic
* Local Gemma reasoning through Ollama

The assistant only answers using uploaded statement evidence, making responses explainable and grounded.

---

# Features

## 📊 Bank Statement Dashboard

Upload CSV, XLS, or XLSX bank statements and automatically:

* Detect transaction columns
* Normalize different statement formats
* Categorize spending
* Generate financial dashboards
* Analyze recurring expenses
* Detect unusual spending patterns

Supported categories include:

* Food
* Transport
* Groceries
* Shopping
* Bills
* Rent
* Subscriptions
* Healthcare
* Travel
* Transfers
* Income

---

## 💬 Ask Guardian Chat

Users can ask questions such as:

* “Where did I spend the most?”
* “How much did I spend in April?”
* “What are my recurring subscriptions?”
* “Which transactions look unusual?”
* “How can I save more money?”

The assistant retrieves relevant statement rows before generating responses.

---

## 🚨 Message Legitimacy Check

Ask Guardian can analyze:

* SMS messages
* WhatsApp messages
* Emails
* Telegram messages
* Investment offers
* Suspicious financial promotions

The system checks for warning signs such as:

* OTP requests
* Password or CVV requests
* Fake urgency
* Suspicious links
* Guaranteed returns
* Crypto payment requests
* Scam-like language

---

## 🏦 Finance Service Risk Checker

Users can paste:

* Finance service descriptions
* Investment advertisements
* Loan offers
* Website URLs

The assistant checks for:

* Unrealistic investment claims
* Fake financial language
* Suspicious payment requests
* Scam-related wording
* Risk indicators

---

# How It Works

## Statement Processing

The app:

1. Reads uploaded statements
2. Detects important columns automatically
3. Cleans malformed values
4. Parses dates and amounts
5. Categorizes transactions
6. Stores searchable transaction memory in ChromaDB

---

## Retrieval-Grounded Responses

Each transaction becomes a searchable document containing:

* Date
* Merchant
* Description
* Category
* Amount
* Month
* Transaction type

Relevant rows are retrieved before asking Gemma to answer.

This keeps explanations grounded in uploaded evidence.

---

# Gemma Integration

Ask Guardian uses Gemma locally through Ollama.

Gemma is responsible for:

* Explaining spending behavior
* Summarizing statement trends
* Interpreting scam signals
* Generating grounded financial explanations

If Gemma becomes unavailable, the system falls back to deterministic rule-based responses to maintain reliability.

---

# Tech Stack

| Component           | Technology             |
| ------------------- | ---------------------- |
| Frontend            | Streamlit              |
| Data Processing     | Pandas                 |
| Vector Database     | ChromaDB               |
| Embeddings          | SentenceTransformers   |
| Fallback Embeddings | Custom Hash Embeddings |
| Charts              | Plotly                 |
| LLM                 | Gemma via Ollama       |
| Image Handling      | Pillow                 |

---

# Project Architecture

```text
User Uploads Statement
        ↓
Statement Cleaning & Normalization
        ↓
Transaction Categorization
        ↓
ChromaDB Indexing
        ↓
Relevant Transaction Retrieval
        ↓
Gemma Prompting
        ↓
Grounded Financial Answer
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ask-guardian.git
cd ask-guardian
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Download Ollama:

https://ollama.com

---

## 5. Pull Gemma Model

```bash
ollama pull gemma3:4b
```

---

## 6. Run the App

```bash
streamlit run app.py
```

---

# Environment Variables

Optional configuration:

```bash
export GEMMA_PROVIDER=ollama
export GEMMA_MODEL=gemma3:4b
export OLLAMA_URL=http://localhost:11434/api/generate
export STATEMENT_CURRENCY=SGD
export STATEMENT_CURRENCY_SYMBOL=S$
```

---

# Example Questions

## Financial Questions

* “What category do I spend the most on?”
* “Show my recurring expenses.”
* “Which month had the highest spending?”
* “Explain this transaction.”

## Safety Questions

* “Does this SMS look suspicious?”
* “Is this investment offer risky?”
* “Does this finance service look legitimate?”

---

# Key Safety Design Choices

## Grounded Responses

The assistant only answers using uploaded statement evidence.

## Conservative Scam Analysis

The app never guarantees legitimacy.

Instead, it:

* Flags suspicious behavior
* Encourages independent verification
* Advises against sharing sensitive information

## Local AI

Gemma runs locally through Ollama for:

* Privacy
* Offline usage
* Safer financial analysis

---

# Challenges Solved

## Messy Bank Statements

Different banks use different formats.

The app handles:

* Multiple encodings
* Different separators
* Debit/credit variations
* Missing column names
* Malformed dates

---

## Trustworthy AI Responses

Instead of allowing hallucinated answers:

* Relevant rows are retrieved first
* Prompting is tightly constrained
* Responses remain evidence-based

---

# Impact

Ask Guardian helps users:

* Understand their finances
* Detect unusual transactions
* Build financial awareness
* Identify scam risks
* Improve spending habits

This is especially useful for:

* Students
* Elderly users
* Young professionals
* First-time banking users
* Scam-vulnerable users

---

# Hackathon Alignment

## Safety & Trust

Ask Guardian focuses on:

* Explainable AI
* Financial safety
* Scam awareness
* Grounded reasoning

## Ollama Track

The project uses:

* Local Gemma inference
* Ollama integration
* Offline AI workflows

---

# Conclusion

Ask Guardian demonstrates how local AI can improve financial safety and trust.

By combining:

* Transaction intelligence
* Retrieval-grounded reasoning
* Scam signal detection
* Explainable AI responses

the project turns Gemma into a practical financial safety companion for everyday users.
