# Ask Guardian

Ask Guardian is a Streamlit demo app that reads bank statements, normalizes transactions, and gives clear financial insights from the uploaded data.

It supports CSV, XLS, and XLSX files, including multiple monthly statements uploaded together.

## Features

- Multi-file bank statement upload
- CSV/XLS/XLSX parsing with robust date and amount handling
- Automatic transaction normalization
- Category detection for food, transport, groceries, shopping, bills, rent, subscriptions, income, transfers, healthcare, and travel
- Multi-month dashboard with month filter
- Guardian Intelligence Report
- Financial health score out of 100
- Unusual spending detection
- Behavioral spending insights
- Month-aware chatbot
- Transaction table with row selection
- Explain-this-transaction panel
- Message scam checker
- Finance service risk checker
- Optional Gemma/Ollama or Hugging Face model support

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Uploading Statements

In the sidebar, upload one or more bank statements:

- `.csv`
- `.xls`
- `.xlsx`

You can upload several monthly statements together. Ask Guardian combines them into one transaction history and removes obvious duplicate rows.

Expected columns can include names like:

- `date`
- `transaction date`
- `posting date`
- `value date`
- `description`
- `transaction description`
- `merchant`
- `amount`
- `debit amount`
- `credit amount`
- `withdrawal amount`
- `deposit amount`
- `balance`

The parser is designed to handle mixed bank formats, malformed date-like values, and separate debit/credit columns.

## Dashboard

The Dashboard shows:

- transaction count
- total spend
- income
- top category
- spending by category
- monthly spending chart
- all transactions table
- month-wise filtering
- transaction chooser
- explanation for the selected transaction

Use the **Statement month** selector to switch between:

- `All months`
- each detected month, such as `2026-04`, `2026-05`

## Guardian Intelligence Report

After upload, the Dashboard shows a **Guardian Intelligence Report**.

It includes:

- Guardian Score out of 100
- rating label, such as `Strong`, `Good`, `Watch`, or `Needs attention`
- positive factors
- key risk factors
- unusual transactions table
- behavioral spending insights
- recommended actions

The score is explainable and rule-based. It considers:

- savings rate
- spending compared with income
- flexible spending categories
- recurring expenses
- unusual high-value transactions

## Ask Guardian Chat

The chat answers from the uploaded statement data.

Example questions:

```text
where do i spend money most
what is my total spend
what income did i get
what should i do to save money
monthly breakdown
which month had highest spending
show transactions in May
list April statement
how much did I spend in April
what subscriptions repeat monthly
```

Month-aware examples:

```text
where do i spend most in April
what is my total spend in May
what should i do to save money in April
show transactions in latest month
what did I spend last month
```

Supported month phrases include:

- full month names, such as `April`
- short month names, such as `Apr`
- `YYYY-MM`
- `YYYY/MM`
- `last month`
- `previous month`
- `latest month`
- `most recent month`

## Teaching Ask Guardian

You can teach small corrections in chat using `remember`.

Examples:

```text
remember apartment rent is rent
remember exclude side project income from normal salary
remember PayNow to Tan Wei is shared rent, not spending
```

Use **Clear learned memory** in the sidebar to reset it.

## Transaction Explanation

In the Dashboard:

1. Open the transaction table.
2. Tick one row in the **Choose** column.
3. Read the **Explain this transaction** section.

The explanation includes:

- transaction type
- amount
- date
- category
- statement description
- share of total spending
- share of monthly spending
- share of category spending
- rank among expenses

## Scam And Finance Checks

Ask Guardian includes two safety tools:

- **Message Check**
- **Finance Service Check**

These tools check for signals like:

- OTP or banking credential requests
- urgent threats
- suspicious links
- payment pressure
- unrealistic investment returns
- missing identity or licensing details

These checks are risk assessments, not proof that a message or service is legitimate. Always verify finance services using the relevant official regulator register.

## Model Configuration

Ask Guardian works even if a model is unavailable because many answers are calculated directly from the dataframe.

Optional Ollama setup:

```bash
export GEMMA_PROVIDER=ollama
export GEMMA_MODEL=gemma3:4b
```

Optional Hugging Face Transformers setup:

```bash
export GEMMA_PROVIDER=transformers
export GEMMA_MODEL=google/gemma-3-4b-it
```

## Currency Configuration

Default display:

```text
SGD / S$
```

Override it with:

```bash
export STATEMENT_CURRENCY=USD
export STATEMENT_CURRENCY_SYMBOL=$
```

## Project Files

```text
app.py                         Main Streamlit app
requirements.txt               Python dependencies
README.md                      Project guide
sample_singapore_bank_statement.csv
guardian_chroma/               Local ChromaDB persistence
```

## Demo Flow

For a hackathon demo:

1. Start the app.
2. Upload multiple monthly statements.
3. Show the Dashboard month selector.
4. Show the Guardian Intelligence Report.
5. Select a transaction and explain it.
6. Ask: `which month had highest spending`
7. Ask: `where do i spend most in April`
8. Ask: `what should i do to save money`
9. Teach: `remember apartment rent is rent`
10. Ask the same question again to show personalization.

## Notes

- The app does not provide financial advice.
- It gives data-driven insights from uploaded statements.
- It may categorize some merchants imperfectly.
- You can improve categories by teaching corrections in chat.
- Always review uploaded data and important financial decisions manually.
# gemma-4-Ask_guardian
