# Security & Privacy Documentation — FraudGraph Intelligence (FGI)

## 1. Data Governance & Sensitive Data Handling

Financial transaction datasets often contain sensitive cardholder data, banking coordinates, and personal identifiable information (PII). FraudGraph Intelligence enforces the following security standards:

1. **No Live PII:** Only anonymized or synthetic transaction datasets (e.g. IEEE-CIS Fraud Detection public benchmark) are permitted within the workspace. Real customer names, full credit card Primary Account Numbers (PANs), CVVs, and plain-text passwords must never be stored.
2. **Strict Anonymization:** Relational identifiers such as `card1` through `card6` in the IEEE-CIS benchmark are masked categorical/integer codes that cannot be reverse-engineered to identify real-world accounts without private decryption keys held exclusively by the payment processor.
3. **Target Leakage Prohibition:** The target label (`isFraud`) must be strictly segregated from all graph construction logic and predictive feature inputs. Models and graph embeddings must only access data chronologically available at prediction time.

---

## 2. API Security & Input Sanitization

1. **Localhost Binding:** In development mode, the FastAPI backend binds exclusively to loopback addresses (`127.0.0.1`) to prevent unauthorized network exposure.
2. **CORS Governance:** Cross-Origin Resource Sharing (CORS) is restricted strictly to local frontend origins (`http://localhost:5173`, `http://127.0.0.1:5173`).
3. **No Dynamic Execution:** The API never exposes endpoints that execute arbitrary shell commands, raw SQL, or unchecked dynamic code.
4. **Input Boundary Validation:** All incoming REST payloads are parsed and validated through strongly typed Pydantic models with explicit upper bounds on node counts, edge indices, and matrix dimensions to prevent memory exhaustion (DoS).

---

## 3. Secret Management & Version Control Hygiene

1. **No Hardcoded Secrets:** API keys, database credentials, or cloud access tokens must never be hardcoded into source files.
2. **Ignored Environment Files:** All `.env` and `.env.local` files are registered in `.gitignore`. Only `.env.example` containing non-sensitive template keys is tracked in version control.
3. **Dataset Segregation:** Large raw archives (`data/raw/`), intermediate embeddings, and uncompressed transaction tables are excluded from Git to prevent inadvertent data exposure and bloated repository history.
