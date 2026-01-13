# PDF RAG Project

This project implements a **Retrieval-Augmented Generation (RAG)** system over PDF documents. It allows you to ingest PDFs, index their contents, and query them using a language model to get grounded, document-based answers.

---

## 🚀 Features

* 📄 PDF ingestion and text extraction
* 🔍 Vector-based retrieval
* 🤖 Language model inference (RAG)
* 🧠 Context-aware answers from your documents
* 🔐 Secrets and tokens kept out of Git

---

## 🗂 Project Structure

```text
pdf_rag/
│
├── backend/            # Backend logic (RAG pipeline, model calls)
├── models/             # Model directory (weights are ignored by git)
├── .gitignore          # Git ignore rules (models, secrets, env files)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

> ⚠️ Large model files and secrets (API tokens) are **not** committed to GitHub.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/chialiahmedanes-beep/pdf_rag.git
cd pdf_rag
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\\Scripts\\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file **locally** (this file is ignored by Git):

```env
HF_TOKEN=hf_your_huggingface_token_here
```

Load it in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
```

---

## ▶️ Running the Project

Example (adjust based on your actual entry point):

```bash
python backend/run_model.py
```

---

## 🛡 Security Notes

* ❌ Never commit `.env` files
* ❌ Never commit model weights
* ✅ Use `.gitignore`
* ✅ Use environment variables for secrets

GitHub Push Protection is enabled for this repository.

---

## 📌 Future Improvements

* Web UI for querying PDFs
* Support for multiple models
* Persistent vector database
* Docker support

---

## 📄 License

This project is for internal / educational use. Add a license if you plan to distribute it.

---

## 👤 Author

**ahmed anes chiali**
GitHub: [https://github.com/chialiahmedanes-beep](https://github.com/chialiahmedanes-beep)

---

If you have questions or want to extend this project, feel free to contribute 🚀
