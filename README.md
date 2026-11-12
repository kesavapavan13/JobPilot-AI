# JobPilot AI

A multi-agent job application assistant built with LangGraph, Google Gemini, Pydantic, Streamlit and MySQL.

Give it a job description and your resume. It scores the fit, rewrites your resume bullets, drafts a cover letter, and a reviewer agent checks every claim against your original resume before the result is saved.

## Why this project

Generic AI resume tools often invent skills, tools, and numbers you never had. JobPilot AI adds a separate Reviewer agent that fact-checks the output. If it finds an unsupported claim, the draft goes back to the Tailor with the exact claims to remove (up to 3 attempts).

## Features

- Match score (0-100) with matched skills and skills gap
- 6-8 tailored resume bullets using the job's keywords, only where truthful
- Cover letter under 300 words
- Self-correcting fact-check loop between the Tailor and Reviewer agents
- MySQL application tracker with status (Draft, Applied, Interview, Offer, Rejected)
- Duplicate detection for the same job description and resume
- PDF upload or pasted text for the resume
- Retries with backoff on LLM errors, and input validation before any tokens are spent
- If MySQL is down, generated content is still shown
- Evaluation script to measure the pipeline on real job postings

## How it works

```
JD + Resume -> Analyzer -> Matcher -> Tailor -> Reviewer --approved--> Tracker (MySQL) -> END
                                         ^          |
                                         +-rejected-+   (max 3 attempts)
```

| Agent | Job | Output |
|---|---|---|
| Analyzer | Extracts role, company, required skills, keywords | `JDInfo` |
| Matcher | Scores fit and lists matched and missing skills | `Match` |
| Tailor | Rewrites bullets and drafts a cover letter from resume facts only | `Tailored` |
| Reviewer | Fact-checks the new content against the original resume | `Review` |
| Tracker | Saves the application to MySQL | DB row |

If the Reviewer still rejects after 3 attempts, the result is saved as not approved and the UI warns you to fix it before sending.

## Tech stack

- LangGraph for orchestration
- Google Gemini via `langchain-google-genai`
- Pydantic for structured outputs
- Streamlit for the UI
- MySQL via `mysql-connector-python`
- pypdf for PDF text extraction
- pytest for tests

## Project structure

```
jobpilot/
├── agent.py          # LangGraph pipeline
├── app.py            # Streamlit front end
├── db.py             # MySQL tracker
├── evaluate.py       # Batch evaluation
├── requirements.txt
├── .streamlit/config.toml
├── eval_data/jds/    # Job descriptions for evaluation (.txt)
└── tests/
    ├── test_agent.py
    └── test_db.py
```

## Setup

Requirements: Python 3.11+, a running MySQL server, and a Gemini API key from Google AI Studio.

```bash
git clone https://github.com/<your-username>/jobpilot.git
cd jobpilot

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=jobpilot

# Optional
# MODEL=gemini-3.5-flash-lite
```

The database and table are created automatically on first run. Do not commit `.env`; keep it in `.gitignore`.

## Run

```bash
streamlit run app.py
```

Open the New application tab, paste a job description, add your resume (PDF or text), and click Tailor my application. Use the History tab to reopen, update or delete past applications. Scanned PDFs have no extractable text, so paste those instead.

## Evaluate

Put 10-15 job descriptions as `.txt` files in `eval_data/jds/` and your resume at `eval_data/resume.txt`, then run:

```bash
python evaluate.py --resume eval_data/resume.txt
```

It reports the share of outputs clean on the first attempt and after the review loop, the average attempts, and the average match score. Details are saved to `eval_results.csv`.

The Reviewer is an LLM, so "clean" means "passed the reviewer". Spot-check a few outputs by hand before quoting any number.

## Test

```bash
pytest
```

Tests use a fake LLM, so they run offline and use no API quota.

## Limitations

- Output quality depends on the Gemini model and your source resume.
- The Reviewer can miss subtle fabrications, so always read the final content.
- Inputs are limited to 20,000 characters each.
- No OCR for scanned PDFs.


## 🧑‍💻 Author

**Kesava Pavan Gadde**

- 📧 [kesavapavangadde@gmail.com](mailto:kesavapavangadde@gmail.com)
- 💼 [LinkedIn]([https://linkedin.com/in/<your-linkedin-id>](https://www.linkedin.com/in/kesava-pavan-gadde/))
