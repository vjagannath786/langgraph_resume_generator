# 🤖 AI Resume Tailoring Assistant

This project is an end-to-end intelligent pipeline that takes a **LinkedIn job description** and a **candidate’s resume PDF**, then produces a **professionally tailored LaTeX resume** optimized for the given job. Built using [LangGraph](https://python.langchain.com/docs/langgraph/), [LangChain](https://www.langchain.com/), and [Gemini 2.0 Flash (via Google GenAI)](https://ai.google.dev/), it automates the resume refinement process using custom agents.

---

## 🧠 How It Works

The system is modeled as a LangGraph state machine with multiple expert agents:
- **Career Coach** (gap analysis)
- **Resume Writer** (tailoring)
- **Resume Architect** (ground-up creation)
- **LaTeX Designer** (professional formatting)

---

## 🔁 Flowchart

![Flowchart] 

---

## ⚙️ Architecture

### Modules

| Node | Description |
|------|-------------|
| `chatbot` | Initializes the conversation and captures user input |
| `extract_jd` | Scrapes and parses the job description from LinkedIn |
| `extract_resume` | Extracts raw text from the candidate’s PDF resume |
| `comparison_agent` | Analyzes gaps between resume and job description |
| `tailored_resume_agent` | Refines the original resume using LLM + suggestions |
| `tailored_resume_groundup_agent` | Builds an ideal resume from scratch if no resume is provided |
| `latex_code_generation` | Generates high-quality LaTeX from the structured resume data |

---

## 📥 Input Required

1. **Job URL** (LinkedIn or similar)
2. **Resume PDF Path** (optional — ground-up resume generated if missing)

---

## 📤 Output

- `tailored_resume.tex` — fully compile-ready LaTeX resume file
- Uses elegant design with:
  - `\usepackage{lato}`
  - `\usepackage{titlesec}`, `enumitem`, `fontawesome5`, `tabularx`
  - One-page modern layout with sections: Summary, Skills, Experience, Projects, Education

---

## 🧪 How to Run

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/your-org/ai-resume-tailor.git
cd ai-resume-tailor
pip install -r requirements.txt
