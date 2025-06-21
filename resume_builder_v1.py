import os
import re
import json
import time
from typing import Optional, List, Dict, Annotated
from pdf_parser import extract_text_from_pdf
from jd_parser import get_linkedin_job_description
from langchain.chat_models import init_chat_model

# --- Third-party imports ---
# Ensure you have these installed:
# pip install langchain-google-genai langgraph python-dotenv beautifulsoup4 requests lxml PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dotenv import load_dotenv




load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
if not key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please create a .env file.")

os.environ["GOOGLE_API_KEY"] = key

#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = init_chat_model("google_genai:gemini-2.0-flash")

class TailoredResume(TypedDict):
    """Defines the structure for the tailored resume data."""
    name: Optional[str]
    contact: Optional[Dict[str, str]]
    summary: str
    experience: List[Dict[str, any]]
    skills: Dict[str, List[str]]
    projects: List[Dict[str, any]]
    education: List[Dict[str, any]]


class State(TypedDict):
    """Defines the state for the graph, tracking all data through the process."""
    jd_text: str
    resume_text: Optional[str]
    missing_info: List[str]
    tailored_resume_data: Optional[TailoredResume]
    latex_code: str
    url: str
    pdf_path: str
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def get_message_content(msg):
    """Helper to extract content from LangChain message objects."""
    if hasattr(msg, "content"):
        return msg.content
    return msg.get("content", "")

def chatbot(state: State):
    """Initial node to set up the graph state."""
    return state

def extract_jd_node(state: State):
    """Node to extract the job description from the provided URL."""
    print("--- Extracting Job Description ---")
    url = state["url"]
    jd_text = get_linkedin_job_description(url)
    state["jd_text"] = jd_text
    if not jd_text:
        state["missing_info"].append("Job Description could not be extracted from URL.")
        state["messages"].append({"role": "assistant", "content": "Error: Could not extract Job Description."})
    else:
        print("Job Description extracted successfully.")
        state["messages"].append({"role": "assistant", "content": "Extracted Job Description."})
    return state

def extract_resume_node(state: State):
    """Node to extract text from the provided resume PDF."""
    print("--- Extracting Resume Text ---")
    pdf_path = state["pdf_path"]
    if not pdf_path:
        state["resume_text"] = ""
        print("No resume PDF provided.")
        return state
    
    resume_text = extract_text_from_pdf(pdf_path)
    state["resume_text"] = resume_text
    if not resume_text:
        state["missing_info"].append("Resume text could not be extracted from PDF.")
        state["messages"].append({"role": "assistant", "content": "Error: Could not extract resume text."})
    else:
        print("Resume text extracted successfully.")
        state["messages"].append({"role": "assistant", "content": "Extracted Resume Text."})
    return state

def comparison_agent_node(state: State):
    """Node where the 'career coach' agent performs a gap analysis."""
    print("--- Analyzing Resume vs. Job Description ---")
    jd_text = state.get("jd_text", "")
    resume_text = state.get("resume_text", "")

    prompt = (
        "You are an expert career coach and resume reviewer. Your task is to perform a detailed gap analysis between the candidate's resume and the provided job description. "
        "Analyze the resume section by section and provide specific, actionable suggestions for improvement. Focus on highlighting missing keywords, skills, and ways to rephrase experience to better match the role's requirements.\n\n"
        "Your output should be a clear, concise list of suggestions, structured by resume section.\n\n"
        "--- Job Description ---\n"
        f"{jd_text}\n\n"
        "--- Current Resume ---\n"
        f"{resume_text}\n\n"
        "--- Gap Analysis & Suggestions for Improvement ---"
    )

    response = llm.invoke(prompt)
    suggestions = get_message_content(response)
    print("Analysis complete.")
    
    state["missing_info"] = [suggestions]
    state["messages"].append({"role": "assistant", "content": f"Suggestions for improvement:\n{suggestions}"})
    return state

def tailored_resume_agent_node(state: State):
    """Node where the 'resume writer' agent enhances an existing resume."""
    print("--- Generating Tailored Resume Content (from existing resume) ---")
    jd_text = state.get("jd_text", "")
    resume_text = state.get("resume_text", "")
    suggestions_text = "\n".join(state.get("missing_info", []))

    prompt = (
        "You are an expert resume writer and career strategist. Transform the 'Current Resume' into a 'Tailored Resume' using the 'Job Description' and 'Suggestions'.\n\n"
        "**Guiding Principles:**\n"
        "1.  **Integrate Keywords:** Seamlessly weave in keywords from the JD (e.g., 'Gen AI', 'MLOps', 'subscriber growth').\n"
        "2.  **Quantify Achievements:** Rephrase experience to be achievement-oriented (e.g., 'Increased X by Y%', 'Reduced costs by Z%').\n"
        "3.  **Align with Responsibilities:** Ensure experience directly reflects the key 'Roles & Responsibilities'.\n"
        "4.  **No Fabrication:** Enhance and rephrase existing experience. Do not invent new roles or jobs.\n\n"
        "**Input:**\n"
        "Job Description:\n"
        f"{jd_text}\n\n"
        "Current Resume:\n"
        f"{resume_text}\n\n"
        "Suggestions:\n"
        f"{suggestions_text}\n\n"
        "**Output:**\n"
        "Return a single, clean JSON object. This object MUST include fields like `name`, `contact` (a dict with email, phone, linkedin, etc.), `summary`, `experience` (a list of dicts, each with title, company, date, and a list of 'bullets'), `skills` (a dict of categories to lists of skills), `projects` (similar to experience), and `education`. Infer missing fields like name and contact if possible."
    )
    
    response = llm.invoke(prompt)
    raw = get_message_content(response)
    cleaned = re.sub(r"^```(?:json)?\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        tailored_resume = json.loads(cleaned)
        state["tailored_resume_data"] = tailored_resume
        print("Tailored resume JSON generated successfully.")
        state["messages"].append({"role": "assistant", "content": "Generated tailored resume data."})
    except Exception as e:
        print(f"Error parsing tailored resume JSON: {e}")
        state["messages"].append({"role": "assistant", "content": f"Error: Could not generate tailored resume data."})

    return state

def tailored_resume_groundup_agent_node(state: State):
    """Node where the 'resume creator' agent builds a resume from scratch for an ideal candidate."""
    print("--- Generating Tailored Resume Content (from scratch) ---")
    jd_text = state.get("jd_text", "")
    
    prompt = (
        "You are an expert resume creator and career strategist. Create a complete, professional resume for an **ideal candidate** applying for the given job description.\n\n"
        "**Process:**\n"
        "1.  **Analyze the JD:** Deduce the required years of experience, technical skills, and leadership qualifications.\n"
        "2.  **Create a Persona:** Build a logical career history for a fictional but realistic candidate (e.g., 'Alex Chen') who perfectly fits this role.\n"
        "3.  **Write Compelling Content:** Author a summary, skill list, and experience section with plausible, quantifiable achievements that directly align with the job's responsibilities.\n"
        "4.  **Mirror the Language:** Use the language and tone from the job description.\n\n"
        "**Input:**\n"
        "Job Description:\n"
        f"{jd_text}\n\n"
        "**Output:**\n"
        "Return a single, clean JSON object. This object MUST include fields like `name` ('Alex Chen' or similar), `contact` (a dict with placeholder email, phone, linkedin, github, location), `summary`, `experience` (a list of dicts, each with title, company, date, and a list of 'bullets'), `skills` (a dict of categories to lists of skills), `projects`, and `education`."
    )

    response = llm.invoke(prompt)
    raw = get_message_content(response)
    cleaned = re.sub(r"^```(?:json)?\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    
    try:
        tailored_resume = json.loads(cleaned)
        state["tailored_resume_data"] = tailored_resume
        print("Resume JSON generated from scratch successfully.")
        state["messages"].append({"role": "assistant", "content": "Generated resume data from scratch."})
    except Exception as e:
        print(f"Error parsing ground-up resume JSON: {e}")
        state["messages"].append({"role": "assistant", "content": f"Error: Could not generate resume data."})

    return state

def latex_code_generation_node(state: State):
    """Node where the 'LaTeX designer' agent generates the final .tex file."""
    print("--- Generating Professional LaTeX Code ---")
    tailored_resume = state.get("tailored_resume_data")
    if not tailored_resume:
        state["missing_info"].append("No tailored resume data available for LaTeX generation.")
        print("Error: No resume data to generate LaTeX.")
        return state

    prompt = (
        r"You are an expert LaTeX resume designer. Your task is to generate a visually polished, modern, and professional single-page resume using the provided JSON data. The final output must be a fully compilable LaTeX document for the **pdfLaTeX** compiler."
        r"\n\n**Design Requirements:**"
        r"\n1.  **Preamble & Packages:** Start with `\documentclass[letterpaper,11pt]{article}`. You MUST use the following packages: `\usepackage[T1]{fontenc}`, `\usepackage{lato}`, `\usepackage[usenames,dvipsnames]{xcolor}`, `\usepackage{titlesec}`, `\usepackage{enumitem}`, `\usepackage[hidelinks]{hyperref}`, `\usepackage{fontawesome5}`, and `\usepackage{tabularx}`."
        r"\n2.  **Layout & Font:** Set `lato` as the default sans-serif font for the entire document using `\renewcommand{\familydefault}{\sfdefault}`. Use balanced margins (e.g., `\addtolength{\oddsidemargin}{-0.6in}`). Define a professional color named `primary`."
        r"\n3.  **Header:** Create a centered header. The name should be `\Huge \scshape \bfseries`. Below the name, use a `tabularx` environment to neatly align contact info into columns with `fontawesome5` icons (e.g., \faIcon{map-marker-alt}, \faIcon{phone})."
        r"\n4.  **Section Formatting:** Use `\titleformat` to create clean, bold section titles with a colored rule underneath."
        r"\n5.  **Subheading Command:** You MUST create and use a custom command `\newcommand{\resumeSubheading}[4]{...}` for all entries in Experience, Projects, and Education. This command must place the title/company on the left and the date on the right, all on a single line, using a `tabular*` environment for perfect alignment."
        r"\n6.  **Compact Lists:** For all bullet points under Experience and Projects, use an `itemize` environment with the `nosep` and `leftmargin=*` options from the `enumitem` package to ensure tight spacing."
        r"\n7.  **Skills Section:** Format the skills using a `tabularx` environment. The first column should contain the bolded skill category (e.g., **Languages**), and the second column should list the corresponding skills."
        r"\n8.  **Final Output:** The response must be **only the complete, raw LaTeX code**. Do not include ```latex, markdown, or any explanations. It must be ready to copy and compile directly."
        r"\n\n**Resume Data (JSON):**\n"
        f"{json.dumps(tailored_resume, indent=2)}\n\n"
        r"**Full LaTeX Code:**"
    )

    response = llm.invoke(prompt)
    latex_code = get_message_content(response)
    cleaned_latex = re.sub(r"^```(?:latex)?\s*|```$", "", latex_code.strip(), flags=re.MULTILINE).strip()

    state["latex_code"] = cleaned_latex
    
    output_filename = "tailored_resume.tex"
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(cleaned_latex)
    print(f"LaTeX code generated and saved to {output_filename}")
    state["messages"].append({"role": "assistant", "content": f"LaTeX resume generated and saved to {output_filename}."})
    return state

def resume_exists_condition(state: State):
    """Conditional edge to decide the path based on resume existence."""
    if state.get("resume_text"):
        print("Condition: Resume exists. Proceeding to comparison.")
        return "comparison_agent"
    else:
        print("Condition: No resume. Proceeding to generate from scratch.")
        return "tailored_resume_groundup_agent"

# --- Graph Definition ---
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("extract_jd", extract_jd_node)
graph_builder.add_node("extract_resume", extract_resume_node)
graph_builder.add_node("comparison_agent", comparison_agent_node)
graph_builder.add_node("tailored_resume_agent", tailored_resume_agent_node)
graph_builder.add_node("tailored_resume_groundup_agent", tailored_resume_groundup_agent_node)
graph_builder.add_node("latex_code_generation", latex_code_generation_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "extract_jd")
graph_builder.add_edge("extract_jd", "extract_resume")
graph_builder.add_conditional_edges(
    "extract_resume",
    resume_exists_condition,
    {
        "comparison_agent": "comparison_agent",
        "tailored_resume_groundup_agent": "tailored_resume_groundup_agent"
    }
)
graph_builder.add_edge("comparison_agent", "tailored_resume_agent")
graph_builder.add_edge("tailored_resume_agent", "latex_code_generation")
graph_builder.add_edge("tailored_resume_groundup_agent", "latex_code_generation")
graph_builder.add_edge("latex_code_generation", "__end__")

graph = graph_builder.compile()


if __name__ == "__main__":
    print("--- Welcome to the AI Resume Tailoring Assistant ---")
    url = input("Please enter the LinkedIn Job Description URL: ")
    pdf_path = input("Please enter the path to your resume PDF (or press Enter if not available): ")

    initial_state = {
        "url": url,
        "pdf_path": pdf_path.strip(),
        "messages": [
            {"role": "user", "content": f"JD URL: {url}"},
            {"role": "user", "content": f"Resume Path: {pdf_path}"},
        ],
        "missing_info": [] # Initialize missing_info
    }

    try:
        # Stream the execution to see progress
        for event in graph.stream(initial_state):
            # The event keys are the node names
            for node_name, state_update in event.items():
                print(f"\n--- Completed Node: {node_name} ---")
                # You can inspect state_update here if needed
                
        # Final state can be accessed after the stream is exhausted
        # but for simplicity, we'll just print a final message.
        print("\n--- Process Complete ---")
        # A simple check to see if the final file was likely created
        if os.path.exists("tailored_resume.tex"):
             print("A tailored resume has been generated in 'tailored_resume.tex'.")
             print("You can now compile this file using a LaTeX editor (like Overleaf, TeXstudio, etc.).")
        else:
             print("Could not generate the final LaTeX file. Please check the logs for errors.")

    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred during the process: {e}")
        traceback.print_exc()