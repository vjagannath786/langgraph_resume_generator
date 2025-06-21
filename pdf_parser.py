from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text

key = "AIzaSyDeXPq3SbgsUOCNfGcPMNiDjM2ejUN82fc"

if __name__ == "__main__":
    pdf_path = "M.pdf"  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)
    print("Extracted Text:")
    print(extracted_text)