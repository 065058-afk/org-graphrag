from docling.document_converter import DocumentConverter
import os

DEPARTMENTS = ["finance", "hr", "esg", "risk", "tech"]

def parse_all():
    converter = DocumentConverter()
    os.makedirs("data/parsed", exist_ok=True)
    for dept in DEPARTMENTS:
        raw_path = f"data/raw/{dept}.pdf"
        if not os.path.exists(raw_path):
            print(f"Skipping {dept} — file not found")
            continue
        result = converter.convert(raw_path)
        text = result.document.export_to_markdown()
        with open(f"data/parsed/{dept}.md", "w") as f:
            f.write(text)
        print(f"Parsed {dept}: {len(text)} chars")

if __name__ == "__main__":
    parse_all()
