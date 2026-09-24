import os
import sys
import io
import json
import csv
from typing import Dict, Any, Optional

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

class DocumentReader:
    """
    Robust reader for arbitrary document formats (PDF, DOCX, TXT, MD, CSV, JSON).
    Extracts structured text with page count and document metadata.
    """
    def extract_text_from_file(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            return {
                "text": "",
                "format": "unknown",
                "page_count": 0,
                "metadata": {},
                "success": False,
                "error": f"File not found: {file_path}"
            }

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        with open(file_path, "rb") as f:
            content_bytes = f.read()

        return self.extract_text_from_bytes(content_bytes, filename=os.path.basename(file_path), file_format=ext)

    def extract_text_from_bytes(self, content: bytes, filename: str = "", file_format: str = "") -> Dict[str, Any]:
        fmt = (file_format or os.path.splitext(filename)[1]).lower().lstrip(".")
        
        # 1. PDF Documents
        if fmt == "pdf":
            return self._extract_pdf(content)

        # 2. Microsoft Word Documents
        elif fmt in ("docx", "doc"):
            return self._extract_docx(content)

        # 3. CSV Tabular Documents
        elif fmt == "csv":
            return self._extract_csv(content)

        # 4. JSON Structured Files
        elif fmt == "json":
            return self._extract_json(content)

        # 5. Plain Text, Markdown, Logs, Code
        else:
            return self._extract_text_plain(content, fmt)

    def _extract_pdf(self, content: bytes) -> Dict[str, Any]:
        if pypdf is None:
            return {"text": "", "format": "pdf", "page_count": 0, "metadata": {}, "success": False, "error": "pypdf library not available"}
        
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages_text = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages_text.append(f"--- Page {idx + 1} ---\n{txt.strip()}")

            full_text = "\n\n".join(pages_text).strip()
            meta = {}
            if reader.metadata:
                for k, v in reader.metadata.items():
                    clean_k = str(k).lstrip("/")
                    meta[clean_k] = str(v)

            return {
                "text": full_text,
                "format": "pdf",
                "page_count": len(reader.pages),
                "metadata": meta,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {"text": "", "format": "pdf", "page_count": 0, "metadata": {}, "success": False, "error": str(e)}

    def _extract_docx(self, content: bytes) -> Dict[str, Any]:
        if docx is None:
            return {"text": "", "format": "docx", "page_count": 1, "metadata": {}, "success": False, "error": "python-docx library not available"}

        try:
            doc = docx.Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)

            full_text = "\n".join(paragraphs).strip()
            return {
                "text": full_text,
                "format": "docx",
                "page_count": 1,
                "metadata": {"paragraph_count": len(paragraphs)},
                "success": True,
                "error": None
            }
        except Exception as e:
            return {"text": "", "format": "docx", "page_count": 0, "metadata": {}, "success": False, "error": str(e)}

    def _extract_csv(self, content: bytes) -> Dict[str, Any]:
        try:
            text_str = self._decode_bytes(content)
            reader = csv.reader(io.StringIO(text_str))
            rows = []
            for r in reader:
                line = " | ".join(c.strip() for c in r if c.strip())
                if line:
                    rows.append(line)
            full_text = "\n".join(rows[:500])  # Cap at 500 rows for summary
            return {
                "text": full_text,
                "format": "csv",
                "page_count": 1,
                "metadata": {"row_count": len(rows)},
                "success": True,
                "error": None
            }
        except Exception as e:
            return {"text": "", "format": "csv", "page_count": 0, "metadata": {}, "success": False, "error": str(e)}

    def _extract_json(self, content: bytes) -> Dict[str, Any]:
        try:
            text_str = self._decode_bytes(content)
            data = json.loads(text_str)
            flat_strings = []
            
            def flatten(obj):
                if isinstance(obj, str):
                    if len(obj.strip()) > 3: flat_strings.append(obj.strip())
                elif isinstance(obj, dict):
                    for v in obj.values(): flatten(v)
                elif isinstance(obj, list):
                    for v in obj: flatten(v)

            flatten(data)
            full_text = "\n".join(flat_strings[:500])
            return {
                "text": full_text,
                "format": "json",
                "page_count": 1,
                "metadata": {"items_extracted": len(flat_strings)},
                "success": True,
                "error": None
            }
        except Exception as e:
            return {"text": "", "format": "json", "page_count": 0, "metadata": {}, "success": False, "error": str(e)}

    def _extract_text_plain(self, content: bytes, fmt: str) -> Dict[str, Any]:
        try:
            full_text = self._decode_bytes(content).strip()
            return {
                "text": full_text,
                "format": fmt or "txt",
                "page_count": 1,
                "metadata": {"char_count": len(full_text)},
                "success": True,
                "error": None
            }
        except Exception as e:
            return {"text": "", "format": fmt or "txt", "page_count": 0, "metadata": {}, "success": False, "error": str(e)}

    def _decode_bytes(self, content: bytes) -> str:
        for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
            try:
                return content.decode(enc)
            except Exception:
                continue
        return content.decode("utf-8", errors="ignore")
