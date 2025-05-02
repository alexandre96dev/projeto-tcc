from crewai.tools import BaseTool
from typing import Optional
import fitz
import os
class VerificarPalavraNoPDFTool(BaseTool):
    name: str = "verificar_palavra_no_pdf"
    description: Optional[str] = "Extrai texto de um PDF e salva como .txt."

    def _run(self, pdf_path: str, destino_path: str) -> str:
        texto = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                texto += page.get_text()

        os.makedirs(os.path.dirname(destino_path), exist_ok=True)
        with open(destino_path, "w", encoding="utf-8") as f:
            f.write(texto)

        return f"Texto extraído de '{pdf_path}' e salvo em '{destino_path}'."
