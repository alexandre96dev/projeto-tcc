# -*- coding: utf-8 -*-
"""
Tools para o pipeline:
- ExtrairTextoPDFTool: extrai texto inteiro do PDF.
- SegmentarPITPorSecoesTool: cria arquivos-âncora por seção e um JSON com itens literais.
- VerificarTrechoTool: valida se um bullet proposto está contido no texto âncora.
- LerArquivoTextoTool: lê arquivos .txt/.md/.json como string (para os agentes).
- LerJSONTool: lê e retorna JSON já parseado (dict em string para o LLM).
"""

from crewai.tools import BaseTool
from typing import Optional, Dict, List, Any
import fitz  # PyMuPDF
import os
import re
import json

# ---------------- utils internas ----------------

SECOES = [
    ("ensino", r"Atividades\s+de\s+Ensino"),
    ("pesquisa", r"Atividades\s+de\s+Pesquisa"),
    ("extensao", r"Atividades\s+de\s+Extens[aã]o"),
    ("admin", r"Atividades\s+Administrativ[oa]-Pedag[oó]gicas"),
    ("obs", r"(?:Complemento/Observa[cç][oõ]es|Observa[cç][oõ]es)"),
]

TITULOS = {
    "ensino": "Atividades de Ensino",
    "pesquisa": "Atividades de Pesquisa",
    "extensao": "Atividades de Extensão",
    "admin": "Atividades Administrativo-Pedagógicas",
    "obs": "Complemento/Observações",
}

def _extrair_texto(pdf_path: str) -> str:
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def _segmentar_por_secoes(texto: str) -> Dict[str, List[str]]:
    secoes: Dict[str, List[str]] = {k: [] for k, _ in SECOES}
    for i, (key, header_regex) in enumerate(SECOES):
        start_match = re.search(header_regex, texto, re.IGNORECASE)
        if not start_match:
            continue
        start = start_match.end()
        end = None
        for _, next_header in SECOES[i + 1:]:
            m = re.search(next_header, texto, re.IGNORECASE)
            if m and (end is None or m.start() < end):
                end = m.start()
        bloco = texto[start:end].strip() if end else texto[start:].strip()
        linhas = [ln.strip() for ln in bloco.splitlines()]
        itens = []
        for ln in linhas:
            if not ln:
                continue
            ln = ln.lstrip("-•\t ").rstrip()
            if re.fullmatch(r"[-–—]*", ln):
                continue
            if len(ln) < 2:
                continue
            itens.append(ln)
        secoes[key] = itens
    return secoes

# ---------------- Tools ----------------

class ExtrairTextoPDFTool(BaseTool):
    name: str = "ExtrairTextoPDFTool"
    description: Optional[str] = "Extrai texto completo de um PDF. Uso: passe o caminho do PDF. Retorna o texto."

    def _run(self, pdf_path: str) -> str:
        return _extrair_texto(pdf_path)


class SegmentarPITPorSecoesTool(BaseTool):
    name: str = "SegmentarPITPorSecoesTool"
    description: Optional[str] = (
        "Segmenta './Planejamento/PIT.pdf' em seções (ensino/pesquisa/extensao/admin/obs), "
        "gera arquivos-âncora em `Relatorio_Final/PIT_sec_<sec>.txt` e `Relatorio_Final/PIT_segmentado.json` "
        "com as listas de itens literais. Retorna o caminho do JSON."
    )

    def _run(self, pdf_path: str = "./Planejamento/PIT.pdf") -> str:
        texto = _extrair_texto(pdf_path)
        os.makedirs("Relatorio_Final", exist_ok=True)

        with open("Relatorio_Final/PIT_raw.txt", "w", encoding="utf-8") as f:
            f.write(texto)

        secoes = _segmentar_por_secoes(texto)

        payload = {"fonte": pdf_path, "secoes": {}}
        for k, itens in secoes.items():
            anchor_path = f"Relatorio_Final/PIT_sec_{k}.txt"
            with open(anchor_path, "w", encoding="utf-8") as f:
                titulo = TITULOS.get(k, k)
                f.write(titulo + "\n\n")
                for it in itens:
                    f.write(it + "\n")
            payload["secoes"][k] = {
                "titulo": TITULOS.get(k, k),
                "anchor_path": anchor_path,
                "itens": itens,
            }

        json_path = "Relatorio_Final/PIT_segmentado.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return json_path


class VerificarTrechoTool(BaseTool):
    name: str = "VerificarTrechoTool"
    description: Optional[str] = (
        "Valida se um trecho está literalmente contido no arquivo de âncora (texto plano). "
        "Uso: dois argumentos: caminho_do_arquivo_ancora, trecho. Retorna 'OK' ou 'NAO_ENCONTRADO'."
    )

    def _run(self, anchor_path: str, trecho: str) -> str:
        if not os.path.exists(anchor_path):
            return "NAO_ENCONTRADO"
        with open(anchor_path, "r", encoding="utf-8") as f:
            base = f.read().lower()
        needle = trecho.strip().lower()
        return "OK" if needle and needle in base else "NAO_ENCONTRADO"


class LerArquivoTextoTool(BaseTool):
    name: str = "LerArquivoTextoTool"
    description: Optional[str] = (
        "Lê um arquivo de texto (.txt/.md/.json) e retorna o conteúdo como string. "
        "Use esta ferramenta para que o agente leia arquivos do projeto."
    )

    def _run(self, file_path: str) -> str:
        print(f"🔍 DEBUG LerArquivoTextoTool: Tentando ler arquivo: {file_path}")
        
        # Normalizar o caminho para funcionar com diferentes formatos
        normalized_path = file_path.replace("./", "")
        
        if not os.path.exists(normalized_path):
            print(f"❌ DEBUG: Arquivo não encontrado: {normalized_path}")
            print(f"📁 DEBUG: Diretório atual: {os.getcwd()}")
            print("📂 DEBUG: Arquivos no diretório pai do caminho solicitado:")
            try:
                parent_dir = os.path.dirname(normalized_path)
                if parent_dir and os.path.exists(parent_dir):
                    files = os.listdir(parent_dir)
                    print(f"   Arquivos encontrados: {files}")
                else:
                    print(f"   Diretório pai não existe: {parent_dir}")
            except Exception as e:
                print(f"   Erro ao listar diretório: {e}")
            return f"[ERRO] Arquivo não encontrado: {normalized_path}"
        
        try:
            with open(normalized_path, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"✅ DEBUG: Arquivo lido com sucesso. Tamanho: {len(content)} caracteres")
                if content.strip():
                    print(f"📄 DEBUG: Primeiros 100 caracteres: {content[:100]}...")
                else:
                    print("📄 DEBUG: Arquivo está vazio")
                return content
        except Exception as e:
            print(f"❌ DEBUG: Erro ao ler arquivo: {str(e)}")
            return f"[ERRO] Falha ao ler arquivo: {str(e)}"


class LerJSONTool(BaseTool):
    name: str = "LerJSONTool"
    description: Optional[str] = (
        "Lê um arquivo JSON e retorna seu conteúdo em texto (string JSON). "
        "Útil para o agente carregar `Relatorio_Final/PIT_segmentado.json`."
    )

    def _run(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return "{}"
        with open(file_path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
        # devolve serializado para o LLM consumir
        return json.dumps(data, ensure_ascii=False)
