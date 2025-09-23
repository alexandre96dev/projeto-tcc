"""
Estudo de Caso 1 - Análise de Texto Científico
Sistema refatorado aplicando princípios SOLID e Clean Code
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, Type, Union
import json
import os
import re
from datetime import datetime

import fitz
import litellm
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from docx import Document



class ModelType(Enum):
    LLAMA_7B = "llama_7b"
    LLAMA_70B = "llama_70b"
    CHATGPT = "gpt"


class GravidadeProblema(Enum):
    BAIXA = "baixa"
    MEDIA = "média"
    ALTA = "alta"


@dataclass(frozen=True)
class ConfiguracaoModelo:
    model_name: str
    display_name: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 4096
    max_retries: int = 200


@dataclass
class ProblemaEncontrado:
    localizacao: str
    trecho_exato: str
    descricao: str
    sugestao: str
    gravidade: GravidadeProblema
    pagina: Optional[int] = None


@dataclass
class ResultadoAnalise:
    erros_gramaticais: List[ProblemaEncontrado]
    necessidades_citacao: List[ProblemaEncontrado]
    melhorias_clareza: List[ProblemaEncontrado]
    modelo_usado: str
    arquivo_analisado: str
    timestamp: datetime



class LeitorPDF(Protocol):
    
    def extrair_texto_completo(self, caminho_arquivo: str) -> str:
        """Extrai todo o texto do arquivo"""
        ...
    
    def extrair_texto_por_paginas(self, caminho_arquivo: str) -> Tuple[str, List[str]]:
        """Extrai texto separado por páginas/seções"""
        ...


class ProcessadorJSON(Protocol):
    """Contrato para processadores de JSON"""
    
    def extrair_json_de_texto(self, texto: str) -> Optional[Dict]:
        """Extrai estrutura JSON de um texto"""
        ...


class ValidadorConteudo(Protocol):
    """Contrato para validadores de conteúdo"""
    
    def validar_problemas_contra_pdf(
        self, 
        problemas: List[Dict], 
        paginas_pdf: List[str]
    ) -> Tuple[List[ProblemaEncontrado], List[Dict]]:
        """Valida problemas encontrados contra o PDF original"""
        ...


class GeradorRelatorio(Protocol):
    """Contrato para geradores de relatório"""
    
    def gerar_relatorio_markdown(self, resultado: ResultadoAnalise) -> str:
        """Gera relatório em formato Markdown"""
        ...


# =============================================================================
# IMPLEMENTAÇÕES CONCRETAS
# =============================================================================

class LeitorDocumentoPyMuPDF:
    """Implementação concreta do leitor de documentos usando PyMuPDF para PDF e python-docx para DOCX"""
    
    def extrair_texto_completo(self, caminho_arquivo: str) -> str:
        """Extrai todo o texto do documento como uma string única"""
        if not self._validar_arquivo_existe(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo '{caminho_arquivo}' não encontrado")
        
        try:
            # Determinar o tipo de arquivo
            if caminho_arquivo.lower().endswith('.docx'):
                return self._extrair_texto_docx(caminho_arquivo)
            elif caminho_arquivo.lower().endswith('.pdf'):
                return self._extrair_texto_pdf(caminho_arquivo)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {caminho_arquivo}")
                
        except ImportError as e:
            if 'docx' in str(e):
                raise ImportError("python-docx não instalado. Instale com: pip install python-docx")
            else:
                raise ImportError("PyMuPDF não instalado. Instale com: pip install PyMuPDF")
        except Exception as e:
            raise RuntimeError(f"Erro ao ler arquivo '{caminho_arquivo}': {str(e)}")
    
    def extrair_texto_por_paginas(self, caminho_arquivo: str) -> Tuple[str, List[str]]:
        """Extrai texto separado por páginas/parágrafos"""
        if not self._validar_arquivo_existe(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo '{caminho_arquivo}' não encontrado")
        
        try:
            if caminho_arquivo.lower().endswith('.docx'):
                return self._extrair_texto_docx_por_paragrafos(caminho_arquivo)
            elif caminho_arquivo.lower().endswith('.pdf'):
                return self._extrair_texto_pdf_por_paginas(caminho_arquivo)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {caminho_arquivo}")
        except Exception as e:
            raise RuntimeError(f"Erro ao extrair seções do arquivo '{caminho_arquivo}': {str(e)}")
    
    def _extrair_texto_docx(self, caminho_arquivo: str) -> str:
        """Extrai texto de arquivo DOCX"""
        doc = Document(caminho_arquivo)
        paragrafos = [paragrafo.text for paragrafo in doc.paragraphs if paragrafo.text.strip()]
        
        if not paragrafos:
            raise ValueError(f"Arquivo '{caminho_arquivo}' está vazio ou não contém texto extraível")
        
        return "\n".join(paragrafos)
    
    def _extrair_texto_docx_por_paragrafos(self, caminho_arquivo: str) -> Tuple[str, List[str]]:
        """Extrai texto de DOCX separado por parágrafos"""
        doc = Document(caminho_arquivo)
        paragrafos = [paragrafo.text for paragrafo in doc.paragraphs if paragrafo.text.strip()]
        texto_completo = "\n".join(paragrafos)
        return texto_completo, paragrafos
    
    def _extrair_texto_pdf(self, caminho_arquivo: str) -> str:
        """Extrai texto de arquivo PDF"""
        with fitz.open(caminho_arquivo) as documento:
            textos_paginas = [pagina.get_text() for pagina in documento]
            texto_completo = "\n".join(textos_paginas)
            
            if not texto_completo.strip():
                raise ValueError(f"Arquivo '{caminho_arquivo}' está vazio ou não contém texto extraível")
            
            return texto_completo
    
    def _extrair_texto_pdf_por_paginas(self, caminho_arquivo: str) -> Tuple[str, List[str]]:
        """Extrai texto de PDF separado por páginas"""
        with fitz.open(caminho_arquivo) as documento:
            paginas = [pagina.get_text() for pagina in documento]
            texto_completo = "\n".join(paginas)
            return texto_completo, paginas
    
    @staticmethod
    def _validar_arquivo_existe(caminho: str) -> bool:
        """Valida se o arquivo existe"""
        return os.path.exists(caminho)


class ProcessadorJSONInteligente:
    """Processador de JSON que tenta múltiplas estratégias de extração"""
    
    def extrair_json_de_texto(self, texto: str) -> Optional[Dict]:
        """Extrai JSON usando múltiplas estratégias"""
        estrategias = [
            self._extrair_json_com_marcadores,
            self._extrair_json_com_codigo_block,
            self._extrair_ultimo_objeto_balanceado,
            self._extrair_json_com_placeholders,
            self._extrair_json_puro
        ]
        
        for estrategia in estrategias:
            resultado = estrategia(texto)
            if resultado:
                return resultado
        
        return None
    
    def _extrair_json_com_marcadores(self, texto: str) -> Optional[Dict]:
        """Extrai JSON entre marcadores ```json```"""
        # Using more specific regex patterns to avoid reluctant quantifiers
        padrao = r"```json\s*(\{[^}]*\}|\[[^\]]*\])\s*```"
        matches = re.findall(padrao, texto, flags=re.DOTALL | re.IGNORECASE)
        
        for match in reversed(matches):  # Prefere o último
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _extrair_json_com_codigo_block(self, texto: str) -> Optional[Dict]:
        """Extrai JSON entre marcadores de código genéricos"""
        # Using more specific regex patterns to avoid reluctant quantifiers
        padrao = r"```\s*(\{[^}]*\}|\[[^\]]*\])\s*```"
        matches = re.findall(padrao, texto, flags=re.DOTALL)
        
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _extrair_ultimo_objeto_balanceado(self, texto: str) -> Optional[Dict]:
        """Extrai o último objeto JSON balanceado do texto"""
        ultimo_objeto = None
        pilha_chaves = []
        indice_inicio = None
        
        for indice, caractere in enumerate(texto):
            if caractere == '{':
                if not pilha_chaves:
                    indice_inicio = indice
                pilha_chaves.append('{')
            elif caractere == '}' and pilha_chaves:  # Merged if conditions
                pilha_chaves.pop()
                if not pilha_chaves and indice_inicio is not None:
                    candidato = texto[indice_inicio:indice + 1]
                    ultimo_objeto = candidato
                    indice_inicio = None
        
        return self._tentar_parsear_json(ultimo_objeto)
    
    def _tentar_parsear_json(self, objeto_json: Optional[str]) -> Optional[Dict]:
        """Tenta fazer parse de um objeto JSON"""
        if objeto_json:
            try:
                return json.loads(objeto_json)
            except json.JSONDecodeError:
                pass
        return None
    
    def _extrair_json_com_placeholders(self, texto: str) -> Optional[Dict]:
        """Tenta detectar e completar JSONs com placeholders como [...]"""
        # Verifica se há estrutura JSON com placeholders
        if "[...]" in texto or "..." in texto:
            # Busca por estrutura básica que contém as seções esperadas
            padrao_estrutura = r'\{\s*"erros_gramaticais".*?"necessidades_citacao".*?"melhorias_clareza".*?\}'
            match = re.search(padrao_estrutura, texto, re.DOTALL | re.IGNORECASE)
            
            if match:
                json_texto = match.group(0)
                # Substitui placeholders por arrays vazios
                json_texto = re.sub(r'\[\.\.\.?\]', '[]', json_texto)
                json_texto = re.sub(r'\.\.\.', '', json_texto)
                
                # Tenta parsear o JSON corrigido
                try:
                    return json.loads(json_texto)
                except json.JSONDecodeError:
                    # Se ainda há erro, cria estrutura mínima válida
                    return {
                        "erros_gramaticais": [],
                        "necessidades_citacao": [],
                        "melhorias_clareza": []
                    }
        
        return None

    def _extrair_json_puro(self, texto: str) -> Optional[Dict]:
        """Tenta parsear o texto inteiro como JSON"""
        try:
            return json.loads(texto.strip())
        except json.JSONDecodeError:
            return None


class ValidadorConteudoPDF:
    """Validador que verifica se problemas encontrados existem no PDF"""
    
    def __init__(self, leitor_pdf: LeitorPDF):
        self._leitor_pdf = leitor_pdf
    
    def validar_problemas_contra_pdf(
        self, 
        problemas: List[Dict], 
        paginas_pdf: List[str]
    ) -> Tuple[List[ProblemaEncontrado], List[Dict]]:
        """Valida problemas contra o conteúdo do PDF"""
        problemas_validos = []
        problemas_descartados = []
        
        for problema_dict in problemas or []:
            trecho = problema_dict.get("trecho_exato", "").strip()
            
            if not trecho:
                problema_dict["_motivo_descarte"] = "Sem 'trecho_exato'"
                problemas_descartados.append(problema_dict)
                continue
            
            pagina_encontrada = self._localizar_pagina_do_trecho(trecho, paginas_pdf)
            
            if pagina_encontrada is None:
                problema_dict["_motivo_descarte"] = "Trecho não encontrado no PDF"
                problemas_descartados.append(problema_dict)
                continue
            
            problema = self._converter_dict_para_problema(problema_dict, pagina_encontrada)
            problemas_validos.append(problema)
        
        return problemas_validos, problemas_descartados
    
    def _localizar_pagina_do_trecho(self, trecho: str, paginas: List[str]) -> Optional[int]:
        """Localiza em qual página está um trecho específico"""
        # Use a robust normalization: remove diacritics, punctuation, collapse whitespace, lowercase
        import unicodedata
        from difflib import SequenceMatcher

        def _normalize(text: str) -> str:
            if not text:
                return ""
            # Normalize unicode and remove combining marks (accents)
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(ch for ch in text if not unicodedata.combining(ch))
            # Lowercase
            text = text.lower()
            # Replace punctuation with spaces, keep alphanumerics and whitespace
            text = re.sub(r"[^\w\s]", " ", text)
            # Collapse whitespace
            text = " ".join(text.split())
            return text

        trecho_normalizado = _normalize(trecho)[:2000]

        if not trecho_normalizado:
            return None

        # First try exact containment on normalized text
        for indice, texto_pagina in enumerate(paginas):
            pagina_normalizada = _normalize(texto_pagina)
            if trecho_normalizado in pagina_normalizada:
                return indice + 1  # Páginas começam em 1

        # Fallback: fuzzy match using SequenceMatcher. This handles small differences/typos.
        for indice, texto_pagina in enumerate(paginas):
            pagina_normalizada = _normalize(texto_pagina)
            # Compute overall similarity
            try:
                ratio = SequenceMatcher(None, trecho_normalizado, pagina_normalizada).ratio()
            except Exception:
                ratio = 0.0

            # If reasonably similar, accept
            if ratio >= 0.70:
                return indice + 1

            # Also try matching a shorter slice (first 200 chars) of the trecho to the page
            slice_trecho = trecho_normalizado[:200]
            if slice_trecho:
                try:
                    slice_ratio = SequenceMatcher(None, slice_trecho, pagina_normalizada).ratio()
                except Exception:
                    slice_ratio = 0.0

                if slice_ratio >= 0.75:
                    return indice + 1

        return None
    
    def _converter_dict_para_problema(self, problema_dict: Dict, pagina: int) -> ProblemaEncontrado:
        """Converte dicionário em objeto ProblemaEncontrado"""
        gravidade_str = problema_dict.get("gravidade", "baixa")
        gravidade = GravidadeProblema(gravidade_str)
        
        return ProblemaEncontrado(
            localizacao=problema_dict.get("localizacao", ""),
            trecho_exato=problema_dict.get("trecho_exato", ""),
            descricao=problema_dict.get("descricao", ""),
            sugestao=problema_dict.get("sugestao", ""),
            gravidade=gravidade,
            pagina=pagina
        )


class GeradorRelatorioMarkdown:
    """Gerador de relatórios em formato Markdown"""
    
    def gerar_relatorio_markdown(self, resultado: ResultadoAnalise) -> str:
        """Gera relatório completo em Markdown"""
        linhas = []
        
        self._adicionar_cabecalho(linhas, resultado)
        self._adicionar_secao_problemas(linhas, "Seção A: Problemas Gramaticais", resultado.erros_gramaticais)
        self._adicionar_secao_problemas(linhas, "Seção B: Necessidades de Citação", resultado.necessidades_citacao)
        self._adicionar_secao_problemas(linhas, "Seção C: Melhorias de Clareza", resultado.melhorias_clareza)
        
        return "\n".join(linhas)
    
    def _adicionar_cabecalho(self, linhas: List[str], resultado: ResultadoAnalise):
        """Adiciona cabeçalho do relatório"""
        linhas.extend([
            "# Relatório de Análise Textual",
            f"**Modelo:** {resultado.modelo_usado}",
            f"**Data:** {resultado.timestamp.strftime('%d/%m/%Y %H:%M:%S')}",
            f"**Arquivo analisado:** {resultado.arquivo_analisado}",
            "",
            "## Resultado Validado (somente itens com evidência literal no PDF)"
        ])
    
    def _adicionar_secao_problemas(self, linhas: List[str], titulo: str, problemas: List[ProblemaEncontrado]):
        """Adiciona seção de problemas ao relatório"""
        linhas.append(f"### {titulo}")
        
        if not problemas:
            linhas.append("_Sem itens validados nessa seção._")
            linhas.append("")
            return
        
        for indice, problema in enumerate(problemas, 1):
            linhas.extend([
                f"**{indice}. Localização:** página {problema.pagina}, {problema.localizacao}",
                f"- **Trecho exato:** \"{problema.trecho_exato}\"",
                f"- **Problema:** {problema.descricao}",
                f"- **Sugestão:** {problema.sugestao}",
                f"- **Gravidade:** {problema.gravidade.value}",
                ""
            ])


class ConfiguradorLiteLLM:
    @staticmethod
    def aplicar_patch_parametros():
        """Aplica patch para remover parâmetros problemáticos"""
        completion_original = litellm.completion
        
        def completion_sem_parametros_problematicos(**parametros):
            # Remove parâmetros que causam erro 422 na Replicate
            parametros_problematicos = ["stop", "stop_sequences", "stops", "stop_tokens"]
            
            for parametro in parametros_problematicos:
                parametros.pop(parametro, None)
            
            # Remove também de extra_body se existir
            if "extra_body" in parametros and isinstance(parametros["extra_body"], dict):
                for parametro in parametros_problematicos:
                    parametros["extra_body"].pop(parametro, None)
            
            return completion_original(**parametros)
        
        litellm.completion = completion_sem_parametros_problematicos


class FabricaModelos:
    """Factory para criação de modelos de IA"""
    
    CONFIGURACOES_PADRAO = {
        ModelType.LLAMA_7B: ConfiguracaoModelo(
            model_name="replicate/meta/meta-llama-3-8b-instruct",
            display_name="Llama 7B",
            api_key=os.getenv('REPLICATE_API_TOKEN', 'r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn')
        ),
        ModelType.LLAMA_70B: ConfiguracaoModelo(
            model_name="replicate/meta/meta-llama-3-70b-instruct",
            display_name="Llama 70B",
            api_key=os.getenv('REPLICATE_API_TOKEN', 'r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn'),
            temperature=0.2,
            max_tokens=8192
        ),
        ModelType.CHATGPT: ConfiguracaoModelo(
            model_name="replicate/openai/gpt-4o-mini",
            display_name="ChatGPT 4.0",
            api_key=os.getenv('OPENAI_API_KEY', 'r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn')
        )
    }
    
    @classmethod
    def criar_modelo(cls, tipo_modelo: ModelType) -> LLM:
        """Cria instância de modelo baseada no tipo"""
        config = cls.CONFIGURACOES_PADRAO[tipo_modelo]
        
        return LLM(
            model=config.model_name,
            api_key=config.api_key,
            drop_params=["stop"],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
        )
    
    @classmethod
    def obter_nome_exibicao(cls, tipo_modelo: ModelType) -> str:
        """Obtém nome de exibição do modelo"""
        return cls.CONFIGURACOES_PADRAO[tipo_modelo].display_name


# =============================================================================
# FERRAMENTAS DO CREWAI
# =============================================================================

class ArgumentosLeituraDocumento(BaseModel):
    """Argumentos para ferramenta de leitura de documentos"""
    pdf_path: str = Field(..., description="Caminho do arquivo de documento a ser lido (PDF ou DOCX)")


class FerramentaLeituraDocumento(BaseTool):
    """Ferramenta do CrewAI para leitura de documentos"""
    
    name: str = "LerTextoPDFTool"
    description: str = (
        "Lê o conteúdo de um arquivo de documento (PDF ou DOCX) e retorna o texto extraído. "
        "Use passando um dicionário com a chave 'pdf_path', por ex.: "
        '{"pdf_path": "texto_estudo_caso1.docx"}'
    )
    args_schema: Type[BaseModel] = ArgumentosLeituraDocumento
    
    def __init__(self, leitor_pdf: LeitorPDF):
        super().__init__()
        self._leitor_pdf = leitor_pdf
    
    def _run(self, pdf_path: str) -> str:
        """Executa a leitura do documento"""
        try:
            return self._leitor_pdf.extrair_texto_completo(pdf_path)
        except FileNotFoundError as e:
            return f"Erro: {str(e)}"
        except ValueError as e:
            return f"Aviso: {str(e)}"
        except ImportError as e:
            return f"Erro: {str(e)}"
        except Exception as e:
            return f"Erro ao ler arquivo de documento: {str(e)}"


# =============================================================================
# SERVIÇOS DE ANÁLISE
# =============================================================================

class FabricaAgentes:
    """Factory para criação de agentes especializados"""
    
    @staticmethod
    def criar_agente_analise_textual(modelo: LLM, ferramenta_documento: FerramentaLeituraDocumento) -> Agent:
        """Cria agente especializado em análise textual"""
        return Agent(
            role="Especialista em Análise Textual Acadêmica",
            goal="Analisar textos científicos identificando problemas gramaticais, falta de citações e oportunidades de melhoria na clareza",
            backstory="""
            Você é um especialista em revisão de textos acadêmicos com vasta experiência em:
            - Correção gramatical e ortográfica
            - Identificação de pontos que necessitam citações
            - Análise de clareza e fluidez textual
            - Padrões de escrita científica
            """,
            verbose=True,
            tools=[ferramenta_documento],
            llm=modelo,
            max_iter=3,
            max_execution_time=300,
        )
    
    @staticmethod
    def criar_agente_melhoria_redacao(modelo: LLM) -> Agent:
        """Cria agente especializado em melhoria e redação"""
        return Agent(
            role="Especialista em Melhoria e Redação Científica",
            goal="Organizar e compilar relatórios de análise textual com sugestões de melhoria estruturadas",
            backstory="""
            Você é um especialista em redação científica responsável por:
            - Organizar apontamentos de revisão de forma clara
            - Estruturar relatórios de análise textual
            - Priorizar melhorias por importância
            - Apresentar sugestões de forma didática
            """,
            verbose=True,
            llm=modelo,
            max_iter=3,
            max_execution_time=300,
        )


class FabricaTarefas:
    """Factory para criação de tarefas especializadas"""
    
    @staticmethod
    def criar_tarefa_analise(agente: Agent, ferramenta_documento: FerramentaLeituraDocumento, caminho_arquivo: str, texto_documento: Optional[str] = None) -> Task:
        """Cria tarefa de análise textual"""
        # Build base description
        base_desc = (
            f"VOCÊ SÓ PODE USAR O CONTEÚDO DO ARQUIVO '{caminho_arquivo}'.\n"
            "NÃO invente exemplos, não use conhecimento externo, não infira conteúdo que não esteja literalmente no documento.\n"
            "Se não encontrar algo, devolva listas vazias.\n\n"
            f"1) **OBRIGATÓRIO**: Leia o documento usando a ferramenta LerTextoPDFTool (passe EXATAMENTE: {{\"pdf_path\": \"{caminho_arquivo}\"}}).\n"
            "   VOCÊ DEVE USAR A FERRAMENTA ANTES DE FAZER QUALQUER ANÁLISE.\n\n"
            "2) Analise APENAS o texto lido e identifique:\n"
            "   - ERROS GRAMATICAIS (problemas de ortografia, gramática ou estilo/clareza acadêmica).\n"
            "   - NECESSIDADES DE CITAÇÃO (apenas se houver afirmações no documento sem referência explícita).\n"
            "   - MELHORIAS DE CLAREZA (frases ambíguas, transições fracas, jargões não explicados).\n\n"
            "3) Para CADA problema encontrado, você DEVE:\n"
            "   - Copiar o **trecho exato** do documento (curto, até ~250 caracteres) no campo \"trecho_exato\".\n"
            "   - Informar a **localização** (ex.: \"Introdução\", \"Metodologia\", \"Resultados\", \"Conclusão\"... ou \"parágrafo X da seção Y\").\n"
            "   - Descrever o problema em \"descricao\".\n"
            "   - Sugerir uma correção em \"sugestao\".\n"
            "   - Estimar \"gravidade\" (baixa|média|alta).\n\n"
            "4) **SAÍDA OBRIGATÓRIA**: retorne APENAS um objeto JSON (sem nenhum outro texto) no formato:\n"
            "{\n  \"erros_gramaticais\": [\n    {\"localizacao\": \"...\", \"trecho_exato\": \"...\", \"descricao\": \"...\", \"sugestao\": \"...\", \"gravidade\": \"...\" }\n  ],\n"
            "  \"necessidades_citacao\": [\n    {\"localizacao\": \"...\", \"trecho_exato\": \"...\", \"descricao\": \"...\", \"sugestao\": \"...\", \"gravidade\": \"...\" }\n  ],\n"
            "  \"melhorias_clareza\": [\n    {\"localizacao\": \"...\", \"trecho_exato\": \"...\", \"descricao\": \"...\", \"sugestao\": \"...\", \"gravidade\": \"...\" }\n  ]\n}\n\n"
            "REGRAS:\n"
            "- \"trecho_exato\" DEVE existir literalmente no documento. Se não tiver certeza, NÃO inclua o item.\n"
            "- Se nada for encontrado em alguma seção, use lista vazia [].\n\n"
            "IMPORTANTE: Se houver erro na leitura do arquivo, retorne a mensagem de erro exata.\n"
        )

        # If texto_documento is provided, append a trimmed fallback block so models that don't call tools still have the content
        if texto_documento:
            trimmed = texto_documento.strip()
            if len(trimmed) > 20000:
                trimmed = trimmed[:20000] + "\n...[TRIMMED]"
            fallback_block = "\n\n--- INÍCIO DO TEXTO DO DOCUMENTO (FALLBACK) ---\n" + trimmed + "\n--- FIM DO TEXTO DO DOCUMENTO ---\n"
            description = base_desc + fallback_block
        else:
            description = base_desc

        return Task(
            description=description,
            expected_output="Lista detalhada de problemas encontrados com localizações e sugestões de correção",
            agent=agente,
            tools=[ferramenta_documento]
        )


class ServicoAnaliseTexto:
    """Serviço principal para análise de texto científico"""
    
    def __init__(
        self,
        leitor_pdf: LeitorPDF,
        processador_json: ProcessadorJSON,
        validador_conteudo: ValidadorConteudo,
        gerador_relatorio: GeradorRelatorio
    ):
        self._leitor_pdf = leitor_pdf
        self._processador_json = processador_json
        self._validador_conteudo = validador_conteudo
        self._gerador_relatorio = gerador_relatorio
    
    def analisar_documento(
        self, 
        tipo_modelo: ModelType, 
        caminho_arquivo: str
    ) -> Tuple[Optional[ResultadoAnalise], Optional[str]]:
        """Executa análise completa de um documento"""
        try:
            # Validações iniciais
            if not os.path.exists(caminho_arquivo):
                erro = f"Arquivo '{caminho_arquivo}' não encontrado!"
                return None, erro
            
            # Configurar modelo e agentes
            modelo = FabricaModelos.criar_modelo(tipo_modelo)
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            
            ferramenta_documento = FerramentaLeituraDocumento(self._leitor_pdf)
            agente_analise = FabricaAgentes.criar_agente_analise_textual(modelo, ferramenta_documento)

            # Extrair texto do documento como fallback para modelos que não chamam ferramentas
            try:
                texto_documento = self._leitor_pdf.extrair_texto_completo(caminho_arquivo)
            except Exception:
                texto_documento = None

            # Criar e executar tarefa (inclui fallback com texto completo do documento)
            tarefa_analise = FabricaTarefas.criar_tarefa_analise(agente_analise, ferramenta_documento, caminho_arquivo, texto_documento)
            
            crew = Crew(
                agents=[agente_analise],
                tasks=[tarefa_analise],
                process=Process.sequential,
                verbose=True,
                max_execution_time=900,
            )
            
            print(f"🚀 Iniciando execução com {nome_modelo}...")
            resultado_crew = crew.kickoff()
            
            # Processar resultado
            if not resultado_crew or str(resultado_crew).strip() == "":
                return None, f"Resultado vazio para {nome_modelo}"
            
            # Extrair e validar JSON
            json_extraido = self._processador_json.extrair_json_de_texto(str(resultado_crew))
            if not json_extraido:
                return None, f"Não foi possível extrair JSON válido da resposta de {nome_modelo}"
            
            # Validar contra PDF
            _, paginas_pdf = self._leitor_pdf.extrair_texto_por_paginas(caminho_arquivo)
            
            erros_validos, _ = self._validador_conteudo.validar_problemas_contra_pdf(
                json_extraido.get("erros_gramaticais", []), paginas_pdf
            )
            citacoes_validas, _ = self._validador_conteudo.validar_problemas_contra_pdf(
                json_extraido.get("necessidades_citacao", []), paginas_pdf
            )
            clareza_valida, _ = self._validador_conteudo.validar_problemas_contra_pdf(
                json_extraido.get("melhorias_clareza", []), paginas_pdf
            )
            
            # Criar resultado final
            resultado_final = ResultadoAnalise(
                erros_gramaticais=erros_validos,
                necessidades_citacao=citacoes_validas,
                melhorias_clareza=clareza_valida,
                modelo_usado=nome_modelo,
                arquivo_analisado=caminho_arquivo,
                timestamp=datetime.now()
            )
            
            return resultado_final, None
            
        except TimeoutError:
            return None, f"Timeout: {nome_modelo} excedeu o tempo limite"
        except Exception as e:
            return None, f"Erro ao executar análise com {nome_modelo}: {str(e)}"


class GerenciadorEstudoCaso:
    """Gerenciador principal do estudo de caso"""
    
    def __init__(self, servico_analise: ServicoAnaliseTexto, gerador_relatorio: GeradorRelatorio):
        self._servico_analise = servico_analise
        self._gerador_relatorio = gerador_relatorio
        self._diretorio_resultados = Path("resultados_estudo_caso")
    
    def executar_estudo_completo(self, caminho_arquivo: str = "texto_estudo_caso1.docx"):
        """Executa estudo de caso completo com todos os modelos"""
        print("🚀 INICIANDO ESTUDO DE CASO 1 - ANÁLISE DE TEXTO CIENTÍFICO")
        print("📋 Conforme especificações do professor:")
        print("   - 2 agentes: Análise Textual + Melhoria/Redação")
        print("   - 3 modelos: Llama 7B, Llama 70B, ChatGPT")
        print()
        
        self._garantir_diretorio_resultados()
        
        resultados_sucesso = {}
        
        for tipo_modelo in ModelType:
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            print(f"\n🔄 Testando modelo: {nome_modelo}")
            print("=" * 60)
            print(f"EXECUTANDO ANÁLISE COM: {nome_modelo}")
            print("=" * 60)
            
            resultado, erro = self._servico_analise.analisar_documento(tipo_modelo, caminho_arquivo)

            # If analysis succeeded, save the normal report. If it failed, still create a debug report
            if resultado:
                caminho_relatorio = self._salvar_relatorio_individual(resultado, tipo_modelo)
                resultados_sucesso[tipo_modelo] = {
                    "resultado": resultado,
                    "arquivo": caminho_relatorio,
                    "status": "✅ Sucesso"
                }
                print(f"✅ {nome_modelo}: Concluído com sucesso")
            else:
                # Create a minimal debug report for failed runs so the file exists and contains the error
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome_arquivo = f"relatorio_{tipo_modelo.value}_{timestamp}_FAILED.md"
                caminho_arquivo_saida = self._diretorio_resultados / nome_arquivo

                linhas = [
                    f"# Relatório (FALHA) - {nome_modelo}",
                    "",
                    f"**Arquivo analisado:** {caminho_arquivo}",
                    f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                    "",
                    "## Erro",
                    "",
                    f"{erro}",
                    "",
                    "## Observações para debug",
                    "",
                    "- Verifique o raw output do modelo e o log de execução.",
                ]

                with open(caminho_arquivo_saida, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(linhas))

                resultados_sucesso[tipo_modelo] = {
                    "resultado": None,
                    "arquivo": str(caminho_arquivo_saida),
                    "status": f"❌ Falhou: {erro}"
                }

                print(f"❌ {nome_modelo}: {erro} - relatório de debug salvo em {caminho_arquivo_saida}")
        
        # Gerar relatório consolidado
        self._gerar_relatorio_consolidado(resultados_sucesso, caminho_arquivo)
        
        print("\n🎉 ESTUDO DE CASO CONCLUÍDO!")
        print(f"📁 Resultados salvos em: {self._diretorio_resultados}/")
        sucessos = sum(1 for r in resultados_sucesso.values() if r["status"].startswith("✅"))
        print(f"📊 Modelos testados: {sucessos}/{len(ModelType)}")
    
    def _garantir_diretorio_resultados(self):
        """Garante que o diretório de resultados existe"""
        self._diretorio_resultados.mkdir(parents=True, exist_ok=True)
    
    def _salvar_relatorio_individual(self, resultado: ResultadoAnalise, tipo_modelo: ModelType) -> str:
        """Salva relatório individual de um modelo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"relatorio_{tipo_modelo.value}_{timestamp}.md"
        caminho_arquivo = self._diretorio_resultados / nome_arquivo
        
        conteudo_relatorio = self._gerador_relatorio.gerar_relatorio_markdown(resultado)
        
        with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo_relatorio)
        
        print(f"\n✅ Relatório salvo em: {caminho_arquivo}")
        return str(caminho_arquivo)
    
    def _gerar_relatorio_consolidado(self, resultados: Dict, caminho_arquivo: str):
        """Gera relatório consolidado de todos os modelos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_consolidado = self._diretorio_resultados / f"relatorio_consolidado_{timestamp}.md"
        
        linhas = [
            "# Relatório Consolidado - Estudo de Caso 1",
            "",
            f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            f"**Modelos testados:** {len(resultados)}",
            f"**Arquivo analisado:** {caminho_arquivo}",
            "",
            "## Resumo dos Testes",
            ""
        ]
        
        sucessos = 0
        for tipo_modelo, dados in resultados.items():
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            linhas.extend([
                f"### {nome_modelo}",
                f"- **Status:** {dados['status']}",
            ])
            
            if dados['arquivo']:
                linhas.append(f"- **Arquivo:** {dados['arquivo']}")
                sucessos += 1
            
            linhas.append("")
        
        # Estatísticas
        linhas.extend([
            "## Estatísticas",
            "",
            f"- **Total de modelos:** {len(ModelType)}",
            f"- **Sucessos:** {sucessos}",
            f"- **Falhas:** {len(resultados) - sucessos}",
            f"- **Taxa de sucesso:** {(sucessos/len(resultados)*100):.1f}%",
            ""
        ])
        
        with open(caminho_consolidado, "w", encoding="utf-8") as arquivo:
            arquivo.write("\n".join(linhas))
        
        print(f"📋 Relatório consolidado salvo em: {caminho_consolidado}")


# =============================================================================
# PONTO DE ENTRADA PRINCIPAL
# =============================================================================

def main():
    """Função principal do programa"""
    # Configurar infraestrutura
    ConfiguradorLiteLLM.aplicar_patch_parametros()
    
    # Criar dependências
    leitor_pdf = LeitorDocumentoPyMuPDF()
    processador_json = ProcessadorJSONInteligente()
    validador_conteudo = ValidadorConteudoPDF(leitor_pdf)
    gerador_relatorio = GeradorRelatorioMarkdown()
    
    # Criar serviços
    servico_analise = ServicoAnaliseTexto(
        leitor_pdf=leitor_pdf,
        processador_json=processador_json,
        validador_conteudo=validador_conteudo,
        gerador_relatorio=gerador_relatorio
    )
    
    # Criar gerenciador e executar
    gerenciador = GerenciadorEstudoCaso(servico_analise, gerador_relatorio)
    gerenciador.executar_estudo_completo()


if __name__ == "__main__":
    main()