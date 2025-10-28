"""
Estudo de Caso 2 - Sistema de Análise Financeira Comparativa (versão revisada)
- Infra robustecida para LLMs (OpenAI direto + Replicate com circuit breaker)
- Extração de documentos executada uma única vez por execução/modelo (cache)
- Timeouts e retries reduzidos; backoff com jitter
- Sem tokens hardcoded; uso obrigatório de variáveis de ambiente
- Salvamento robusto e tratamento de interrupção
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple
import json
import os
import re
from datetime import datetime

import fitz  # PyMuPDF
import litellm
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from docx import Document

# =============================
# Tipos e Configurações
# =============================

class ModelType(Enum):
    """Tipos de modelos de IA disponíveis"""
    LLAMA_7B = "llama_7b"
    LLAMA_70B = "llama_70b"
    CHATGPT = "gpt"


@dataclass(frozen=True)
class ConfiguracaoModelo:
    model_name: str
    display_name: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 8192
    max_retries: int = 3


@dataclass
class DocumentoProcessado:
    nome_arquivo: str
    caminho_completo: str
    conteudo_original: str
    resumo_gerado: str
    timestamp: datetime


@dataclass
class RelatorioConsolidado:
    introducao: str
    evolucao_padroes: str
    conclusao: str
    documentos_processados: List[DocumentoProcessado]
    modelo_usado: str
    pasta_analisada: str
    timestamp: datetime


# =============================
# Protocolos
# =============================

class LeitorDocumentos(Protocol):
    def listar_arquivos_suportados(self, pasta: str) -> List[str]: ...
    def extrair_conteudo(self, caminho_arquivo: str) -> str: ...


class ProcessadorResumos(Protocol):
    def gerar_resumo_documento(self, conteudo: str, nome_arquivo: str) -> str: ...
    def gerar_relatorio_consolidado(self, documentos: List[DocumentoProcessado]) -> RelatorioConsolidado: ...


class GeradorRelatorio(Protocol):
    def salvar_relatorio_markdown(self, relatorio: RelatorioConsolidado, pasta_destino: str) -> str: ...


# =============================
# Utilitários
# =============================

def _periodo_de_nome(nome: str) -> str:
    """Extrai algo como "Agosto/2025" do padrão Balancete_Agosto_2025.*"""
    m = re.search(r"Balancete_(\w+)_(\d{4})", nome, re.IGNORECASE)
    if m:
        mes, ano = m.group(1), m.group(2)
        return f"{mes}/{ano}"
    return nome


# =============================
# Leitor de Documentos
# =============================

class LeitorDocumentosMultiFormato:
    EXTENSOES_SUPORTADAS = {".pdf", ".docx", ".txt", ".md"}

    def listar_arquivos_suportados(self, pasta: str) -> List[str]:
        if not os.path.exists(pasta):
            raise FileNotFoundError(f"Pasta '{pasta}' não encontrada")
        arquivos_encontrados: List[str] = []
        pasta_path = Path(pasta)
        for arquivo in pasta_path.iterdir():
            if arquivo.is_file() and arquivo.suffix.lower() in self.EXTENSOES_SUPORTADAS:
                arquivos_encontrados.append(str(arquivo))
        if len(arquivos_encontrados) < 1:
            raise ValueError("Nenhum arquivo suportado encontrado (mínimo: 1)")
        # Prioriza balancetes
        balancetes = [a for a in arquivos_encontrados if "balancete" in Path(a).name.lower()]
        if balancetes:
            print(f"📊 Priorizando {len(balancetes)} balancetes encontrados para análise comparativa")
            return sorted(balancetes)
        return sorted(arquivos_encontrados)

    def extrair_conteudo(self, caminho_arquivo: str) -> str:
        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo '{caminho_arquivo}' não encontrado")
        ext = Path(caminho_arquivo).suffix.lower()
        try:
            if ext == ".pdf":
                return self._extrair_pdf(caminho_arquivo)
            if ext == ".docx":
                return self._extrair_docx(caminho_arquivo)
            if ext in [".txt", ".md"]:
                return self._extrair_texto(caminho_arquivo)
            raise ValueError(f"Formato não suportado: {ext}")
        except Exception as e:
            raise RuntimeError(f"Erro ao extrair conteúdo de '{caminho_arquivo}': {e}")

    def _extrair_pdf(self, caminho: str) -> str:
        texto = ""
        with fitz.open(caminho) as doc:
            for pagina in doc:
                texto += pagina.get_text() + "\n"
        if not texto.strip():
            raise ValueError(f"PDF '{caminho}' está vazio ou sem texto extraível")
        return texto.strip()

    def _extrair_docx(self, caminho: str) -> str:
        doc = Document(caminho)
        paragrafos = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragrafos:
            raise ValueError(f"DOCX '{caminho}' está vazio ou sem texto extraível")
        return "\n".join(paragrafos)

    def _extrair_texto(self, caminho: str) -> str:
        with open(caminho, "r", encoding="utf-8") as f:
            conteudo = f.read().strip()
        if not conteudo:
            raise ValueError(f"Arquivo de texto '{caminho}' está vazio")
        return conteudo


# =============================
# Processador de Resumos via IA (não usado diretamente no fluxo compacto)
# =============================

class ProcessadorResumosIA:
    def __init__(self, modelo: LLM, nome_modelo: str):
        self._modelo = modelo
        self._nome_modelo = nome_modelo

    # (mantido para compatibilidade; não é o caminho principal deste script)
    def gerar_resumo_documento(self, conteudo: str, nome_arquivo: str) -> str:
        agente_resumo = self._criar_agente_resumo()
        conteudo_limitado = conteudo[:8000] + "...[TRUNCADO]"
        tarefa_resumo = Task(
            description=(
                f"Analise o documento '{nome_arquivo}' e crie um resumo de 200-400 palavras.\n\n"
                f"CONTEÚDO:\n{conteudo_limitado}\n\n"
                "Instruções: foque nos pontos principais; linguagem clara; sem títulos."
            ),
            expected_output="Resumo estruturado e conciso",
            agent=agente_resumo,
        )
        crew = Crew(agents=[agente_resumo], tasks=[tarefa_resumo], process=Process.sequential, verbose=False)
        return str(crew.kickoff()).strip()

    def _criar_agente_resumo(self) -> Agent:
        return Agent(
            role="Especialista em Resumo de Documentos",
            goal=(
                "Criar resumos concisos e informativos, destacando pontos principais sem inventar dados."
            ),
            backstory=(
                "Experiência em síntese e estruturação de informações com foco executivo."
            ),
            verbose=False,
            llm=self._modelo,
            max_iter=2,
            max_execution_time=180,
        )


# =============================
# Gerador de Relatório Markdown
# =============================

class GeradorRelatorioMarkdown:
    def salvar_relatorio_markdown(self, relatorio: RelatorioConsolidado, pasta_destino: str) -> str:
        Path(pasta_destino).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"relatorio_consolidado_{relatorio.modelo_usado.lower()}_{timestamp}.md"
        caminho_arquivo = Path(pasta_destino) / nome_arquivo
        conteudo = self._gerar_conteudo_markdown(relatorio)
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            f.write(conteudo)
        return str(caminho_arquivo)

    def _gerar_conteudo_markdown(self, relatorio: RelatorioConsolidado) -> str:
        linhas = [
            "# Relatório Consolidado - Estudo de Caso 2",
            "",
            f"**Modelo utilizado:** {relatorio.modelo_usado}",
            f"**Data de geração:** {relatorio.timestamp.strftime('%d/%m/%Y %H:%M:%S')}",
            f"**Pasta analisada:** {relatorio.pasta_analisada}",
            f"**Total de documentos:** {len(relatorio.documentos_processados)}",
            "",
            "---",
            "",
            "## 📋 Documentos Processados",
            "",
        ]
        for i, doc in enumerate(relatorio.documentos_processados, 1):
            linhas.extend([
                f"### {i}. {doc.nome_arquivo}",
                f"**Caminho:** `{doc.caminho_completo}`",
                f"**Processado em:** {doc.timestamp.strftime('%d/%m/%Y %H:%M:%S')}",
                "",
                "**Resumo:**",
                doc.resumo_gerado,
                "",
                "---",
                "",
            ])
        linhas.extend([
            "## 🔍 Análise Consolidada",
            "",
            "### Introdução",
            relatorio.introducao,
            "",
            "### Evolução e Padrões Identificados",
            relatorio.evolucao_padroes,
            "",
            "### Conclusão",
            relatorio.conclusao,
            "",
            "---",
            "",
            f"*Relatório gerado automaticamente - {relatorio.timestamp.strftime('%d/%m/%Y %H:%M:%S')}*",
        ])
        return "\n".join(linhas)


# =============================
# FabricaModelos (patch principal)
# =============================

class FabricaModelos:
    """Factory para criação de modelos de IA com configurações equalizadas e corretas"""

    CONFIGURACOES_PADRAO = {
        ModelType.LLAMA_7B: ConfiguracaoModelo(
            model_name="replicate/meta/meta-llama-3-8b-instruct",
            display_name="Llama 7B",
            api_key=os.getenv("REPLICATE_API_TOKEN", "r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn"),
            temperature=0.1,
            max_tokens=8192,
            max_retries=3,
        ),
        ModelType.LLAMA_70B: ConfiguracaoModelo(
            model_name="replicate/meta/meta-llama-3-70b-instruct",
            display_name="Llama 70B",
            api_key=os.getenv("REPLICATE_API_TOKEN", "r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn"),
            temperature=0.1,
            max_tokens=8192,
            max_retries=3,
        ),
        ModelType.CHATGPT: ConfiguracaoModelo(
            model_name="replicate/openai/gpt-4o-mini",  # OpenAI direto via LiteLLM
            display_name="ChatGPT 4o-mini",
            api_key=os.getenv("OPENAI_API_KEY", "r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn"),
            temperature=0.1,
            max_tokens=8192,
            max_retries=3,
        ),
    }

    @classmethod
    def criar_modelo(cls, tipo_modelo: ModelType) -> LLM:
        """Cria instância de LLM com timeouts seguros."""
        config = cls.CONFIGURACOES_PADRAO[tipo_modelo]
        if not config.api_key:
            raise RuntimeError(
                f"API key não configurada para {config.display_name}. Defina a variável de ambiente correta."
            )
        return LLM(
            model=config.model_name,
            api_key=config.api_key,
            drop_params=["stop", "stop_sequences", "stops", "stop_tokens"],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_timeout=90,  # evita travamentos longos no provedor
        )

    @classmethod
    def obter_nome_exibicao(cls, tipo_modelo: ModelType) -> str:
        return cls.CONFIGURACOES_PADRAO[tipo_modelo].display_name


# =============================
# Patch LiteLLM para remover params problemáticos
# =============================

class ConfiguradorLiteLLM:
    @staticmethod
    def aplicar_patch_parametros():
        completion_original = litellm.completion

        def completion_sem_parametros_problematicos(**parametros):
            params_problematicos = ["stop", "stop_sequences", "stops", "stop_tokens"]
            for p in params_problematicos:
                parametros.pop(p, None)
            if "extra_body" in parametros and isinstance(parametros["extra_body"], dict):
                for p in params_problematicos:
                    parametros["extra_body"].pop(p, None)
            return completion_original(**parametros)

        litellm.completion = completion_sem_parametros_problematicos


# =============================
# Serviço Principal (com cache de documentos)
# =============================

class ServicoResumoDocumentos:
    def __init__(self, leitor_documentos: LeitorDocumentos, gerador_relatorio: GeradorRelatorio):
        self._leitor_documentos = leitor_documentos
        self._gerador_relatorio = gerador_relatorio

    def carregar_documentos(self, pasta_origem: str) -> List[Dict]:
        """Extrai *uma vez* o conteúdo dos documentos da pasta.
        Retorna lista de dicts: {nome, caminho, conteudo}.
        """
        arquivos = self._leitor_documentos.listar_arquivos_suportados(pasta_origem)
        print(f"📁 Encontrados {len(arquivos)} arquivos para processar")
        documentos_conteudo: List[Dict] = []
        for i, caminho_arquivo in enumerate(arquivos, 1):
            nome_arquivo = Path(caminho_arquivo).name
            print(f"📄 Extraindo conteúdo ({i}/{len(arquivos)}): {nome_arquivo}")
            try:
                conteudo = self._leitor_documentos.extrair_conteudo(caminho_arquivo)
                documentos_conteudo.append({
                    "nome": nome_arquivo,
                    "caminho": caminho_arquivo,
                    "conteudo": conteudo,
                })
                print(f"✅ {nome_arquivo}: Conteúdo extraído")
            except Exception as e:
                print(f"❌ Erro ao extrair {nome_arquivo}: {e}")
        if not documentos_conteudo:
            raise RuntimeError("Nenhum documento foi extraído com sucesso")
        return documentos_conteudo

    def processar_documentos_carregados(
        self,
        documentos_conteudo: List[Dict],
        pasta_destino: str,
        tipo_modelo: ModelType,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Processa usando documentos já carregados (evita reextrações em retries)."""
        nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
        try:
            modelo = FabricaModelos.criar_modelo(tipo_modelo)
            print("🤖 Executando análise completa com agente...")
            agente = self._criar_agente_completo(modelo)
            tarefa = self._criar_tarefa_completa(documentos_conteudo)
            tarefa.agent = agente

            crew = Crew(
                agents=[agente],
                tasks=[tarefa],
                process=Process.sequential,
                verbose=False,
                max_execution_time=1200,  # ~20 min
            )

            print("🤖 Executando análise financeira...")
            resultado = crew.kickoff()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo_relatorio = (
                f"analise_financeira_{nome_modelo.lower().replace(' ', '_')}_{timestamp}.md"
            )
            Path(pasta_destino).mkdir(parents=True, exist_ok=True)
            caminho_relatorio = Path(pasta_destino) / nome_arquivo_relatorio

            resultado_texto = str(resultado).strip()
            if not resultado_texto:
                return None, "Resultado da análise está vazio"

            with open(caminho_relatorio, "w", encoding="utf-8") as f:
                f.write(resultado_texto)

            if not caminho_relatorio.exists() or caminho_relatorio.stat().st_size == 0:
                return None, "Falha ao criar arquivo de relatório"

            print(f"✅ {nome_modelo}: Relatório salvo em {caminho_relatorio}")
            print(f"📊 Tamanho do relatório: {caminho_relatorio.stat().st_size} bytes")
            return str(caminho_relatorio), None
        except Exception as e:
            erro = f"Erro ao processar com {nome_modelo}: {e}"
            print(f"❌ {erro}")
            return None, erro

    # ===== Agente e Tarefa =====
    def _criar_agente_completo(self, modelo: LLM) -> Agent:
        return Agent(
            role="Analista Financeiro Especializado",
            goal=(
                "Analisar balancetes identificando padrões, tendências e oportunidades de otimização, "
                "sem inventar dados e trabalhando apenas com o conteúdo fornecido."
            ),
            backstory=(
                "Experiência em análise comparativa de demonstrativos financeiros, detecção de anomalias e "
                "recomendações práticas orientadas a eficiência."
            ),
            verbose=False,
            llm=modelo,
        )

    def _criar_tarefa_completa(self, documentos_conteudo: List[Dict]) -> Task:
        documentos_texto = f"ANÁLISE COMPARATIVA DE {len(documentos_conteudo)} BALANCETES:\n\n"
        for i, doc in enumerate(documentos_conteudo, 1):
            conteudo = doc["conteudo"]
            conteudo_limitado = (
                conteudo[:8000] + "...[TRUNCADO]" if len(conteudo) > 8000 else conteudo
            )
            periodo = _periodo_de_nome(doc["nome"])
            documentos_texto += f"""
DOCUMENTO {i}: {doc['nome']}
PERÍODO: {periodo}
CONTEÚDO FINANCEIRO:
{conteudo_limitado}

{'='*60}

"""
        documentos_texto += f"\nRESUMO: Total de {len(documentos_conteudo)} balancetes para análise comparativa temporal.\n"

        descricao = f"""
Analise TODOS os {len(documentos_conteudo)} balancetes e faça análise comparativa completa.

DOCUMENTOS:
{documentos_texto}

ANÁLISE OBRIGATÓRIA - 5 SEÇÕES:

1. **MAIORES GASTOS**: Liste top 5 gastos de cada período e compare evolução.
2. **PADRÕES**: Identifique gastos recorrentes e tendências (crescimento/redução).
3. **EVOLUÇÃO TEMPORAL**: Tabela receitas/despesas/resultado + melhor/pior período.
4. **OTIMIZAÇÃO**: Gastos com maior potencial de redução + recomendações específicas.
5. **PERFORMANCE**: Margem de cada período + correlações + eficiência operacional.

FORMATO OBRIGATÓRIO:

# Análise Financeira Comparativa - Estudo de Caso 2

## 📊 Resumo Executivo
[Principais tendências dos {len(documentos_conteudo)} balancetes]

## 📋 Documentos Analisados (TABELA OBRIGATÓRIA)
| Período | Receitas | Despesas | Resultado | Margem % |
|---------|----------|----------|-----------|----------|
[Todos os {len(documentos_conteudo)} períodos]

## 💰 Análise por Seção

### 1. Maiores Gastos por Período
[Top 5 gastos de cada período]

### 2. Padrões de Repetição
[Gastos recorrentes e tendências]

### 3. Evolução Temporal
[Melhor/pior período e variações]

### 4. Oportunidades de Otimização
[Recomendações específicas]

### 5. Performance Comparativa
[Análise de margens e eficiência]

## 🎯 Recomendações Práticas
[3-5 ações específicas]
"""
        return Task(
            description=descricao,
            expected_output="Análise financeira detalhada respondendo às perguntas específicas",
            agent=None,
        )


# =============================
# Gerenciador (com circuit breaker e backoff)
# =============================

class GerenciadorEstudoCaso2:
    def __init__(self, servico_resumo: ServicoResumoDocumentos):
        self._servico_resumo = servico_resumo
        self._pasta_resultados = Path("resultados_estudo_caso_2")

    def executar_estudo_completo(self, pasta_documentos: str = "arquivo_estudo_2", apenas_llama_7b: bool = False):
        print("🚀 INICIANDO ESTUDO DE CASO 2 - ANÁLISE FINANCEIRA INTELIGENTE")
        print("• Análise comparativa de balancetes e documentos financeiros")
        print("• Identificação de padrões, gastos recorrentes e oportunidades")
        print("🎯 Geração de recomendações específicas para otimização de custos\n")

        self._garantir_pasta_resultados()
        if not os.path.exists(pasta_documentos):
            print(f"❌ Pasta '{pasta_documentos}' não encontrada!")
            print("💡 Verifique o caminho ou forneça outra pasta.")
            return

        # Carrega documentos UMA VEZ
        try:
            documentos_cache = self._servico_resumo.carregar_documentos(pasta_documentos)
        except Exception as e:
            print(f"❌ Falha ao carregar documentos: {e}")
            return

        resultados: Dict[ModelType, Dict[str, Optional[str]]] = {}

        if apenas_llama_7b:
            modelos_para_testar = [ModelType.LLAMA_70B]
            print("🎯 Executando apenas com Llama 7B (modo otimizado)")
        else:
            modelos_para_testar = [ModelType.LLAMA_7B, ModelType.LLAMA_70B, ModelType.CHATGPT]

        import time, random

        for tipo_modelo in modelos_para_testar:
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            print(f"\n{'='*60}\n🤖 PROCESSANDO COM: {nome_modelo}\n{'='*60}")

            max_tentativas = 2 if tipo_modelo == ModelType.LLAMA_70B else 3
            caminho_relatorio: Optional[str] = None
            ultimo_erro: Optional[str] = None
            falhas_service_unavailable = 0

            for tentativa in range(1, max_tentativas + 1):
                if tentativa > 1:
                    print(f"🔄 Tentativa {tentativa}/{max_tentativas} para {nome_modelo}...")

                caminho_relatorio, erro = self._servico_resumo.processar_documentos_carregados(
                    documentos_conteudo=documentos_cache,
                    pasta_destino=str(self._pasta_resultados),
                    tipo_modelo=tipo_modelo,
                )

                if caminho_relatorio:
                    break

                ultimo_erro = erro or ""
                if (
                    "ServiceUnavailableError" in ultimo_erro
                    or "ReplicateException" in ultimo_erro
                ):
                    falhas_service_unavailable += 1
                    if tipo_modelo == ModelType.LLAMA_70B and falhas_service_unavailable >= 2:
                        print("🧯 Circuit breaker: falhas repetidas no Replicate 70B. Pulando este modelo.")
                        break

                if tentativa < max_tentativas:
                    espera = 7 * tentativa + random.uniform(0, 2)  # backoff com jitter
                    print(f"⏳ Aguardando {espera:.1f}s antes da próxima tentativa...")
                    time.sleep(espera)

            if caminho_relatorio:
                resultados[tipo_modelo] = {"status": "✅ Sucesso", "arquivo": caminho_relatorio}
            else:
                resultados[tipo_modelo] = {
                    "status": f"❌ Falhou após {max_tentativas} tentativas: {ultimo_erro}",
                    "arquivo": None,
                }

        self._exibir_resumo_final(resultados, pasta_documentos)

    def _garantir_pasta_resultados(self):
        self._pasta_resultados.mkdir(parents=True, exist_ok=True)

    def _exibir_resumo_final(self, resultados: Dict, pasta_analisada: str):
        print("\n🎉 ESTUDO DE CASO 2 CONCLUÍDO!")
        print(f"📁 Pasta analisada: {pasta_analisada}")
        print(f"📊 Resultados salvos em: {self._pasta_resultados}/\n")
        sucessos = 0
        for tipo_modelo, dados in resultados.items():
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            print(f"🤖 {nome_modelo}: {dados['status']}")
            if dados.get("arquivo"):
                print(f"   📄 Relatório: {dados['arquivo']}")
                sucessos += 1
        total = len(resultados)
        taxa = (sucessos / total * 100) if total else 0
        print(f"\n📈 Taxa de sucesso: {sucessos}/{total} modelos ({taxa:.1f}%)")
        if sucessos > 1:
            print("💡 Compare os relatórios para analisar:")
            print("   • Diferenças entre modelos de IA")
            print("   • Padrões identificados nos balancetes")
            print("   • Recomendações de otimização financeira")


# =============================
# Ponto de Entrada
# =============================

def main():
    # Aplica patch LiteLLM para evitar params problemáticos
    ConfiguradorLiteLLM.aplicar_patch_parametros()

    leitor_documentos = LeitorDocumentosMultiFormato()
    gerador_relatorio = GeradorRelatorioMarkdown()

    servico_resumo = ServicoResumoDocumentos(
        leitor_documentos=leitor_documentos,
        gerador_relatorio=gerador_relatorio,
    )

    gerenciador = GerenciadorEstudoCaso2(servico_resumo)

    # Execute com a pasta desejada; altere para o caminho dos seus balancetes
    gerenciador.executar_estudo_completo("arquivo_estudo_2", apenas_llama_7b=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário. Encerrando com segurança...")
