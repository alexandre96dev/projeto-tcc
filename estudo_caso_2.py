"""
Estudo de Caso 2 - Sistema de Análise Financeira Comparativa (VERSÃO MULTI-AGENTE)
- Arquitetura com 3 agentes especializados: Extrator, Analista e Sintetizador
- Cada agente tem responsabilidade específica no pipeline de análise
- Mantém mesmo resultado final com melhor modularidade
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
# Tipos e Configurações (mantidos iguais)
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
    dados_extraidos: str  # Novo: dados estruturados extraídos
    analise_individual: str  # Novo: análise de cada documento
    timestamp: datetime


@dataclass
class RelatorioConsolidado:
    resumo_executivo: str
    tabela_comparativa: str
    analise_detalhada: str
    recomendacoes: str
    documentos_processados: List[DocumentoProcessado]
    modelo_usado: str
    pasta_analisada: str
    timestamp: datetime


# =============================
# Classes auxiliares mantidas iguais
# =============================

def _periodo_de_nome(nome: str) -> str:
    """Extrai algo como "Agosto/2025" do padrão Balancete_Agosto_2025.*"""
    m = re.search(r"Balancete_(\w+)_(\d{4})", nome, re.IGNORECASE)
    if m:
        mes, ano = m.group(1), m.group(2)
        return f"{mes}/{ano}"
    return nome


class LeitorDocumentosMultiFormato:
    """Mantido igual - responsável apenas pela extração de texto"""
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


class FabricaModelos:
    """Mantido igual"""
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
            model_name="openai/gpt-4o-mini",
            display_name="ChatGPT 4o Mini",
            api_key=os.getenv('OPENAI_API_KEY', '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')
        )
    }

    @classmethod
    def criar_modelo(cls, tipo_modelo: ModelType) -> LLM:
        config = cls.CONFIGURACOES_PADRAO[tipo_modelo]
        if not config.api_key:
            raise RuntimeError(
                f"API key não configurada para {config.display_name}. Defina a variável de ambiente correta."
            )
        if tipo_modelo in [ModelType.CHATGPT]:
            return LLM(
                model=config.model_name,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        else:
            return LLM(
                model=config.model_name,
                api_key=config.api_key,
                drop_params=["stop", "stop_sequences", "stops", "stop_tokens", "drop_params"],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                request_timeout=90,
            )

    @classmethod
    def obter_nome_exibicao(cls, tipo_modelo: ModelType) -> str:
        return cls.CONFIGURACOES_PADRAO[tipo_modelo].display_name


class ConfiguradorLiteLLM:
    """Mantido igual"""
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
# SISTEMA MULTI-AGENTE (NOVO)
# =============================

class FabricaAgentesFinanceiros:
    """Factory para criar os 3 agentes especializados"""
    
    @staticmethod
    def criar_agente_extrator(modelo: LLM) -> Agent:
        """Agente 1: Especialista em extração de dados financeiros"""
        return Agent(
            role="Extrator de Dados Financeiros",
            goal=(
                "Extrair e estruturar dados financeiros (receitas, despesas, resultados) "
                "de balancetes e documentos contábeis de forma precisa e organizada."
            ),
            backstory=(
                "Especialista em leitura e interpretação de demonstrativos financeiros. "
                "Experiência em identificar valores, categorias de gastos e estruturar dados "
                "para análise posterior. Foco em precisão e detalhamento."
            ),
            verbose=False,
            llm=modelo,
            max_iter=2,
            max_execution_time=300,
        )
    
    @staticmethod
    def criar_agente_analista(modelo: LLM) -> Agent:
        """Agente 2: Especialista em análise comparativa e padrões"""
        return Agent(
            role="Analista Financeiro Comparativo",
            goal=(
                "Analisar dados financeiros estruturados identificando padrões, tendências, "
                "correlações e oportunidades de otimização entre diferentes períodos."
            ),
            backstory=(
                "Analista sênior com experiência em análise comparativa temporal. "
                "Especialista em identificar gastos recorrentes, variações sazonais, "
                "eficiência operacional e pontos de otimização financeira."
            ),
            verbose=False,
            llm=modelo,
            max_iter=2,
            max_execution_time=300,
        )
    
    @staticmethod
    def criar_agente_sintetizador(modelo: LLM) -> Agent:
        """Agente 3: Especialista em síntese e recomendações executivas"""
        return Agent(
            role="Sintetizador Executivo",
            goal=(
                "Consolidar análises financeiras em relatórios executivos claros, "
                "com recomendações práticas e insights acionáveis para gestão."
            ),
            backstory=(
                "Consultor executivo especializado em traduzir análises técnicas "
                "em insights estratégicos. Experiência em elaboração de relatórios "
                "para alta gestão com foco em ações práticas e resultados."
            ),
            verbose=False,
            llm=modelo,
            max_iter=2,
            max_execution_time=300,
        )


class FabricaTarefasFinanceiras:
    """Factory para criar as 3 tarefas especializadas"""
    
    @staticmethod
    def criar_tarefa_extracao(documentos_conteudo: List[Dict], agente_extrator: Agent) -> Task:
        """Tarefa 1: Extrair dados estruturados de todos os documentos"""
        
        documentos_texto = f"EXTRAÇÃO DE DADOS DE {len(documentos_conteudo)} BALANCETES:\n\n"
        for i, doc in enumerate(documentos_conteudo, 1):
            conteudo = doc["conteudo"]
            conteudo_limitado = conteudo[:6000] + "...[TRUNCADO]" if len(conteudo) > 6000 else conteudo
            periodo = _periodo_de_nome(doc["nome"])
            documentos_texto += f"""
DOCUMENTO {i}: {doc['nome']}
PERÍODO: {periodo}
CONTEÚDO:
{conteudo_limitado}

{'='*50}
"""
        
        return Task(
            description=f"""
EXTRAIA dados financeiros estruturados de CADA um dos {len(documentos_conteudo)} balancetes.

DOCUMENTOS:
{documentos_texto}

INSTRUÇÕES ESPECÍFICAS:
1. Para CADA documento, extraia:
   - Receitas totais (valor numérico)
   - Despesas totais (valor numérico)
   - Resultado líquido (receitas - despesas)
   - Top 5 maiores gastos com valores
   - Período/mês de referência

2. FORMATO DE SAÍDA OBRIGATÓRIO:
```json
{{
  "documentos_extraidos": [
    {{
      "periodo": "Mês/Ano",
      "receitas_total": 0.00,
      "despesas_total": 0.00,
      "resultado": 0.00,
      "margem_percent": 0.00,
      "top_gastos": [
        {{"categoria": "Nome", "valor": 0.00}},
        {{"categoria": "Nome", "valor": 0.00}}
      ]
    }}
  ]
}}
```

3. VALIDAÇÃO: Certifique-se que receitas - despesas = resultado para cada período.
4. NÃO INVENTE dados que não estão nos documentos.
""",
            expected_output="JSON estruturado com dados financeiros extraídos de todos os documentos",
            agent=agente_extrator,
        )
    
    @staticmethod
    def criar_tarefa_analise(dados_extraidos: str, agente_analista: Agent) -> Task:
        """Tarefa 2: Analisar padrões e tendências nos dados estruturados"""
        
        return Task(
            description=f"""
ANALISE os dados financeiros estruturados e identifique padrões, tendências e oportunidades.

DADOS EXTRAÍDOS PARA ANÁLISE:
{dados_extraidos}

ANÁLISES OBRIGATÓRIAS:

1. **EVOLUÇÃO TEMPORAL**:
   - Compare receitas, despesas e resultados entre períodos
   - Identifique melhor e pior período (margem %)
   - Calcule variação percentual entre períodos

2. **PADRÕES DE GASTOS**:
   - Gastos que aparecem em todos os períodos (recorrentes)
   - Categorias com maior variação entre períodos
   - Ranking dos 5 maiores gastos consolidados

3. **TENDÊNCIAS E CORRELAÇÕES**:
   - Tendência geral (crescimento/declínio/estabilidade)
   - Sazonalidade ou padrões específicos
   - Correlação entre receitas e tipos de gastos

4. **OPORTUNIDADES DE OTIMIZAÇÃO**:
   - Gastos com maior potencial de redução
   - Categorias com crescimento desproporcional
   - Ineficiências operacionais identificadas

FORMATO DE SAÍDA:
Análise estruturada em texto claro, destacando insights quantitativos e qualitativos.
NÃO repita os dados brutos - ANALISE e INTERPRETE.
""",
            expected_output="Análise detalhada com insights, padrões e oportunidades identificadas",
            agent=agente_analista,
        )
    
    @staticmethod
    def criar_tarefa_sintese(dados_extraidos: str, analise_realizada: str, agente_sintetizador: Agent) -> Task:
        """Tarefa 3: Consolidar em relatório executivo final"""
        
        return Task(
            description=f"""
CONSOLIDE os dados e análises em um relatório executivo final para gestão.

DADOS FINANCEIROS:
{dados_extraidos}

ANÁLISE REALIZADA:
{analise_realizada}

RELATÓRIO FINAL OBRIGATÓRIO - FORMATO MARKDOWN:

# Análise Financeira Comparativa - Estudo de Caso 2

## 📊 Resumo Executivo
[Síntese das principais descobertas em 3-4 frases]

## 📋 Dados Consolidados
| Período | Receitas | Despesas | Resultado | Margem % |
|---------|----------|----------|-----------|----------|
[Tabela com TODOS os períodos analisados]

## 💰 Principais Insights

### 1. Performance por Período
- **Melhor período**: [Qual e por quê]
- **Pior período**: [Qual e por quê]
- **Variação geral**: [Tendência observada]

### 2. Padrões de Gastos
- **Top 5 gastos recorrentes**: [Lista consolidada]
- **Maior variação**: [Categoria com maior oscilação]
- **Gastos em crescimento**: [Tendências preocupantes]

### 3. Oportunidades Identificadas
- **Otimização imediata**: [1-2 ações de curto prazo]
- **Eficiência operacional**: [Melhorias de processo]
- **Controle de custos**: [Categorias prioritárias]

## 🎯 Recomendações Executivas
1. [Ação específica com resultado esperado]
2. [Ação específica com resultado esperado]
3. [Ação específica com resultado esperado]

## 📈 Próximos Passos
[2-3 ações para implementação]

---
*Relatório baseado na análise de [X] períodos financeiros*
""",
            expected_output="Relatório executivo completo em formato Markdown",
            agent=agente_sintetizador,
        )


# =============================
# SERVIÇO PRINCIPAL MULTI-AGENTE
# =============================

class ServicoResumoDocumentosMultiAgente:
    """Serviço principal com arquitetura multi-agente"""
    
    def __init__(self, leitor_documentos: LeitorDocumentosMultiFormato):
        self._leitor_documentos = leitor_documentos

    def carregar_documentos(self, pasta_origem: str) -> List[Dict]:
        """Mantido igual - carrega documentos uma vez"""
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

    def processar_documentos_multiagente(
        self,
        documentos_conteudo: List[Dict],
        pasta_destino: str,
        tipo_modelo: ModelType,
    ) -> Tuple[Optional[str], Optional[str]]:
        """NOVO: Processamento com 3 agentes especializados"""
        
        nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
        
        try:
            print(f"🤖 Criando agentes especializados para {nome_modelo}...")
            modelo = FabricaModelos.criar_modelo(tipo_modelo)
            
            # Criar os 3 agentes especializados
            agente_extrator = FabricaAgentesFinanceiros.criar_agente_extrator(modelo)
            agente_analista = FabricaAgentesFinanceiros.criar_agente_analista(modelo)
            agente_sintetizador = FabricaAgentesFinanceiros.criar_agente_sintetizador(modelo)
            
            print("📊 ETAPA 1: Extraindo dados estruturados...")
            
            # Tarefa 1: Extração de dados
            tarefa_extracao = FabricaTarefasFinanceiras.criar_tarefa_extracao(
                documentos_conteudo, agente_extrator
            )
            
            crew_extracao = Crew(
                agents=[agente_extrator],
                tasks=[tarefa_extracao],
                process=Process.sequential,
                verbose=False,
                max_execution_time=600,
            )
            
            dados_extraidos = str(crew_extracao.kickoff()).strip()
            
            if not dados_extraidos:
                return None, "Falha na extração de dados estruturados"
            
            print("🔍 ETAPA 2: Analisando padrões e tendências...")
            
            # Tarefa 2: Análise de padrões
            tarefa_analise = FabricaTarefasFinanceiras.criar_tarefa_analise(
                dados_extraidos, agente_analista
            )
            
            crew_analise = Crew(
                agents=[agente_analista],
                tasks=[tarefa_analise],
                process=Process.sequential,
                verbose=False,
                max_execution_time=600,
            )
            
            analise_realizada = str(crew_analise.kickoff()).strip()
            
            if not analise_realizada:
                return None, "Falha na análise de padrões"
            
            print("📝 ETAPA 3: Consolidando relatório executivo...")
            
            # Tarefa 3: Síntese final
            tarefa_sintese = FabricaTarefasFinanceiras.criar_tarefa_sintese(
                dados_extraidos, analise_realizada, agente_sintetizador
            )
            
            crew_sintese = Crew(
                agents=[agente_sintetizador],
                tasks=[tarefa_sintese],
                process=Process.sequential,
                verbose=False,
                max_execution_time=600,
            )
            
            relatorio_final = str(crew_sintese.kickoff()).strip()
            
            if not relatorio_final:
                return None, "Falha na consolidação do relatório final"
            
            # Salvar relatório final
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo_relatorio = (
                f"analise_financeira_{nome_modelo.lower().replace(' ', '_')}_{timestamp}.md"
            )
            Path(pasta_destino).mkdir(parents=True, exist_ok=True)
            caminho_relatorio = Path(pasta_destino) / nome_arquivo_relatorio
            
            with open(caminho_relatorio, "w", encoding="utf-8") as f:
                f.write(relatorio_final)
            
            if not caminho_relatorio.exists() or caminho_relatorio.stat().st_size == 0:
                return None, "Falha ao criar arquivo de relatório"
            
            print(f"✅ {nome_modelo}: Relatório multi-agente salvo em {caminho_relatorio}")
            print(f"📊 Tamanho do relatório: {caminho_relatorio.stat().st_size} bytes")
            print(f"🤖 Processamento realizado por 3 agentes especializados:")
            print(f"   • Extrator de Dados → Analista Comparativo → Sintetizador Executivo")
            
            return str(caminho_relatorio), None
            
        except Exception as e:
            erro = f"Erro no processamento multi-agente com {nome_modelo}: {e}"
            print(f"❌ {erro}")
            return None, erro


# =============================
# GERENCIADOR (atualizado para multi-agente)
# =============================

class GerenciadorEstudoCaso2MultiAgente:
    """Gerenciador adaptado para arquitetura multi-agente"""
    
    def __init__(self, servico_resumo: ServicoResumoDocumentosMultiAgente):
        self._servico_resumo = servico_resumo
        self._pasta_resultados = Path("resultados_estudo_caso_2")

    def executar_estudo_completo(self, pasta_documentos: str = "arquivo_estudo_2", apenas_llama_7b: bool = False):
        print("🚀 ESTUDO DE CASO 2 - ANÁLISE FINANCEIRA MULTI-AGENTE")
        print("🤖 Arquitetura: 3 Agentes Especializados")
        print("   • Agente 1: Extrator de Dados Financeiros")
        print("   • Agente 2: Analista Comparativo") 
        print("   • Agente 3: Sintetizador Executivo")
        print("🎯 Pipeline: Extração → Análise → Síntese\n")

        self._garantir_pasta_resultados()
        if not os.path.exists(pasta_documentos):
            print(f"❌ Pasta '{pasta_documentos}' não encontrada!")
            return

        # Carregar documentos uma vez
        try:
            documentos_cache = self._servico_resumo.carregar_documentos(pasta_documentos)
        except Exception as e:
            print(f"❌ Falha ao carregar documentos: {e}")
            return

        resultados: Dict[ModelType, Dict[str, Optional[str]]] = {}

        if apenas_llama_7b:
            modelos_para_testar = [ModelType.LLAMA_7B]
            print("🎯 Executando apenas com Llama 7B (modo otimizado)")
        else:
            modelos_para_testar = [ModelType.LLAMA_7B, ModelType.LLAMA_70B, ModelType.CHATGPT]

        import time, random

        for tipo_modelo in modelos_para_testar:
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            print(f"\n{'='*60}\n🤖 PROCESSAMENTO MULTI-AGENTE: {nome_modelo}\n{'='*60}")

            max_tentativas = 2 if tipo_modelo == ModelType.LLAMA_70B else 3
            caminho_relatorio: Optional[str] = None
            ultimo_erro: Optional[str] = None

            for tentativa in range(1, max_tentativas + 1):
                if tentativa > 1:
                    print(f"🔄 Tentativa {tentativa}/{max_tentativas} para {nome_modelo}...")

                # Usar método multi-agente
                caminho_relatorio, erro = self._servico_resumo.processar_documentos_multiagente(
                    documentos_conteudo=documentos_cache,
                    pasta_destino=str(self._pasta_resultados),
                    tipo_modelo=tipo_modelo,
                )

                if caminho_relatorio:
                    break

                ultimo_erro = erro or ""
                
                if tentativa < max_tentativas:
                    espera = 7 * tentativa + random.uniform(0, 2)
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
        print("\n🎉 ESTUDO DE CASO 2 MULTI-AGENTE CONCLUÍDO!")
        print(f"📁 Pasta analisada: {pasta_analisada}")
        print(f"📊 Resultados salvos em: {self._pasta_resultados}/")
        print(f"🤖 Arquitetura: 3 agentes especializados por modelo\n")
        
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
            print("💡 Compare os relatórios multi-agente para analisar:")
            print("   • Especialização vs. abordagem generalista")
            print("   • Qualidade da extração de dados estruturados")
            print("   • Profundidade das análises comparativas")
            print("   • Clareza das recomendações executivas")


# =============================
# PONTO DE ENTRADA
# =============================

def main():
    """Ponto de entrada para versão multi-agente"""
    ConfiguradorLiteLLM.aplicar_patch_parametros()

    leitor_documentos = LeitorDocumentosMultiFormato()
    
    # Usar serviço multi-agente
    servico_resumo = ServicoResumoDocumentosMultiAgente(
        leitor_documentos=leitor_documentos
    )

    gerenciador = GerenciadorEstudoCaso2MultiAgente(servico_resumo)
    
    # Executar com arquitetura multi-agente
    gerenciador.executar_estudo_completo("arquivo_estudo_2", apenas_llama_7b=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário. Encerrando com segurança...")