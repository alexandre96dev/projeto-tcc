"""
Estudo de Caso 2 - Sistema de Análise Financeira Comparativa
Sistema limpo para processamento de balancetes com 3 modelos de IA independentes
Cada modelo executa uma única vez sem fallbacks ou modificações
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple
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
    max_tokens: int = 4096
    max_retries: int = 500


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


class LeitorDocumentos(Protocol):
    
    def listar_arquivos_suportados(self, pasta: str) -> List[str]:
        ...
    
    def extrair_conteudo(self, caminho_arquivo: str) -> str:
        ...


class ProcessadorResumos(Protocol):
    
    def gerar_resumo_documento(self, conteudo: str, nome_arquivo: str) -> str:
        ...
    
    def gerar_relatorio_consolidado(self, documentos: List[DocumentoProcessado]) -> RelatorioConsolidado:
        ...


class GeradorRelatorio(Protocol):
    
    def salvar_relatorio_markdown(self, relatorio: RelatorioConsolidado, pasta_destino: str) -> str:
        ...


class LeitorDocumentosMultiFormato:
    
    EXTENSOES_SUPORTADAS = {'.pdf', '.docx', '.txt', '.md'}
    
    def listar_arquivos_suportados(self, pasta: str) -> List[str]:
        if not os.path.exists(pasta):
            raise FileNotFoundError(f"Pasta '{pasta}' não encontrada")
        
        arquivos_encontrados = []
        pasta_path = Path(pasta)
        
        for arquivo in pasta_path.iterdir():
            if arquivo.is_file() and arquivo.suffix.lower() in self.EXTENSOES_SUPORTADAS:
                arquivos_encontrados.append(str(arquivo))
        
        if len(arquivos_encontrados) < 1:
            raise ValueError(f"Encontrados apenas {len(arquivos_encontrados)} arquivos suportados. Mínimo: 1")
        
        if len(arquivos_encontrados) > 1:
            balancetes = [arq for arq in arquivos_encontrados if 'balancete' in Path(arq).name.lower()]
            if balancetes:
                print(f"📊 Priorizando {len(balancetes)} balancetes encontrados para análise comparativa")
                return sorted(balancetes)
        
        return sorted(arquivos_encontrados)
    
    def extrair_conteudo(self, caminho_arquivo: str) -> str:
        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo '{caminho_arquivo}' não encontrado")
        
        extensao = Path(caminho_arquivo).suffix.lower()
        
        try:
            if extensao == '.pdf':
                return self._extrair_pdf(caminho_arquivo)
            elif extensao == '.docx':
                return self._extrair_docx(caminho_arquivo)
            elif extensao in ['.txt', '.md']:
                return self._extrair_texto(caminho_arquivo)
            else:
                raise ValueError(f"Formato não suportado: {extensao}")
        except Exception as e:
            raise RuntimeError(f"Erro ao extrair conteúdo de '{caminho_arquivo}': {str(e)}")
    
    def _extrair_pdf(self, caminho: str) -> str:
        with fitz.open(caminho) as doc:
            texto = ""
            for pagina in doc:
                texto += pagina.get_text() + "\n"
        
        if not texto.strip():
            raise ValueError(f"PDF '{caminho}' está vazio ou não contém texto extraível")
        
        return texto.strip()
    
    def _extrair_docx(self, caminho: str) -> str:
        doc = Document(caminho)
        paragrafos = [p.text for p in doc.paragraphs if p.text.strip()]
        
        if not paragrafos:
            raise ValueError(f"DOCX '{caminho}' está vazio ou não contém texto extraível")
        
        return "\n".join(paragrafos)
    
    def _extrair_texto(self, caminho: str) -> str:
        with open(caminho, 'r', encoding='utf-8') as f:
            conteudo = f.read().strip()
        
        if not conteudo:
            raise ValueError(f"Arquivo de texto '{caminho}' está vazio")
        
        return conteudo


class ProcessadorResumosIA:
    
    def __init__(self, modelo: LLM, nome_modelo: str):
        self._modelo = modelo
        self._nome_modelo = nome_modelo
    
    def gerar_resumo_documento(self, conteudo: str, nome_arquivo: str) -> str:
        agente_resumo = self._criar_agente_resumo()
        tarefa_resumo = self._criar_tarefa_resumo(conteudo, nome_arquivo)
        
        crew = Crew(
            agents=[agente_resumo],
            tasks=[tarefa_resumo],
            process=Process.sequential,
            verbose=False,
            max_execution_time=300
        )
        
        resultado = crew.kickoff()
        return str(resultado).strip()
    
    def gerar_relatorio_consolidado(self, documentos: List[DocumentoProcessado]) -> RelatorioConsolidado:
        """Gera relatório consolidado usando IA"""
        agente_relatorio = self._criar_agente_relatorio()
        tarefa_relatorio = self._criar_tarefa_relatorio(documentos)
        
        crew = Crew(
            agents=[agente_relatorio],
            tasks=[tarefa_relatorio],
            process=Process.sequential,
            verbose=False,
            max_execution_time=600
        )
        
        resultado = crew.kickoff()
        
        texto_resultado = str(resultado)
        secoes = self._extrair_secoes_relatorio(texto_resultado)
        
        return RelatorioConsolidado(
            introducao=secoes.get("introducao", "Não foi possível gerar introdução"),
            evolucao_padroes=secoes.get("evolucao_padroes", "Não foi possível analisar padrões"),
            conclusao=secoes.get("conclusao", "Não foi possível gerar conclusão"),
            documentos_processados=documentos,
            modelo_usado=self._nome_modelo,
            pasta_analisada=os.path.dirname(documentos[0].caminho_completo) if documentos else "",
            timestamp=datetime.now()
        )
    
    def _criar_agente_resumo(self) -> Agent:
        """Cria agente especializado em resumos"""
        return Agent(
            role="Especialista em Resumo de Documentos",
            goal="Criar resumos concisos e informativos de documentos, destacando os pontos principais",
            backstory="""
            Você é um especialista em análise e síntese de documentos com experiência em:
            - Identificação de informações principais
            - Criação de resumos estruturados
            - Extração de pontos-chave
            - Manutenção da essência do documento original
            """,
            verbose=False,
            llm=self._modelo,
            max_iter=2,
            max_execution_time=300
        )
    
    def _criar_agente_relatorio(self) -> Agent:
        """Cria agente especializado em relatórios consolidados"""
        return Agent(
            role="Analista de Documentos e Redator de Relatórios",
            goal="Analisar múltiplos documentos e criar relatórios consolidados com análise de padrões",
            backstory="""
            Você é um analista experiente em:
            - Análise comparativa de múltiplos documentos
            - Identificação de padrões e tendências
            - Criação de relatórios executivos
            - Síntese de informações complexas
            """,
            verbose=False,
            llm=self._modelo,
            max_iter=2,
            max_execution_time=600
        )
    
    def _criar_tarefa_resumo(self, conteudo: str, nome_arquivo: str) -> Task:
        """Cria tarefa de resumo para um documento"""
        conteudo_limitado = conteudo[:15000] + "...[TRUNCADO]" if len(conteudo) > 15000 else conteudo
        
        return Task(
            description=f"""
            Analise o seguinte documento '{nome_arquivo}' e crie um resumo estruturado.

            CONTEÚDO DO DOCUMENTO:
            {conteudo_limitado}

            INSTRUÇÕES:
            1. Crie um resumo de 200-400 palavras
            2. Identifique os pontos principais
            3. Mantenha a estrutura lógica do documento
            4. Use linguagem clara e objetiva
            5. Destaque informações importantes

            FORMATO DA RESPOSTA:
            Retorne apenas o resumo, sem títulos ou marcações adicionais.
            """,
            expected_output="Resumo estruturado e conciso do documento",
            agent=self._criar_agente_resumo()
        )
    
    def _criar_tarefa_relatorio(self, documentos: List[DocumentoProcessado]) -> Task:
        """Cria tarefa de relatório consolidado"""
        resumos_texto = "\n\n".join([
            f"ARQUIVO: {doc.nome_arquivo}\nRESUMO: {doc.resumo_gerado}"
            for doc in documentos
        ])
        
        return Task(
            description=f"""
            Com base nos resumos dos documentos abaixo, crie um relatório consolidado estruturado.

            RESUMOS DOS DOCUMENTOS:
            {resumos_texto}

            INSTRUÇÕES:
            1. Analise todos os resumos em conjunto
            2. Identifique padrões, temas comuns e diferenças
            3. Crie um relatório com EXATAMENTE estas 3 seções:

            SEÇÃO 1 - INTRODUÇÃO:
            - Visão geral dos documentos analisados
            - Contexto e propósito da análise
            - Resumo dos principais temas encontrados

            SEÇÃO 2 - EVOLUÇÃO E PADRÕES:
            - Análise comparativa entre os documentos
            - Identificação de padrões e tendências
            - Evolução temporal (se aplicável)
            - Pontos de convergência e divergência

            SEÇÃO 3 - CONCLUSÃO:
            - Síntese dos principais achados
            - Implicações dos padrões identificados
            - Recomendações ou insights finais

            FORMATO DA RESPOSTA:
            Use EXATAMENTE estes marcadores:
            [INTRODUÇÃO]
            (texto da introdução)

            [EVOLUÇÃO_PADRÕES]
            (texto da análise de evolução e padrões)

            [CONCLUSÃO]
            (texto da conclusão)
            """,
            expected_output="Relatório consolidado com as três seções estruturadas",
            agent=self._criar_agente_relatorio()
        )
    
    def _extrair_secoes_relatorio(self, texto: str) -> Dict[str, str]:
        """Extrai as seções do relatório usando marcadores"""
        secoes = {}
        
        padroes = {
            "introducao": r"\[INTRODUÇÃO\](.*?)(?=\[|$)",
            "evolucao_padroes": r"\[EVOLUÇÃO_PADRÕES\](.*?)(?=\[|$)",
            "conclusao": r"\[CONCLUSÃO\](.*?)(?=\[|$)"
        }
        
        for nome_secao, padrao in padroes.items():
            match = re.search(padrao, texto, re.DOTALL | re.IGNORECASE)
            if match:
                secoes[nome_secao] = match.group(1).strip()
            else:
                secoes[nome_secao] = self._extrair_secao_fallback(texto, nome_secao)
        
        return secoes
    
    def _extrair_secao_fallback(self, texto: str, nome_secao: str) -> str:
        """Extração de fallback quando marcadores não funcionam"""
        linhas = texto.split('\n')
        palavras_chave = {
            "introducao": ["introdução", "visão geral", "contexto", "propósito"],
            "evolucao_padroes": ["evolução", "padrões", "análise", "comparativa", "tendências"],
            "conclusao": ["conclusão", "síntese", "achados", "recomendações"]
        }
        
        secao_linhas = []
        capturando = False
        
        for linha in linhas:
            linha_lower = linha.lower()
            
            if any(palavra in linha_lower for palavra in palavras_chave[nome_secao]):
                capturando = True
                if linha.strip(): 
                    continue
            
            elif capturando and any(
                palavra in linha_lower 
                for outras_secoes in palavras_chave.values() 
                if outras_secoes != palavras_chave[nome_secao]
                for palavra in outras_secoes
            ):
                break
            
            if capturando and linha.strip():
                secao_linhas.append(linha.strip())
        
        return '\n'.join(secao_linhas) if secao_linhas else f"Seção {nome_secao} não encontrada"


class GeradorRelatorioMarkdown:
    """Gerador de relatórios em formato Markdown"""
    
    def salvar_relatorio_markdown(self, relatorio: RelatorioConsolidado, pasta_destino: str) -> str:
        """Salva relatório consolidado em Markdown"""
        Path(pasta_destino).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"relatorio_consolidado_{relatorio.modelo_usado.lower()}_{timestamp}.md"
        caminho_arquivo = Path(pasta_destino) / nome_arquivo
        
        conteudo = self._gerar_conteudo_markdown(relatorio)
        
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        
        return str(caminho_arquivo)
    
    def _gerar_conteudo_markdown(self, relatorio: RelatorioConsolidado) -> str:
        """Gera o conteúdo do relatório em Markdown"""
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
            ""
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
                ""
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
            f"*Relatório gerado automaticamente pelo sistema de análise - {relatorio.timestamp.strftime('%d/%m/%Y %H:%M:%S')}*"
        ])
        
        return "\n".join(linhas)


class FabricaModelos:
    """Factory para criação de modelos de IA com configurações equalizadas"""
    
    CONFIGURACOES_PADRAO = {
        ModelType.LLAMA_7B: ConfiguracaoModelo(
            model_name="replicate/meta/meta-llama-3-8b-instruct",
            display_name="Llama 7B",
            api_key=os.getenv('REPLICATE_API_TOKEN', 'r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn'),
            temperature=0.1,
            max_tokens=16384,
            max_retries=300
        ),
        ModelType.LLAMA_70B: ConfiguracaoModelo(
            model_name="replicate/meta/meta-llama-3-70b-instruct",
            display_name="Llama 70B",
            api_key=os.getenv('REPLICATE_API_TOKEN', 'r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn'),
            temperature=0.1,  
            max_tokens=16384, 
            max_retries=300   
        ),
        ModelType.CHATGPT: ConfiguracaoModelo(
            model_name="replicate/openai/gpt-4o-mini",
            display_name="ChatGPT 4.0",
            api_key=os.getenv('OPENAI_API_KEY', 'r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn'),
            temperature=0.1,
            max_tokens=16384,
            max_retries=300
        )
    }
    
    @classmethod
    def criar_modelo(cls, tipo_modelo: ModelType) -> LLM:
        """Cria instância de modelo baseada no tipo com configurações equalizadas"""
        config = cls.CONFIGURACOES_PADRAO[tipo_modelo]
        
        return LLM(
            model=config.model_name,
            api_key=config.api_key,
            drop_params=["stop", "stop_sequences", "stops", "stop_tokens"],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_timeout=120,
        )
    
    @classmethod
    def obter_nome_exibicao(cls, tipo_modelo: ModelType) -> str:
        """Obtém nome de exibição do modelo"""
        return cls.CONFIGURACOES_PADRAO[tipo_modelo].display_name


class ConfiguradorLiteLLM:
    """Configurador para LiteLLM - Reutilizado do Estudo de Caso 1"""
    
    @staticmethod
    def aplicar_patch_parametros():
        """Aplica patch para remover parâmetros problemáticos"""
        completion_original = litellm.completion
        
        def completion_sem_parametros_problematicos(**parametros):
            parametros_problematicos = ["stop", "stop_sequences", "stops", "stop_tokens"]
            
            for parametro in parametros_problematicos:
                parametros.pop(parametro, None)
            
            if "extra_body" in parametros and isinstance(parametros["extra_body"], dict):
                for parametro in parametros_problematicos:
                    parametros["extra_body"].pop(parametro, None)
            
            return completion_original(**parametros)
        
        litellm.completion = completion_sem_parametros_problematicos


# =============================================================================
# SERVIÇO PRINCIPAL
# =============================================================================

class ServicoResumoDocumentos:
    """Serviço principal para processamento em lote de documentos"""
    
    def __init__(
        self,
        leitor_documentos: LeitorDocumentos,
        gerador_relatorio: GeradorRelatorio
    ):
        self._leitor_documentos = leitor_documentos
        self._gerador_relatorio = gerador_relatorio
    
    def processar_pasta_documentos(
        self,
        pasta_origem: str,
        pasta_destino: str,
        tipo_modelo: ModelType
    ) -> Tuple[Optional[str], Optional[str]]:
        """Processa todos os documentos usando apenas um agente para tudo"""
        nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
        
        try:
            modelo = FabricaModelos.criar_modelo(tipo_modelo)
            
            print(f"🔄 Processando com {nome_modelo}...")
            
            arquivos = self._leitor_documentos.listar_arquivos_suportados(pasta_origem)
            print(f"📁 Encontrados {len(arquivos)} arquivos para processar")
            
            documentos_conteudo = []
            for i, caminho_arquivo in enumerate(arquivos, 1):
                nome_arquivo = Path(caminho_arquivo).name
                print(f"📄 Extraindo conteúdo ({i}/{len(arquivos)}): {nome_arquivo}")
                
                try:
                    conteudo = self._leitor_documentos.extrair_conteudo(caminho_arquivo)
                    documentos_conteudo.append({
                        'nome': nome_arquivo,
                        'caminho': caminho_arquivo,
                        'conteudo': conteudo
                    })
                    print(f"✅ {nome_arquivo}: Conteúdo extraído")
                    
                except Exception as e:
                    print(f"❌ Erro ao extrair {nome_arquivo}: {str(e)}")
                    continue
            
            if not documentos_conteudo:
                return None, "Nenhum documento foi extraído com sucesso"
            
            print("🤖 Executando análise completa com agente...")
            agente = self._criar_agente_completo(modelo)
            tarefa = self._criar_tarefa_completa(documentos_conteudo)
            tarefa.agent = agente 
            
            # Timeout equalizado para todos os modelos
            timeout_padrao = 1800  # 30 minutos para todos
            
            crew = Crew(
                agents=[agente],
                tasks=[tarefa],
                process=Process.sequential,
                verbose=False,
                max_execution_time=timeout_padrao
            )
            
            print("🤖 Executando análise financeira...")
            resultado = crew.kickoff()
            
            # Salvar relatório diretamente
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo_relatorio = f"analise_financeira_{nome_modelo.lower().replace(' ', '_')}_{timestamp}.md"
            caminho_relatorio = Path(pasta_destino) / nome_arquivo_relatorio
            
            # Garantir que pasta existe
            Path(pasta_destino).mkdir(parents=True, exist_ok=True)
            
            # Verificar se resultado não está vazio
            resultado_texto = str(resultado).strip()
            if not resultado_texto:
                return None, "Resultado da análise está vazio"
            
            # Salvar com tratamento de erro robusto
            try:
                with open(caminho_relatorio, 'w', encoding='utf-8') as f:
                    f.write(resultado_texto)
                
                # Verificar se arquivo foi realmente criado
                if not caminho_relatorio.exists() or caminho_relatorio.stat().st_size == 0:
                    return None, "Falha ao criar arquivo de relatório"
                
                print(f"✅ {nome_modelo}: Relatório salvo em {caminho_relatorio}")
                print(f"📊 Tamanho do relatório: {caminho_relatorio.stat().st_size} bytes")
                return str(caminho_relatorio), None
                
            except Exception as e_save:
                return None, f"Erro ao salvar relatório: {str(e_save)}"
            
        except Exception as e:
            erro = f"Erro ao processar com {nome_modelo}: {str(e)}"
            print(f"❌ {erro}")
            return None, erro
    

    

    

    
    def _criar_agente_completo(self, modelo: LLM) -> Agent:
        """Cria agente especializado em análise financeira comparativa"""
        return Agent(
            role="Analista Financeiro Especializado",
            goal="Analisar balancetes financeiros identificando padrões, tendências e oportunidades de otimização de custos",
            backstory="""
            Você é um analista financeiro experiente especializado em:
            - Análise comparativa de balancetes e demonstrativos financeiros
            - Identificação de padrões de gastos e receitas ao longo do tempo
            - Detecção de anomalias e oportunidades de redução de custos
            - Análise de eficiência operacional e financeira
            - Geração de recomendações práticas baseadas em dados reais
            - Trabalha EXCLUSIVAMENTE com dados presentes nos documentos
            - NUNCA inventa números ou informações não documentadas
            """,
            verbose=False,
            llm=modelo,
        )
    
    def _criar_tarefa_completa(self, documentos_conteudo: List[Dict]) -> Task:
        """Cria tarefa para análise financeira específica e comparativa"""
        
        # Preparar conteúdo de TODOS os documentos para análise comparativa
        documentos_texto = f"ANÁLISE COMPARATIVA DE {len(documentos_conteudo)} BALANCETES:\n\n"
        
        for i, doc in enumerate(documentos_conteudo, 1):
            # Para análise comparativa, incluir mais conteúdo de cada documento
            conteudo_limitado = doc['conteudo'][:8000] + "...[TRUNCADO]" if len(doc['conteudo']) > 8000 else doc['conteudo']
            documentos_texto += f"""
DOCUMENTO {i}: {doc['nome']}
PERÍODO: {doc['nome'].replace('Balancete_', '').replace('.pdf', '').replace('_', '/')}
CONTEÚDO FINANCEIRO:
{conteudo_limitado}

{'='*60}

"""
        
        # Adicionar resumo dos documentos para facilitar comparação
        documentos_texto += f"\nRESUMO: Total de {len(documentos_conteudo)} balancetes para análise comparativa temporal.\n"
        
        return Task(
            description=f"""
            Analise TODOS os {len(documentos_conteudo)} balancetes e faça análise comparativa completa.

            DOCUMENTOS:
            {documentos_texto}

            ANÁLISE OBRIGATÓRIA - 5 SEÇÕES:

            1. **MAIORES GASTOS**: Liste top 5 gastos de cada período e compare evolução.

            2. **PADRÕES**: Identifique gastos recorrentes e suas tendências (crescimento/redução).

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
            """,
            expected_output="Análise financeira detalhada respondendo às perguntas específicas com recomendações práticas",
            agent=None
        )


class GerenciadorEstudoCaso2:
    """Gerenciador principal do Estudo de Caso 2"""
    
    def __init__(self, servico_resumo: ServicoResumoDocumentos):
        self._servico_resumo = servico_resumo
        self._pasta_resultados = Path("resultados_estudo_caso_2")
    
    def executar_estudo_completo(
        self,
        pasta_documentos: str = "arquivo_estudo_2",
        apenas_llama_7b: bool = False
    ):
        """Executa o estudo completo com todos os modelos"""
        print("🚀 INICIANDO ESTUDO DE CASO 2 - ANÁLISE FINANCEIRA INTELIGENTE")
        print("� Análise comparativa de balancetes e documentos financeiros")
        print("� Identificação de padrões, gastos recorrentes e oportunidades")
        print("🎯 Geração de recomendações específicas para otimização de custos")
        print()
        
        self._garantir_pasta_resultados()
        
        # Verificar se pasta de documentos existe
        if not os.path.exists(pasta_documentos):
            print(f"❌ Pasta '{pasta_documentos}' não encontrada!")
            print("💡 Sugestão: verifique se a pasta existe ou use outro caminho")
            return
        
        resultados = {}
        
        # Processar com cada modelo (priorizando Llama 7B que está estável)
        if apenas_llama_7b:
            modelos_para_testar = [ModelType.LLAMA_7B]
            print("🎯 Executando apenas com Llama 7B (modo otimizado)")
        else:
            modelos_para_testar = [ModelType.LLAMA_7B, ModelType.LLAMA_70B, ModelType.CHATGPT]
        
        for tipo_modelo in modelos_para_testar:
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            print(f"\n{'='*60}")
            print(f"🤖 PROCESSANDO COM: {nome_modelo}")
            print(f"{'='*60}")
            
            # Mecanismo de retry - máximo 3 tentativas com as mesmas configurações
            max_tentativas = 10
            caminho_relatorio = None
            ultimo_erro = None
            
            for tentativa in range(1, max_tentativas + 1):
                if tentativa > 1:
                    print(f"🔄 Tentativa {tentativa}/{max_tentativas} para {nome_modelo}...")
                
                caminho_relatorio, erro = self._servico_resumo.processar_pasta_documentos(
                    pasta_origem=pasta_documentos,
                    pasta_destino=str(self._pasta_resultados),
                    tipo_modelo=tipo_modelo
                )
                
                if caminho_relatorio:
                    # Sucesso! Sair do loop de retry
                    break
                else:
                    ultimo_erro = erro
                    if tentativa < max_tentativas:
                        print(f"⚠️ Tentativa {tentativa} falhou: {erro}")
                        print(f"🔄 Tentando novamente em 5 segundos...")
                        import time
                        time.sleep(5)
            
            if caminho_relatorio:
                resultados[tipo_modelo] = {
                    "status": "✅ Sucesso",
                    "arquivo": caminho_relatorio
                }
            else:
                resultados[tipo_modelo] = {
                    "status": f"❌ Falhou após {max_tentativas} tentativas: {ultimo_erro}",
                    "arquivo": None
                }
        
        # Exibir resumo final
        self._exibir_resumo_final(resultados, pasta_documentos)
    
    def _garantir_pasta_resultados(self):
        """Garante que a pasta de resultados existe"""
        self._pasta_resultados.mkdir(parents=True, exist_ok=True)
    
    def _exibir_resumo_final(self, resultados: Dict, pasta_analisada: str):
        """Exibe resumo final dos resultados"""
        print("\n🎉 ESTUDO DE CASO 2 CONCLUÍDO!")
        print(f"📁 Pasta analisada: {pasta_analisada}")
        print(f"📊 Resultados salvos em: {self._pasta_resultados}/")
        print()
        
        sucessos = 0
        for tipo_modelo, dados in resultados.items():
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            print(f"🤖 {nome_modelo}: {dados['status']}")
            if dados['arquivo']:
                print(f"   📄 Relatório: {dados['arquivo']}")
                sucessos += 1
        
        print()
        print(f"📈 Taxa de sucesso: {sucessos}/{len(ModelType)} modelos ({(sucessos/len(ModelType)*100):.1f}%)")
        
        if sucessos > 1:
            print("💡 Compare os relatórios para analisar:")
            print("   • Diferenças entre modelos de IA")
            print("   • Padrões identificados nos balancetes")
            print("   • Recomendações de otimização financeira")


# =============================================================================
# PONTO DE ENTRADA PRINCIPAL
# =============================================================================

def main():
    """Função principal do programa"""
    # Configurar infraestrutura
    ConfiguradorLiteLLM.aplicar_patch_parametros()
    
    # Criar dependências
    leitor_documentos = LeitorDocumentosMultiFormato()
    gerador_relatorio = GeradorRelatorioMarkdown()
    
    # Criar serviços
    servico_resumo = ServicoResumoDocumentos(
        leitor_documentos=leitor_documentos,
        gerador_relatorio=gerador_relatorio
    )
    
    # Criar gerenciador e executar
    gerenciador = GerenciadorEstudoCaso2(servico_resumo)
    
    # Exemplo de uso com pasta específica
    # Pode ser alterado para qualquer pasta com documentos
    gerenciador.executar_estudo_completo("arquivo_estudo_2", apenas_llama_7b=False)


if __name__ == "__main__":
    main()