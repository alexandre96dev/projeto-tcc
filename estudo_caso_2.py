"""
Estudo de Caso 2 - Sistema de Resumo Automático de Documentos
Sistema simplificado para processamento em lote de múltiplos documentos
Baseado na conversa com orientador - foco em simplicidade e resultados práticos
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
    """Configuração para cada modelo de IA"""
    model_name: str
    display_name: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 4096
    max_retries: int = 100


@dataclass
class DocumentoProcessado:
    """Representa um documento após processamento"""
    nome_arquivo: str
    caminho_completo: str
    conteudo_original: str
    resumo_gerado: str
    timestamp: datetime


@dataclass
class RelatorioConsolidado:
    """Relatório final consolidado de todos os documentos"""
    introducao: str
    evolucao_padroes: str
    conclusao: str
    documentos_processados: List[DocumentoProcessado]
    modelo_usado: str
    pasta_analisada: str
    timestamp: datetime


class LeitorDocumentos(Protocol):
    """Contrato para leitores de documentos"""
    
    def listar_arquivos_suportados(self, pasta: str) -> List[str]:
        """Lista todos os arquivos suportados em uma pasta"""
        ...
    
    def extrair_conteudo(self, caminho_arquivo: str) -> str:
        """Extrai conteúdo de um arquivo específico"""
        ...


class ProcessadorResumos(Protocol):
    """Contrato para processadores de resumos"""
    
    def gerar_resumo_documento(self, conteudo: str, nome_arquivo: str) -> str:
        """Gera resumo de um documento individual"""
        ...
    
    def gerar_relatorio_consolidado(self, documentos: List[DocumentoProcessado]) -> RelatorioConsolidado:
        """Gera relatório consolidado de múltiplos documentos"""
        ...


class GeradorRelatorio(Protocol):
    """Contrato para geradores de relatório"""
    
    def salvar_relatorio_markdown(self, relatorio: RelatorioConsolidado, pasta_destino: str) -> str:
        """Salva relatório em formato Markdown"""
        ...


# =============================================================================
# IMPLEMENTAÇÕES CONCRETAS
# =============================================================================

class LeitorDocumentosMultiFormato:
    """Leitor que suporta múltiplos formatos de documento"""
    
    EXTENSOES_SUPORTADAS = {'.pdf', '.docx', '.txt', '.md'}
    
    def listar_arquivos_suportados(self, pasta: str) -> List[str]:
        """Lista todos os arquivos suportados em uma pasta"""
        if not os.path.exists(pasta):
            raise FileNotFoundError(f"Pasta '{pasta}' não encontrada")
        
        arquivos_encontrados = []
        pasta_path = Path(pasta)
        
        for arquivo in pasta_path.iterdir():
            if arquivo.is_file() and arquivo.suffix.lower() in self.EXTENSOES_SUPORTADAS:
                arquivos_encontrados.append(str(arquivo))
        
        if len(arquivos_encontrados) < 2:
            raise ValueError(f"Encontrados apenas {len(arquivos_encontrados)} arquivos suportados. Mínimo: 2")
        
        return sorted(arquivos_encontrados)
    
    def extrair_conteudo(self, caminho_arquivo: str) -> str:
        """Extrai conteúdo baseado na extensão do arquivo"""
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
        """Extrai texto de PDF usando PyMuPDF"""
        with fitz.open(caminho) as doc:
            texto = ""
            for pagina in doc:
                texto += pagina.get_text() + "\n"
        
        if not texto.strip():
            raise ValueError(f"PDF '{caminho}' está vazio ou não contém texto extraível")
        
        return texto.strip()
    
    def _extrair_docx(self, caminho: str) -> str:
        """Extrai texto de DOCX usando python-docx"""
        doc = Document(caminho)
        paragrafos = [p.text for p in doc.paragraphs if p.text.strip()]
        
        if not paragrafos:
            raise ValueError(f"DOCX '{caminho}' está vazio ou não contém texto extraível")
        
        return "\n".join(paragrafos)
    
    def _extrair_texto(self, caminho: str) -> str:
        """Extrai texto de arquivos .txt ou .md"""
        with open(caminho, 'r', encoding='utf-8') as f:
            conteudo = f.read().strip()
        
        if not conteudo:
            raise ValueError(f"Arquivo de texto '{caminho}' está vazio")
        
        return conteudo


class ProcessadorResumosIA:
    """Processador que usa IA para gerar resumos e relatórios"""
    
    def __init__(self, modelo: LLM, nome_modelo: str):
        self._modelo = modelo
        self._nome_modelo = nome_modelo
    
    def gerar_resumo_documento(self, conteudo: str, nome_arquivo: str) -> str:
        """Gera resumo usando agente especializado"""
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
        
        # Processar resultado e extrair seções
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
        
        # Padrões para identificar seções
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
                # Fallback: tentar extrair seções sem marcadores
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
            
            # Verifica se linha contém palavras-chave da seção desejada
            if any(palavra in linha_lower for palavra in palavras_chave[nome_secao]):
                capturando = True
                if linha.strip():  # Não adiciona a linha do título
                    continue
            
            # Para de capturar se encontrar palavras de outra seção
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
        
        # Lista de documentos processados
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
        
        # Relatório consolidado
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
    """Factory para criação de modelos de IA - Reutilizada do Estudo de Caso 1"""
    
    CONFIGURACOES_PADRAO = {
        ModelType.LLAMA_7B: ConfiguracaoModelo(
            model_name="replicate/meta/meta-llama-3-8b-instruct",
            display_name="Llama 7B",
            api_key=os.getenv('REPLICATE_API_TOKEN', ''),
            temperature=0.1,
            max_tokens=4096
        ),
        ModelType.LLAMA_70B: ConfiguracaoModelo(
            model_name="replicate/meta/meta-llama-3-70b-instruct",
            display_name="Llama 70B", 
            api_key=os.getenv('REPLICATE_API_TOKEN', ''),
            temperature=0.2,
            max_tokens=8192
        ),
        ModelType.CHATGPT: ConfiguracaoModelo(
            model_name="replicate/openai/gpt-4o-mini",
            display_name="ChatGPT 4.0",
            api_key=os.getenv('OPENAI_API_KEY', ''),
            temperature=0.1,
            max_tokens=6144
        )
    }
    
    @classmethod
    def criar_modelo(cls, tipo_modelo: ModelType) -> LLM:
        """Cria instância de modelo baseada no tipo"""
        config = cls.CONFIGURACOES_PADRAO[tipo_modelo]
        
        if not config.api_key:
            raise ValueError(f"API key não configurada para {config.display_name}")
        
        return LLM(
            model=config.model_name,
            api_key=config.api_key,
            drop_params=["stop"],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries
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
        """Processa todos os documentos de uma pasta com um modelo específico"""
        try:
            # Configurar modelo
            modelo = FabricaModelos.criar_modelo(tipo_modelo)
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            processador = ProcessadorResumosIA(modelo, nome_modelo)
            
            print(f"🔄 Processando com {nome_modelo}...")
            
            # Listar arquivos suportados
            arquivos = self._leitor_documentos.listar_arquivos_suportados(pasta_origem)
            print(f"📁 Encontrados {len(arquivos)} arquivos para processar")
            
            # Processar cada documento
            documentos_processados = []
            
            for i, caminho_arquivo in enumerate(arquivos, 1):
                nome_arquivo = Path(caminho_arquivo).name
                print(f"📄 Processando ({i}/{len(arquivos)}): {nome_arquivo}")
                
                try:
                    # Extrair conteúdo
                    conteudo = self._leitor_documentos.extrair_conteudo(caminho_arquivo)
                    
                    # Gerar resumo
                    resumo = processador.gerar_resumo_documento(conteudo, nome_arquivo)
                    
                    # Criar documento processado
                    doc_processado = DocumentoProcessado(
                        nome_arquivo=nome_arquivo,
                        caminho_completo=caminho_arquivo,
                        conteudo_original=conteudo,
                        resumo_gerado=resumo,
                        timestamp=datetime.now()
                    )
                    
                    documentos_processados.append(doc_processado)
                    print(f"✅ {nome_arquivo}: Resumo gerado com sucesso")
                    
                except Exception as e:
                    print(f"❌ Erro ao processar {nome_arquivo}: {str(e)}")
                    continue
            
            if not documentos_processados:
                return None, "Nenhum documento foi processado com sucesso"
            
            # Gerar relatório consolidado
            print("📊 Gerando relatório consolidado...")
            relatorio = processador.gerar_relatorio_consolidado(documentos_processados)
            
            # Salvar relatório
            caminho_relatorio = self._gerador_relatorio.salvar_relatorio_markdown(
                relatorio, pasta_destino
            )
            
            print(f"✅ {nome_modelo}: Relatório salvo em {caminho_relatorio}")
            return caminho_relatorio, None
            
        except Exception as e:
            erro = f"Erro ao processar com {nome_modelo}: {str(e)}"
            print(f"❌ {erro}")
            return None, erro


class GerenciadorEstudoCaso2:
    """Gerenciador principal do Estudo de Caso 2"""
    
    def __init__(self, servico_resumo: ServicoResumoDocumentos):
        self._servico_resumo = servico_resumo
        self._pasta_resultados = Path("resultados_estudo_caso_2")
    
    def executar_estudo_completo(
        self,
        pasta_documentos: str = "ADMINISTRATIVO_PEDAGOGICO"
    ):
        """Executa o estudo completo com todos os modelos"""
        print("🚀 INICIANDO ESTUDO DE CASO 2 - SISTEMA DE RESUMO AUTOMÁTICO")
        print("📋 Processamento em lote de múltiplos documentos")
        print("📁 Gerando relatórios consolidados com análise de padrões")
        print()
        
        self._garantir_pasta_resultados()
        
        # Verificar se pasta de documentos existe
        if not os.path.exists(pasta_documentos):
            print(f"❌ Pasta '{pasta_documentos}' não encontrada!")
            print("💡 Sugestão: verifique se a pasta existe ou use outro caminho")
            return
        
        resultados = {}
        
        # Processar com cada modelo
        for tipo_modelo in ModelType:
            nome_modelo = FabricaModelos.obter_nome_exibicao(tipo_modelo)
            print(f"\n{'='*60}")
            print(f"🤖 PROCESSANDO COM: {nome_modelo}")
            print(f"{'='*60}")
            
            caminho_relatorio, erro = self._servico_resumo.processar_pasta_documentos(
                pasta_origem=pasta_documentos,
                pasta_destino=str(self._pasta_resultados),
                tipo_modelo=tipo_modelo
            )
            
            if caminho_relatorio:
                resultados[tipo_modelo] = {
                    "status": "✅ Sucesso",
                    "arquivo": caminho_relatorio
                }
            else:
                resultados[tipo_modelo] = {
                    "status": f"❌ Falhou: {erro}",
                    "arquivo": None
                }
        
        # Exibir resumo final
        self._exibir_resumo_final(resultados, pasta_documentos)
    
    def _garantir_pasta_resultados(self):
        """Garante que a pasta de resultados existe"""
        self._pasta_resultados.mkdir(parents=True, exist_ok=True)
    
    def _exibir_resumo_final(self, resultados: Dict, pasta_analisada: str):
        """Exibe resumo final dos resultados"""
        print(f"\n🎉 ESTUDO DE CASO 2 CONCLUÍDO!")
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
            print("💡 Compare os relatórios gerados para analisar as diferenças entre os modelos!")


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
    gerenciador.executar_estudo_completo("ADMINISTRATIVO_PEDAGOGICO")


if __name__ == "__main__":
    main()