# -*- coding: utf-8 -*-
from crewai import Agent, Task, LLM, Crew, Process
from textwrap import dedent
import os
import json

from custom_pit import (
    SegmentarPITPorSecoesTool,
    ExtrairTextoPDFTool,
    VerificarTrechoTool,
    LerArquivoTextoTool,
    LerJSONTool,
)

# -------------------- PATCH CRÍTICO (remove stop/stop_sequences) --------------------
import litellm

# (Opcional) ative para ver o payload detalhado no console
# litellm._turn_on_debug()

_original_completion = litellm.completion
def _completion_no_stop(**params):
    # Remover campos de stop que causam 422 na Replicate (algumas libs injetam listas)
    for k in ["stop", "stop_sequences", "stops", "stop_tokens"]:
        if k in params:
            params.pop(k, None)
    # Algumas integrações aninham nas extras:
    if "extra_body" in params and isinstance(params["extra_body"], dict):
        for k in ["stop", "stop_sequences"]:
            params["extra_body"].pop(k, None)
    return _original_completion(**params)

# aplica o patch globalmente antes de qualquer chamada
litellm.completion = _completion_no_stop
# ------------------------------------------------------------------------------------
# Usar Ollama local que é mais estável
client = LLM(
    model="replicate/openai/gpt-4o-mini",
    api_key='r8_MPjPwXOOQ4ZORa5teY6esvCY6AfJr2p1frYPn',
    # Pode manter estes sem problemas; o patch acima garante remoção se aparecerem
    drop_params=["stop"],
    temperature=0.1,  # Baixa temperatura para mais determinismo
    max_tokens=4096,
    max_retries=200,
)

# -------- Tools --------
segmentar_pit = SegmentarPITPorSecoesTool()
verificar_trecho = VerificarTrechoTool()
ler_arquivo = LerArquivoTextoTool()
ler_json = LerJSONTool()

# -------- Agents --------
planner = Agent(
    name="Planejador Acadêmico",
    role="Extrator literal de conteúdo do PIT",
    goal=(
        "IMPORTANTE: Extrair APENAS texto que existe literalmente no PIT. "
        "NÃO interprete, NÃO elabore, NÃO adicione contexto. "
        "Se não estiver escrito exatamente assim no documento, NÃO inclua. "
        "Gerar `Relatorio_Final/planejamento.txt` com cópia fiel do PIT."
    ),
    backstory=(
        "Especialista em extração literal de texto. "
        "NUNCA inventa ou interpreta conteúdo. "
        "Trabalha exclusivamente com texto que pode ser verificado no documento original."
    ),
    verbose=True,
    llm=client,
)

research_agents = {
    "teaching": Agent(
        name="Extrator de Ensino",
        role="Extrator literal de atividades de ensino",
        goal=(
            "COPIAR literalmente o conteúdo do arquivo. "
            "NÃO resumir, NÃO interpretar, NÃO adicionar informações. "
            "Se arquivo vazio, informar exatamente isso."
        ),
        backstory=(
            "Especialista em cópia literal de texto. "
            "PROIBIDO inventar ou elaborar qualquer conteúdo."
        ),
        verbose=True,
        llm=client,
    ),
    "research": Agent(
        name="Extrator de Pesquisa",
        role="Extrator literal de atividades de pesquisa",
        goal=(
            "COPIAR literalmente o conteúdo do arquivo. "
            "NÃO resumir, NÃO interpretar, NÃO adicionar informações. "
            "Se arquivo vazio, informar exatamente isso."
        ),
        backstory=(
            "Especialista em cópia literal de texto. "
            "PROIBIDO inventar ou elaborar qualquer conteúdo."
        ),
        verbose=True,
        llm=client,
    ),
    "extension": Agent(
        name="Extrator de Extensão",
        role="Extrator literal de atividades de extensão",
        goal=(
            "COPIAR literalmente o conteúdo do arquivo. "
            "NÃO resumir, NÃO interpretar, NÃO adicionar informações. "
            "Se arquivo vazio, informar exatamente isso."
        ),
        backstory=(
            "Especialista em cópia literal de texto. "
            "PROIBIDO inventar ou elaborar qualquer conteúdo."
        ),
        verbose=True,
        llm=client,
    ),
    "admin": Agent(
        name="Extrator Administrativo",
        role="Extrator literal de atividades administrativo-pedagógicas",
        goal=(
            "COPIAR literalmente o conteúdo do arquivo. "
            "NÃO resumir, NÃO interpretar, NÃO adicionar informações. "
            "Se arquivo vazio, informar exatamente isso."
        ),
        backstory=(
            "Especialista em cópia literal de texto. "
            "PROIBIDO inventar ou elaborar qualquer conteúdo."
        ),
        verbose=True,
        llm=client,
    ),
}

writer = Agent(
    name="Compilador de Relatório",
    role="Compilador especializado em agregação de dados",
    goal=(
        "COMPILAR arquivos de dados em um relatório estruturado. "
        "Ler TODOS os arquivos necessários e montar relatório completo. "
        "Usar conteúdo literal dos arquivos, sem interpretação ou análise."
    ),
    backstory=(
        "Especialista em compilação de documentos acadêmicos. "
        "Trabalha metodicamente lendo cada arquivo e montando relatórios estruturados. "
        "Focado em eficiência e completude, evitando travamentos."
    ),
    verbose=True,
    llm=client,
    max_iter=3,  # Limita iterações para evitar loops
    max_execution_time=300,  # Timeout de 5 minutos
)

reviewer_report = Agent(
    name="Validador Final",
    role="Verificador de precisão literal",
    goal=(
        "REVISAR apenas clareza e formatação SEM alterar nenhum fato ou conteúdo. "
        "PROIBIDO adicionar, remover ou modificar qualquer informação factual."
    ),
    backstory=(
        "Revisor técnico especializado em formatação. "
        "Trabalha exclusivamente com correções de forma, nunca de conteúdo."
    ),
    verbose=True,
    llm=client,
)

# Adicionar agent validador
validator = Agent(
    name="Validador de Completude",
    role="Validador especializado em verificação de seções obrigatórias",
    goal=(
        "GARANTIR que o relatório acadêmico contenha TODAS as 4 seções obrigatórias: "
        "Ensino, Pesquisa, Extensão e Administrativo-Pedagógicas. "
        "Se alguma seção estiver faltando, ADICIONAR usando os arquivos fonte."
    ),
    backstory=(
        "Especialista em controle de qualidade de documentos acadêmicos. "
        "Responsável por garantir que nenhuma seção obrigatória seja omitida."
    ),
    verbose=True,
    llm=client,
)

# -------- Tasks --------

planning_task = Task(
    description="""
        Execute APENAS: LerArquivoTextoTool com file_path="./Planejamento/PIT.md"
        
        Retorne EXATAMENTE o conteúdo lido, sem modificações.
    """,
    expected_output="Conteúdo completo do arquivo ./Planejamento/PIT.md",
    agent=planner,
    output_file="Relatorio_Final/planejamento.txt",
    tools=[ler_arquivo],
)

# arquivos de entrada
sections = {
    "teaching": "ENSINO/ensino.txt",
    "research": "PESQUISA/pesquisa.txt",
    "extension": "EXTENSAO/extensao.txt",
    "admin": "ADMINISTRATIVO_PEDAGOGICO/admin.txt",
}

research_tasks = []
for section, file_path in sections.items():
    research_tasks.append(
        Task(
            description=f"""
                Execute APENAS: LerArquivoTextoTool com file_path="{file_path}"
                
                Retorne EXATAMENTE o conteúdo lido, sem modificações.
                Se houver erro, retorne a mensagem de erro.
            """,
            expected_output=f"Conteúdo completo do arquivo {file_path}",
            agent=research_agents[section],
            output_file=f"Relatorio_Final/{section}.txt",
            tools=[ler_arquivo],
        )
    )

writing_task = Task(
    description="""
        1. Leia cada arquivo com LerArquivoTextoTool:
           - Relatorio_Final/planejamento.txt
           - Relatorio_Final/teaching.txt  
           - Relatorio_Final/research.txt
           - Relatorio_Final/extension.txt
           - Relatorio_Final/admin.txt

        2. Crie um relatório Markdown com estas seções:
           # Relatório Acadêmico
           
           ## Atividades de Ensino
           **Promessas do PIT:** [extrair do planejamento.txt]
           **Atividades realizadas:** [conteúdo de teaching.txt]
           
           ## Atividades de Pesquisa  
           **Promessas do PIT:** [extrair do planejamento.txt]
           **Atividades realizadas:** [conteúdo de research.txt]
           
           ## Atividades de Extensão
           **Promessas do PIT:** [extrair do planejamento.txt] 
           **Atividades realizadas:** [conteúdo de extension.txt]
           
           ## Atividades Administrativo-Pedagógicas
           **Promessas do PIT:** [extrair do planejamento.txt]
           **Atividades realizadas:** [conteúdo de admin.txt]
    """,
    expected_output="Relatório completo em Relatorio_Final/relatorio_academico.md",
    agent=writer,
    output_file="Relatorio_Final/relatorio_academico.md",
    tools=[ler_arquivo],
)

validate_sections_task = Task(
    description=dedent("""
        VALIDAÇÃO CRÍTICA - VERIFICAR SEÇÕES OBRIGATÓRIAS
        
        MISSÃO: Garantir que o relatório acadêmico contenha TODAS as 4 seções obrigatórias
        
        PROCEDIMENTO:
        1. Use `LerArquivoTextoTool` para ler `Relatorio_Final/relatorio_academico.md`
        2. Verifique se o arquivo contém estas seções OBRIGATÓRIAS:
           - ## Atividades de Ensino
           - ## Atividades de Pesquisa  
           - ## Atividades de Extensão
           - ## Atividades Administrativo-Pedagógicas
        3. VERIFIQUE se há placeholders como "( contents of ... )" no arquivo
        
        AÇÃO CORRETIVA (se alguma seção estiver faltando OU se houver placeholders):
        4. Leia os arquivos fonte necessários:
           - `Relatorio_Final/planejamento.txt`
           - `Relatorio_Final/teaching.txt`
           - `Relatorio_Final/research.txt`
           - `Relatorio_Final/extension.txt`
           - `Relatorio_Final/admin.txt`
        5. SUBSTITUA qualquer placeholder pelo conteúdo real dos arquivos
        6. ADICIONE seções faltantes com conteúdo real
        
        FOCO ESPECIAL: Se "Atividades Administrativo-Pedagógicas" estiver faltando:
        - OBRIGATORIAMENTE adicione esta seção
        - Use o conteúdo de planejamento.txt e admin.txt
        - Posicione antes das Observações ou no final
        
        RESULTADO: Relatório com TODAS as 4 seções presentes e completas
    """),
    expected_output="Relatório validado com TODAS as seções obrigatórias incluindo Administrativo-Pedagógicas",
    agent=validator,
    tools=[ler_arquivo],
)

review_final_report_task = Task(
    description=dedent("""
        REVISÃO FINAL - VERIFICAÇÃO OBRIGATÓRIA DE COMPLETUDE
        
        INSTRUÇÃO OBRIGATÓRIA:
        1. Use `LerArquivoTextoTool` para ler `Relatorio_Final/relatorio_academico.md`
        2. VERIFIQUE OBRIGATORIAMENTE se o arquivo contém TODAS as seções:
           ✓ ## Atividades de Ensino (completa com promessas + realizadas)
           ✓ ## Atividades de Pesquisa (completa com promessas + realizadas)
           ✓ ## Atividades de Extensão (completa com promessas + realizadas)
           ✓ ## Atividades Administrativo-Pedagógicas (OBRIGATÓRIA - completa com promessas + realizadas)
           ✓ ## Observações (se houver)
        
        AÇÃO CORRETIVA OBRIGATÓRIA:
        - Se a seção "Atividades Administrativo-Pedagógicas" estiver faltando ou incompleta:
          a) Leia `Relatorio_Final/planejamento.txt` e `Relatorio_Final/admin.txt`
          b) ADICIONE a seção completa ao final do relatório antes das observações
        
        VALIDAÇÃO FINAL:
        - O arquivo DEVE ter pelo menos 60 linhas
        - TODAS as 4 seções principais devem estar presentes
        - Se algo estiver faltando, COMPLETE baseado nos arquivos fonte
        
        CORREÇÕES PERMITIDAS (além da verificação de completude):
        - Melhorar formatação Markdown
        - Corrigir espaçamentos
        - Padronizar títulos
    """),
    expected_output="Relatório final COMPLETO e verificado com TODAS as seções em `Relatorio_Final/relatorio_academico.md`",
    agent=reviewer_report,
    tools=[ler_arquivo],
)

# -------- Crew --------
crew = Crew(
    agents=[planner, *research_agents.values(), writer],
    tasks=[planning_task, *research_tasks, writing_task],
    process=Process.sequential,
    verbose=True,
    llm=client,
)

if __name__ == "__main__":
    os.makedirs("Relatorio_Final", exist_ok=True)
    result = crew.kickoff()
    print("✅ Pipeline concluído.")
