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

REPLICATE_MODEL = os.getenv(
    "REPLICATE_MODEL_SLUG",
    "replicate/meta/meta-llama-3-70b-instruct"
)

# Usar Ollama local que é mais estável
client = LLM(
    model=REPLICATE_MODEL,
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
    description=dedent("""
        PRIMEIRO: Execute `LerArquivoTextoTool` com file_path="./Planejamento/PIT.md"

        SEGUNDO: Do conteúdo lido, extraia e escreva EXATAMENTE o conteudo do arquivo lido:
        Atividades de Ensino:
        - Banco de Dados (60h total, 3h semanais)
        - Inteligência Artificial (60h total, 3h semanais)
        - Mineração de Dados (60h total, 3h semanais)
        - Tópicos Especiais em Bancos de Dados (60h total, 3h semanais)
        - Projeto de Desenvolvimento de Software Web (30h total, 1,5h semanais)

        Atividades de Pesquisa:
        - (sem atividades de pesquisa no PIT)

        Atividades de Extensão:
        - Desenvolvimento de Ferramenta de Apoio à Manutenção Industrial (8h semanais)

        Atividades Administrativo-Pedagógicas:
        - Link do Núcleo de Inovação Tecnológica (NIT) (1h semanal)
        - Membro do Núcleo de Empreendedorismo (NEI) (1h semanal)

        Complemento/Observações:
        - Orientação de estágio de estudantes Álvaro Silva, José Marcos Filho e Matheus Barros
        - Declarações e portarias de apoio ao ensino e atividades administrativas em anexo
        
        IMPORTANTE: Use EXATAMENTE este texto, copiando literalmente as informações do PIT.md.
    """),
    expected_output="Arquivo `Relatorio_Final/planejamento.txt` com conteúdo literal do PIT.md.",
    agent=planner,
    output_file="Relatorio_Final/planejamento.txt",
    tools=[ler_arquivo],
)

# arquivos de entrada
sections = {
    "teaching": "./ENSINO/ensino.txt",
    "research": "./PESQUISA/pesquisa.txt",
    "extension": "./EXTENSAO/extensao.txt",
    "admin": "./ADMINISTRATIVO_PEDAGOGICO/admin.txt",
}

research_tasks = []
for section, file_path in sections.items():
    research_tasks.append(
        Task(
            description=dedent(f"""
                TAREFA ESPECÍFICA: Extrair conteúdo do arquivo {file_path}
                
                INSTRUÇÕES OBRIGATÓRIAS:
                1. Execute SOMENTE: `LerArquivoTextoTool` com file_path="{file_path}"
                2. Se o arquivo contém texto: COPIE TODO O CONTEÚDO COMPLETO sem cortar ou resumir
                3. Se o arquivo está vazio: escreva exatamente "sem informações"
                4. Se o arquivo não existe: escreva exatamente "sem informações"
                
                IMPORTANTE: 
                - Copie o conteúdo COMPLETO sem truncar
                - NÃO misture informações de outros arquivos
                - NÃO use contexto de tarefas anteriores
                - Este arquivo deve conter APENAS o conteúdo de "{file_path}"
                
                VALIDAÇÃO: O arquivo final deve ser uma cópia exata do conteúdo de "{file_path}"
            """),
            expected_output=f"Arquivo `Relatorio_Final/{section}.txt` contendo EXCLUSIVAMENTE o conteúdo completo de {file_path}",
            agent=research_agents[section],
            output_file=f"Relatorio_Final/{section}.txt",
            tools=[ler_arquivo],
        )
    )

writing_task = Task(
    description=dedent("""
        GERAÇÃO DE RELATÓRIO COMPLETO - SIGA RIGOROSAMENTE ESTA SEQUÊNCIA:
        
        ETAPA 1 - LEITURA OBRIGATÓRIA (use LerArquivoTextoTool para CADA arquivo):
        1. Leia `Relatorio_Final/planejamento.txt`
        2. Leia `Relatorio_Final/teaching.txt`
        3. Leia `Relatorio_Final/research.txt`
        4. Leia `Relatorio_Final/extension.txt`
        5. Leia `Relatorio_Final/admin.txt`
        
        ETAPA 2 - COMPILAÇÃO DO RELATÓRIO COMPLETO:
        Crie um relatório com TODAS as seções abaixo (formato Markdown):

        # Relatório Acadêmico

        ## Atividades de Ensino
        **Promessas do PIT:**
        ESCREVA LITERALMENTE o conteudo da seção "Atividades de Ensino:" completa do arquivo planejamento.txt
        
        **Atividades realizadas:**
        ESCREVA LITERALMENTE todo o conteúdo do arquivo teaching.txt - se vazio, escreva "sem informações"

        ## Atividades de Pesquisa
        **Promessas do PIT:**
        ESCREVA LITERALMENTE a seção "Atividades de Pesquisa:" completa do arquivo planejamento.txt
        
        **Atividades realizadas:**
        ESCREVA LITERALMENTE todo o conteúdo do arquivo research.txt - se vazio, escreva "sem informações"

        ## Atividades de Extensão
        **Promessas do PIT:**
        ESCREVA LITERALMENTE a seção "Atividades de Extensão:" completa do arquivo planejamento.txt
        
        **Atividades realizadas:**
        ESCREVA LITERALMENTE todo o conteúdo do arquivo extension.txt - se vazio, escreva "sem informações"

        ## Atividades Administrativo-Pedagógicas
        **Promessas do PIT:**
        ESCREVA LITERALMENTE a seção "Atividades Administrativo-Pedagógicas:" completa do arquivo planejamento.txt
        
        **Atividades realizadas:**
        ESCREVA LITERALMENTE todo o conteúdo do arquivo admin.txt - se vazio, escreva "sem informações"

        ## Observações e Complementos
        ESCREVA LITERALMENTE a seção "Complemento/Observações" do arquivo planejamento.txt se existir
        
        IMPORTANTE:
        - NÃO use placeholders como "( contents of ... )" EM NENHUMA CIRCUNSTÂNCIA
        - ESCREVA o conteúdo real de cada arquivo que você leu com LerArquivoTextoTool
        - Substitua os textos pelos dados reais que você leu
        - Se um arquivo estiver vazio, escreva "sem informações"
        - NUNCA deixe texto como "( contents of teaching.txt )" no resultado final
        
        EXEMPLO DO QUE NÃO FAZER:
        **Atividades realizadas:**
        ( contents of teaching.txt )
        
        EXEMPLO DO QUE FAZER:
        **Atividades realizadas:**
        O ensino neste semestre foi marcado por... [conteúdo real do arquivo]
        
        VALIDAÇÃO OBRIGATÓRIA:
        - O arquivo DEVE ter EXATAMENTE 4 seções principais: Ensino, Pesquisa, Extensão, Administrativo-Pedagógicas
        - Se alguma seção estiver faltando, RECRIE ela com base nos arquivos lidos
        - O arquivo final deve ter pelo menos 60 linhas para estar completo
        - NÃO termine o arquivo antes da seção Administrativo-Pedagógicas estar completa
    """),
    expected_output="Arquivo `Relatorio_Final/relatorio_academico.md` COMPLETO com TODAS as 4 seções obrigatórias incluindo Administrativo-Pedagógicas",
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
    agents=[planner, *research_agents.values(), writer, validator, reviewer_report],
    tasks=[planning_task, *research_tasks, writing_task, validate_sections_task, review_final_report_task],
    process=Process.sequential,
    verbose=True,
    llm=client,
)

if __name__ == "__main__":
    os.makedirs("Relatorio_Final", exist_ok=True)
    result = crew.kickoff()
    print("✅ Pipeline concluído.")
