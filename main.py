from crewai import Agent, Task, LLM, Crew, Process
from textwrap import dedent
from langchain_community.chat_models import ChatLiteLLM
from custom_pit import VerificarPalavraNoPDFTool

client =  LLM(
    model='ollama/llama3.1',
    api_base='http://localhost:11434'
)

verificar_palavra_no_pdf = VerificarPalavraNoPDFTool()

planner = Agent(
    name="Planejador Acadêmico",
    role="Estruturador e Organizador do Relatório Acadêmico",
    goal=(
        "Analisar o Plano Individual de Trabalho (PIT), identificar todas as metas e atividades propostas "
        "e organizar as informações de forma estruturada para a composição do relatório final. "
        "Colaborar com outros agentes para garantir a integração das diferentes seções."
    ),
    backstory=(
        "Especialista em análise e estruturação de documentos acadêmicos, o Planejador Acadêmico "
        "possui uma visão estratégica para garantir que todas as atividades prometidas no PIT "
        "sejam devidamente consideradas e documentadas no relatório final. "
        "Trabalha em estreita colaboração com outros agentes para otimizar o processo."
    ),
    verbose=True,
    llm=client,
)

# Agentes de Pesquisa para cada seção
research_agents = {
    "teaching": Agent(
        name="Pesquisador de Ensino",
        role="Analista de Atividades Acadêmicas e Pedagógicas",
        goal=(
            "Investigar detalhadamente os documentos relacionados às atividades de ensino, "
            "extraindo informações sobre carga horária, disciplinas ministradas, metodologias utilizadas "
            "e impactos na formação dos alunos."
        ),
        backstory="Especialista em avaliação educacional e análise pedagógica.",
        verbose=True,
        llm=client
    ),
    "research": Agent(
        name="Pesquisador de Pesquisa",
        role="Especialista em Produção e Análise Científica",
        goal=(
            "Examinar os documentos disponíveis na pasta de pesquisa, identificando projetos científicos desenvolvidos, "
            "artigos publicados, participação em eventos acadêmicos e impacto da pesquisa na comunidade acadêmica."
        ),
        backstory="Especialista em análise de produção científica e métricas de impacto.",
        verbose=True,
        llm=client
    ),
    "extension": Agent(
        name="Pesquisador de Extensão",
        role="Analista de Projetos de Extensão Universitária",
        goal=(
            "Analisar os documentos relacionados às atividades de extensão, identificando projetos realizados, "
            "parcerias institucionais, impacto na comunidade e resultados alcançados."
        ),
        backstory="Especialista na articulação entre ensino, pesquisa e comunidade.",
        verbose=True,
        llm=client
    ),
    "admin": Agent(
        name="Pesquisador Administrativo",
        role="Analista de Processos Administrativo-Pedagógicos",
        goal=(
            "Investigar e extrair informações sobre atividades administrativas e pedagógicas, incluindo participação em colegiados, "
            "gestão acadêmica, organização de eventos e demais ações institucionais."
        ),
        backstory="Especialista em gestão acadêmica e análise administrativa.",
        verbose=True,
        llm=client
    )
}

# Agente de Escrita
writer = Agent(
    name="Redator Acadêmico",
    role="Responsável pela Redação do Relatório Final",
    goal=(
        "Compilar as informações coletadas pelos pesquisadores e cruzar os dados com as promessas do PIT, "
        "elaborando um relatório acadêmico estruturado e bem formatado."
    ),
    backstory="Especialista em redação técnica e acadêmica.",
    verbose=True,
    llm=client
)

# ===================== TAREFAS =====================

# Tarefa de Planejamento
planning_task = Task(
    description=dedent("""
        Analisar o Plano Individual de Trabalho no diretório './Planejamento/PIT.pdf'.
        Identificar todas as metas e atividades propostas pelo docente para as diferentes áreas do relatório:
        - Atividades de Ensino
        - Atividades de Pesquisa
        - Atividades de Extensão
        - Atividades Administrativo-Pedagógicas
        - Complemento/Observações
        Organizar essas informações de forma estruturada e salvar no arquivo `Relatorio_Final/planejamento.txt`.
    """),
    expected_output="Arquivo `./Relatorio_Final/planejamento.txt` contendo as promessas organizadas por seção.",
    agent=planner,
    output_file="./Relatorio_Final/planejamento.txt",
    tools=[verificar_palavra_no_pdf]
)

# Tarefas de Pesquisa para cada seção
research_tasks = []
sections = {
    "teaching": "./ENSINO/ensino.txt",
    "research": "./PESQUISA/pesquisa.txt",
    "extension": "./EXTENSAO/extensao.txt",
    "admin": "./ADMINISTRATIVO_PEDAGOGICO/admin.txt"
}
for section, file_path in sections.items():
    research_tasks.append(Task(
        description=dedent(f"""
            Examinar o documento '{file_path}'.
            Extrair informações relevantes para a seção de {section.upper()}.
            Salvar o resumo no arquivo `Relatorio_Final/{section}.txt`.
        """),
        expected_output=f"Arquivo `Relatorio_Final/{section}.txt` contendo o resumo das atividades de {section}.",
        agent=research_agents[section],
        output_file=f"Relatorio_Final/{section}.txt",
        tools=[verificar_palavra_no_pdf]
    ))

# Tarefa de Escrita
writing_task = Task(
    description=dedent("""
        Compilar as informações extraídas das seguintes fontes:
        - `Relatorio_Final/planejamento.txt`
        - `Relatorio_Final/teaching.txt`
        - `Relatorio_Final/research.txt`
        - `Relatorio_Final/extension.txt`
        - `Relatorio_Final/admin.txt`
        Cruzar os dados obtidos com as metas e atividades propostas no PIT, redigir um relatório acadêmico coeso, formatado e bem estruturado.
        Salvar o relatório final em `Relatorio_Final/relatorio_academico.md`.
    """),
    expected_output="Arquivo `Relatorio_Final/relatorio_academico.md` contendo o relatório acadêmico final.",
    agent=writer,
    output_file="Relatorio_Final/relatorio_academico.md",
    tools=[]
)

crew = Crew(
    agents=[planner, *research_agents.values(), writer],
    tasks=[planning_task, *research_tasks, writing_task],
    process=Process.sequential,
    verbose=True,
    llm=client
)

if __name__ == "__main__":
    crew.kickoff()
