from crewai import Agent, Task, LLM, Crew, Process
from textwrap import dedent
from custom_pit import VerificarPalavraNoPDFTool

# Configuração do LLM
client = LLM(
    model='openai/gpt-4-turbo',
    api_key='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'
)

# Ferramentas
verificar_palavra_no_pdf = VerificarPalavraNoPDFTool()

planner = Agent(
    name="Planejador Acadêmico",
    role="Estruturador e Organizador Mestre do Relatório Acadêmico",
    goal=(
        "Analisar profundamente o Plano Individual de Trabalho (PIT) para identificar **todas as metas, atividades e entregas prometidas** pelo docente. "
        "Estruturar essas informações de forma lógica e categorizada por área (Ensino, Pesquisa, Extensão, Administrativo-Pedagógicas e Complementos/Observações) "
        "para servir como o esqueleto do relatório final. Sua precisão é vital para o alinhamento de todo o projeto."
    ),
    backstory=(
        "Com uma visão estratégica e anos de experiência na análise de documentos acadêmicos e planejamento institucional, "
        "este agente garante que cada promessa e atividade do PIT seja meticulosamente extraída e organizada. "
        "Ele é o arquiteto da coerência do relatório, fornecendo a base sólida para os demais agentes."
    ),
    verbose=True,
    llm=client,
)


research_agents = {
    "teaching": Agent(
        name="Pesquisador de Ensino",
        role="Analista Detalhista de Atividades Acadêmicas e Pedagógicas",
        goal=(
            "Investigar minuciosamente os documentos fornecidos no diretório de ENSINO."
        ),
        backstory="Um veterano em educação, com expertise em pedagogia e análise de currículos. Seu foco é desvendar a essência das atividades de ensino e seu real impacto.",
        verbose=True,
        llm=client
    ),
    "research": Agent(
        name="Pesquisador de Pesquisa",
        role="Especialista em Produção e Análise Científica com Foco em Impacto",
        goal=(
            "Examinar a fundo os documentos da pasta de PESQUISA, identificando projetos de pesquisa desenvolvidos, "
            "publicações científicas (artigos, livros, capítulos), participações em congressos, "
            "captação de recursos e o impacto mensurável dessas atividades na ciência e na sociedade. "
            "Priorize a identificação de resultados e relevância."
        ),
        backstory="Um pesquisador experiente, com olhar apurado para a produção acadêmica e suas métricas de impacto. Ele garante que cada descoberta seja devidamente documentada.",
        verbose=True,
        llm=client
    ),
    "extension": Agent(
        name="Pesquisador de Extensão",
        role="Analista de Projetos de Extensão Universitária e Engajamento Comunitário",
        goal=(
            "Analisar detalhadamente os documentos relacionados às atividades de EXTENSÃO."
        ),
        backstory="Com um histórico em projetos sociais e universitários, este agente é perito em traduzir as ações de extensão em narrativas de impacto real na comunidade.",
        verbose=True,
        llm=client
    ),
    "admin": Agent(
        name="Pesquisador Administrativo",
        role="Analista de Processos Administrativo-Pedagógicos e Governança Acadêmica",
        goal=(
            "Investigar e extrair informações cruciais sobre atividades administrativas e pedagógicas"
        ),
        backstory="Um especialista em gestão universitária, capaz de identificar a relevância das atividades administrativas no panorama acadêmico geral. Ele organiza as contribuições que mantêm a instituição funcionando.",
        verbose=True,
        llm=client
    )
}

writer = Agent(
    name="Redator Acadêmico Chefe",
    role="Arquiteto e Executor da Redação Final do Relatório",
    goal=(
        "Compilar e sintetizar de forma magistral todas as informações coletadas pelos pesquisadores, "
        "cruzando-as rigorosamente com as promessas detalhadas no PIT. "
        "Sua tarefa é elaborar um relatório acadêmico final **coeso, estruturado, formal e bem formatado em Markdown**, "
        "garantindo que todas as seções estejam interligadas e apresentem uma narrativa fluida e profissional. "
        "O relatório deve ser uma representação fiel e persuasiva das atividades realizadas em relação ao planejado."
    ),
    backstory="Com vasta experiência em redação técnica e acadêmica de alto nível, este agente é mestre em transformar dados brutos em uma narrativa convincente e impecável. Ele é o responsável pela voz e pela estrutura do documento final.",
    verbose=True,
    llm=client
)

reviewer_research = Agent(
    name="Revisor Mestre de Conteúdo e Integridade Acadêmica",
    role="Supervisor Crítico das Análises de Ensino, Pesquisa, Extensão e Administração",
    goal=(
        "Revisar e validar profundamente os resumos gerados por cada pesquisador de seção (Ensino, Pesquisa, Extensão, Administrativo), "
        "garantindo a **fidelidade e precisão das informações em relação às fontes originais**, "
        "a **clareza da linguagem**, a **coerência interna** de cada resumo e, crucialmente, sua **coerência com as metas e atividades do PIT** "
        "conforme identificado pelo Planejador. Faça ajustes significativos se necessário para otimizar a qualidade dos dados para o redator."
    ),
    backstory="Um rigoroso especialista em revisão técnica e integridade acadêmica, com um olhar implacável para a precisão dos dados e a conformidade com o planejamento. Ele é o guardião da verdade factual antes da redação final.",
    verbose=True,
    llm=client
)

reviewer_report = Agent(
    name="Revisor Editorial do Relatório Acadêmico Final",
    role="Editor-Chefe e Validador Final do Documento",
    goal=(
        "Realizar uma revisão crítica e abrangente do `Relatorio_Final/relatorio_academico.md` para assegurar sua **coesão estrutural, clareza textual, fluidez da narrativa e conformidade acadêmica geral**. "
        "Verifique a formatação Markdown, a lógica da apresentação e a ausência de redundâncias. "
        "Seu trabalho é garantir que o relatório esteja impecável e pronto para apresentação final."
    ),
    backstory="Um revisor editorial com vasta experiência em publicações acadêmicas, dotado de um olhar clínico para a estrutura, o estilo e a apresentação. Ele lapida o diamante bruto no documento final.",
    verbose=True,
    llm=client
)

evaluator_textual = Agent(
    name="Avaliador Textual Linguístico e Estilístico",
    role="Especialista em Qualidade Linguística e Estilística de Textos Acadêmicos",
    goal=(
        "Avaliar aprofundadamente a **gramática, ortografia, clareza, coesão textual e adequação ao estilo acadêmico** "
        "de **TODAS as seções do relatório** (incluindo `planejamento.txt`, os resumos individuais e o `relatorio_academico.md`). "
        "Você deve identificar e apontar detalhadamente **oportunidades de melhoria** na linguagem, estrutura frasal, uso de vocabulário, "
        "e garantir que o texto atinja um padrão acadêmico elevado, livre de jargões desnecessários ou informalidades."
    ),
    backstory="Um linguista e revisor de textos acadêmicos com um domínio impecável da língua portuguesa e dos requisitos de escrita científica. Ele é o crítico literário do seu relatório.",
    verbose=True,
    llm=client
)

evaluator_metrics = Agent(
    name="Avaliador de Métricas de Aderência ao PIT",
    role="Analista de Conteúdo e Conformidade com o Planejamento Acadêmico",
    goal=(
        "Realizar uma avaliação sistemática da **aderência do conteúdo do relatório final às metas e atividades previstas no PIT**. "
        "Você aplicará e documentará métricas específicas como: "
        "1. **Cobertura:** Quão bem as metas do PIT foram abordadas em cada seção. "
        "2. **Consistência:** A coerência entre o planejado no PIT e o que foi relatado. "
        "3. **Relevância:** A importância e profundidade das atividades relatadas em relação às promessas. "
        "4. **Proporção Promessa vs. Entrega:** O equilíbrio entre o que foi prometido e o que foi efetivamente detalhado. "
        "Gere um relatório de avaliação detalhado com pontuações (se aplicável) e justificativas."
    ),
    backstory="Um especialista em avaliação de projetos institucionais, com experiência em mensurar o sucesso de iniciativas acadêmicas em relação aos seus planos originais. Ele é o auditor do seu desempenho.",
    verbose=True,
    llm=client
)

planning_task = Task(
    description=dedent("""
        Analise o documento do Plano Individual de Trabalho (PIT) localizado em './Planejamento/PIT.pdf'.
        Sua principal responsabilidade é **identificar e extrair todas as metas, atividades e entregas prometidas** pelo docente, categorizando-as claramente por seção:
        - Atividades de Ensino
        - Atividades de Pesquisa
        - Atividades de Extensão
        - Atividades Administrativo-Pedagógicas
        - Complemento/Observações
        Organize essas informações de forma estruturada, com listas claras ou tópicos e crie o arquivo `Relatorio_Final/planejamento.txt` com as informações coletadas. Não invente informações
        **O formato deve ser limpo e fácil de ler, servindo como um índice de promessas.**
    """),
    expected_output="Arquivo `./Relatorio_Final/planejamento.txt` contendo as promessas do PIT organizadas de forma estruturada por seção, prontas para cruzamento com o relatório.",
    agent=planner,
    output_file="./Relatorio_Final/planejamento.txt",
    tools=[verificar_palavra_no_pdf] # Mantendo a ferramenta se ela for útil para verificar a presença de termos chave do PIT
)


research_tasks = []
sections = {
    "teaching": "./ENSINO/ensino.pdf",
    "research": "./PESQUISA/pesquisa.pdf",
    "extension": "./EXTENSAO/extensao.pdf",
    "admin": "./ADMINISTRATIVO_PEDAGOGICO/admin.pdf"
}

for section, file_path in sections.items():
    research_tasks.append(Task(
        description=dedent(f"""
            Leia o documento '{file_path}'. Caso ele esteja vazio, escreva APENAS: 'Sem informações fornecidas para esta seção'.
            Caso haja informações, extraia e resuma os dados mais relevantes e quantificáveis para a seção de {section.upper()}.
            NÃO invente ou complemente dados! O resumo deve ser estritamente baseado no conteúdo do arquivo.
            Salve o resumo em Relatorio_Final/{section}.txt.
        """),
        expected_output=f"Arquivo Relatorio_Final/{section}.txt com um resumo factual ESTRITAMENTE com base no arquivo fonte, ou 'Sem informações fornecidas para esta seção'.",
        agent=research_agents[section],
        output_file=f"Relatorio_Final/{section}.txt",
        tools=[verificar_palavra_no_pdf]
    ))

writing_task = Task(
    description=dedent("""
        Sua missão é compilar o relatório acadêmico final, ESTRITAMENTE com base nos seguintes arquivos já gerados:
        - Relatorio_Final/planejamento.txt
        - Relatorio_Final/teaching.txt
        - Relatorio_Final/research.txt
        - Relatorio_Final/extension.txt
        - Relatorio_Final/admin.txt

        Para cada seção (Ensino, Pesquisa, Extensão, Administrativo), só insira informações que estejam nos arquivos correspondentes. 
        Caso algum arquivo contenha a frase 'Sem informações fornecidas para esta seção', inclua esse aviso na respectiva seção do relatório.

        NÃO invente, complemente ou deduza informações de nenhum outro lugar. Sua função é apenas organizar e compilar.

        Estruture o relatório em Markdown, com:
        - Introdução breve contextualizando o período do relatório.
        - Seções separadas para Ensino, Pesquisa, Extensão e Administrativo-Pedagógico, contendo o conteúdo dos respectivos arquivos, sem alterações ou inferências.
        - Uma conclusão, que apenas sumariza a entrega dos arquivos, sem criar análises além do que está presente.

        Atenção: Se qualquer seção estiver sem dados, escreva explicitamente 'Sem informações reportadas nesta seção'.

        Salve o relatório final em Relatorio_Final/relatorio_academico.md.
    """),
    expected_output="Arquivo Relatorio_Final/relatorio_academico.md contendo o relatório acadêmico final, sem dados inventados, apenas compilados dos arquivos das seções.",
    agent=writer,
    output_file="Relatorio_Final/relatorio_academico.md",
    tools=[]
)

review_final_report_task = Task(
    description=dedent("""
        Revise o relatório final em Relatorio_Final/relatorio_academico.md.
        - Garanta que a estrutura, formatação Markdown e linguagem estejam corretas e formais.
        - Não adicione conteúdo novo; apenas melhore a apresentação textual e formatação.
        - Assegure que, se houver seção sem dados, o texto 'Sem informações reportadas nesta seção' esteja presente e visível.

        Faça correções linguísticas, ortográficas e de formatação, mas NÃO invente informações ou modifique o conteúdo factual.
        Salve o arquivo revisado no mesmo local.
    """),
    expected_output="Relatório final revisado e impecável em Relatorio_Final/relatorio_academico.md, sem alterar informações.",
    agent=reviewer_report,
    tools=[]
)

textual_review_task = Task(
    description=dedent("""
        Avalie TODOS os seguintes arquivos:
        - Relatorio_Final/planejamento.txt
        - Relatorio_Final/teaching.txt
        - Relatorio_Final/research.txt
        - Relatorio_Final/extension.txt
        - Relatorio_Final/admin.txt
        - Relatorio_Final/relatorio_academico.md

        Para cada arquivo, comente obrigatoriamente:
        - Clareza textual
        - Coesão e fluidez
        - Correção gramatical
        - Adequação acadêmica

        Mesmo que algum arquivo esteja vazio ou contenha apenas a mensagem 'Sem informações fornecidas para esta seção', faça uma nota sobre isso.

        Gere um relatório detalhado em Markdown, incluindo sugestões de melhoria, mesmo que seja apenas para indicar que não há conteúdo a avaliar. Salve como Relatorio_Final/avaliacao_textual.md.
    """),
    expected_output="Arquivo Relatorio_Final/avaliacao_textual.md com avaliação linguística de TODOS os arquivos, mesmo que estejam vazios.",
    agent=evaluator_textual,
    output_file="Relatorio_Final/avaliacao_textual.md",
    tools=[verificar_palavra_no_pdf]
)

academic_metrics_task = Task(
    description=dedent("""
        Avalie a aderência do relatório final ao PIT, ESTRITAMENTE com base no conteúdo dos arquivos:
        - Relatorio_Final/planejamento.txt
        - Relatorio_Final/teaching.txt
        - Relatorio_Final/research.txt
        - Relatorio_Final/extension.txt
        - Relatorio_Final/admin.txt
        - Relatorio_Final/relatorio_academico.md

        NÃO invente dados nem crie justificativas além do que está explicitamente presente nos arquivos.

        Critérios:
        1. Cobertura de Metas (por Seção): liste as metas do PIT que foram abordadas, e as que não foram (conforme os arquivos das seções).
        2. Consistência: cite qualquer discrepância entre promessas e entregas dos arquivos.
        3. Relevância e Profundidade: só avalie o que está escrito, não o que seria esperado.
        4. Proporção Promessa vs. Entrega: analise quantitativamente (porcentagem de metas entregues, se possível).

        O resultado final deve ser salvo em Relatorio_Final/avaliacao_metrica.md.
    """),
    expected_output="Arquivo Relatorio_Final/avaliacao_metrica.md detalhando aderência do relatório ao PIT apenas com base nos arquivos disponíveis.",
    agent=evaluator_metrics,
    output_file="Relatorio_Final/avaliacao_metrica.md",
    tools=[verificar_palavra_no_pdf]
)

crew = Crew(
    agents=[planner, *research_agents.values(), writer, reviewer_research, reviewer_report, evaluator_textual, evaluator_metrics],
    tasks=[planning_task, *research_tasks, writing_task, review_final_report_task, textual_review_task,
        academic_metrics_task],
    process=Process.sequential,
    verbose=True,
    llm=client
)

if __name__ == "__main__":
    crew.kickoff()