from crewai import Agent, Task, LLM, Crew, Process
from textwrap import dedent
from langchain_community.chat_models import ChatLiteLLM
from custom_pit import VerificarPalavraNoPDFTool

# Configuração do LLM
client = LLM(
    model='openai/gpt-3.5-turbo',
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
            "Investigar minuciosamente os documentos fornecidos no diretório de ENSINO, "
            "extraindo dados específicos sobre carga horária, ementas, metodologias didáticas, "
            "projetos pedagógicos e o impacto direto nas turmas e na formação dos alunos. "
            "O resumo deve ser factual, detalhado e conciso."
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
            "Analisar detalhadamente os documentos relacionados às atividades de EXTENSÃO, "
            "identificando projetos de extensão realizados, parcerias com a comunidade/instituições, "
            "o público beneficiado, resultados alcançados e o impacto social das ações. "
            "Busque evidências do engajamento e transformação."
        ),
        backstory="Com um histórico em projetos sociais e universitários, este agente é perito em traduzir as ações de extensão em narrativas de impacto real na comunidade.",
        verbose=True,
        llm=client
    ),
    "admin": Agent(
        name="Pesquisador Administrativo",
        role="Analista de Processos Administrativo-Pedagógicos e Governança Acadêmica",
        goal=(
            "Investigar e extrair informações cruciais sobre atividades administrativas e pedagógicas, "
            "incluindo participação em colegiados e comissões, gestão de cursos, organização de eventos institucionais, "
            "desenvolvimento de políticas acadêmicas e outras contribuições para a governança universitária. "
            "Destaque a contribuição para a estrutura e funcionamento da instituição."
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
        Organize essas informações de forma estruturada, com listas claras ou tópicos, no arquivo `Relatorio_Final/planejamento.txt`.
        **O formato deve ser limpo e fácil de ler, servindo como um índice de promessas.**
    """),
    expected_output="Arquivo `./Relatorio_Final/planejamento.txt` contendo as promessas do PIT organizadas de forma estruturada por seção, prontas para cruzamento com o relatório.",
    agent=planner,
    output_file="./Relatorio_Final/planejamento.txt",
    tools=[verificar_palavra_no_pdf] # Mantendo a ferramenta se ela for útil para verificar a presença de termos chave do PIT
)


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
            Examine o documento '{file_path}' (ou o conteúdo relevante, se o arquivo for um placeholder).
            Sua tarefa é **extrair e resumir as informações mais relevantes e quantificáveis** para a seção de {section.upper()}.
            Foco em:
            - **Ensino:** Carga horária, disciplinas, metodologias, projetos pedagógicos, número de alunos/turmas, resultados de avaliação.
            - **Pesquisa:** Títulos de projetos, publicações (com local/data), eventos (participação/apresentação), financiamentos, resultados chave.
            - **Extensão:** Nomes de projetos, parcerias, número de beneficiados, resultados sociais, eventos comunitários.
            - **Administrativo:** Comissões/colegiados (com papel), participação em eventos de gestão, ações de organização acadêmica, desenvolvimento de políticas.
            O resumo deve ser objetivo, factual e conter apenas as informações essenciais para a composição do relatório. Salve o resumo no arquivo `Relatorio_Final/{section}.txt`.
        """),
        expected_output=f"Arquivo `Relatorio_Final/{section}.txt` contendo um resumo factual e detalhado das atividades de {section}, pronto para ser usado na redação do relatório.",
        agent=research_agents[section],
        output_file=f"Relatorio_Final/{section}.txt",
        tools=[verificar_palavra_no_pdf] 
    ))

writing_task = Task(
    description=dedent("""
        Sua missão é criar o relatório acadêmico final. Compile e sintetize as informações das seguintes fontes:
        - O planejamento das promessas do PIT: `Relatorio_Final/planejamento.txt`
        - Os resumos das atividades de ensino: `Relatorio_Final/teaching.txt`
        - Os resumos das atividades de pesquisa: `Relatorio_Final/research.txt`
        - Os resumos das atividades de extensão: `Relatorio_Final/extension.txt`
        - Os resumos das atividades administrativo-pedagógicas: `Relatorio_Final/admin.txt`
        Cruze todos esses dados com as metas e atividades propostas no PIT, elaborando um relatório acadêmico **coeso, bem estruturado, formal e formatado em Markdown**.
        O relatório deve incluir:
        - Uma introdução que contextualize o período e o objetivo do relatório.
        - Seções claras para Ensino, Pesquisa, Extensão e Administrativo-Pedagógicas, cada uma detalhando as atividades realizadas e fazendo referência explícita (se possível) às promessas do PIT.
        - Uma seção de Conclusão/Considerações Finais, que resuma as principais conquistas e talvez futuras projeções.
        Assegure uma linguagem acadêmica e profissional, evitando jargões excessivos e mantendo a fluidez. Salve o relatório final em `Relatorio_Final/relatorio_academico.md`.
    """),
    expected_output="Arquivo `Relatorio_Final/relatorio_academico.md` contendo o relatório acadêmico final completo, estruturado em Markdown, coerente e formal.",
    agent=writer,
    output_file="Relatorio_Final/relatorio_academico.md",
    tools=[]
)

review_final_report_task = Task(
    description=dedent("""
        Realize uma revisão editorial e estrutural abrangente do relatório em `Relatorio_Final/relatorio_academico.md`.
        Seu foco é garantir:
        - **Coesão e Fluxo:** A transição suave entre seções e parágrafos.
        - **Estrutura:** A correta aplicação das seções, subseções e formatação Markdown.
        - **Clareza:** A ausência de ambiguidades e a facilidade de compreensão do texto.
        - **Conformidade Acadêmica:** O respeito às normas de escrita acadêmica e à linguagem formal.
        - **Consistência:** Que as informações apresentadas no relatório final sejam consistentes com os resumos das seções e o planejamento do PIT.
        **Corrija quaisquer erros gramaticais, ortográficos ou de pontuação restantes.** Otimize a fraseologia para concisão e impacto. Salve a versão final corrigida no mesmo arquivo.
    """),
    expected_output="Relatório final revisado e impecável em `Relatorio_Final/relatorio_academico.md`, pronto para ser avaliado.",
    agent=reviewer_report,
    tools=[]
)

textual_review_task = Task(
    description=dedent("""
        Sua tarefa é realizar uma análise crítica da qualidade linguística e textual de **TODOS** os seguintes arquivos:
        - `Relatorio_Final/planejamento.txt`
        - `Relatorio_Final/teaching.txt`
        - `Relatorio_Final/research.txt`
        - `Relatorio_Final/extension.txt`
        - `Relatorio_Final/admin.txt`
        - `Relatorio_Final/relatorio_academico.md`

        Para cada arquivo, avalie rigorosamente os seguintes aspectos:
        - **Clareza Textual:** O texto é fácil de entender? As ideias são apresentadas de forma direta?
        - **Coesão e Fluidez:** As frases e parágrafos se conectam logicamente? Há transições suaves?
        - **Correção Gramatical e Ortográfica:** Identifique quaisquer erros de gramática, pontuação ou ortografia.
        - **Adequação ao Estilo Acadêmico:** O tom é formal e objetivo? Há uso apropriado de terminologia? Evita informalidades ou jargões desnecessários?

        Gere um relatório de avaliação detalhado em **Markdown**, descrevendo as observações para cada arquivo e apontando **oportunidades específicas de melhoria**. Inclua exemplos de trechos problemáticos e sugestões de reescrita. Salve este relatório em `Relatorio_Final/avaliacao_textual.md`.
    """),
    expected_output="Arquivo `Relatorio_Final/avaliacao_textual.md` com um relatório detalhado em Markdown, contendo as avaliações linguísticas e sugestões para cada arquivo analisado.",
    agent=evaluator_textual,
    output_file="Relatorio_Final/avaliacao_textual.md",
    tools=[verificar_palavra_no_pdf]
)

academic_metrics_task = Task(
    description=dedent("""
        Com base na análise comparativa dos seguintes arquivos:
        - O planejamento detalhado do PIT: `Relatorio_Final/planejamento.txt`
        - Os resumos das atividades de ensino: `Relatorio_Final/teaching.txt`
        - Os resumos das atividades de pesquisa: `Relatorio_Final/research.txt`
        - Os resumos das atividades de extensão: `Relatorio_Final/extension.txt`
        - Os resumos das atividades administrativo-pedagógicas: `Relatorio_Final/admin.txt`
        - O relatório acadêmico final gerado: `Relatorio_Final/relatorio_academico.md`

        Sua tarefa é produzir uma avaliação estruturada da **aderência do relatório final ao PIT**, seguindo os critérios abaixo.
        A avaliação deve ser detalhada, com justificativas e, quando possível, quantificações (e.g., escalas, porcentagens).

        1.  **Cobertura de Metas (por Seção):** Para cada seção (Ensino, Pesquisa, Extensão, Administrativo-Pedagógico), liste as metas do PIT que foram claramente abordadas e as que não foram ou foram incompletamente. Atribua uma **pontuação de 0 a 10** para a cobertura geral de cada seção.
        2.  **Consistência dos Dados:** Verifique se há **discrepâncias ou inconsistências** entre o que foi prometido no `planejamento.txt` e o que foi relatado nas seções de `teaching.txt`, `research.txt`, etc., e no `relatorio_academico.md`. Destaque exemplos específicos.
        3.  **Relevância e Profundidade:** Avalie se as atividades descritas no relatório recebem a profundidade de detalhe e a ênfase adequadas, considerando sua relevância no PIT. Comente sobre áreas que poderiam ser mais elaboradas ou concisas.
        4.  **Proporção Promessa vs. Entrega:** Analise o equilíbrio geral entre o volume e a ambição das promessas do PIT e o que foi efetivamente reportado no relatório final. Indique se houve superação ou deficiência em relação ao planejado.

        A avaliação deve ser formatada em **Markdown**, com cabeçalhos para cada critério e seções. Inclua exemplos de texto do relatório ou do PIT para justificar suas avaliações.
        O resultado final desta avaliação detalhada deve ser salvo no arquivo `Relatorio_Final/avaliacao_metrica.md`.
    """),
    expected_output="Arquivo `Relatorio_Final/avaliacao_metrica.md` com uma análise detalhada em Markdown, avaliando a aderência do relatório ao PIT com base em cobertura, consistência, relevância e proporção promessa vs. entrega.",
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