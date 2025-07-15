from crewai import Agent, Task, LLM, Crew, Process
from textwrap import dedent
from custom_pit import VerificarPalavraNoPDFTool
import os

# Configuração do LLM
client = LLM(
    model='openai/gpt-3.5-turbo',
    api_key='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'
)

# Ferramentas
verificar_palavra_no_pdf = VerificarPalavraNoPDFTool()

# Agente para verificar seções aplicáveis
section_analyzer = Agent(
    name="Analisador de Seções",
    role="Especialista em Análise de Aplicabilidade de Seções do PIT",
    goal=(
        "Analisar o planejamento do PIT e determinar quais seções são aplicáveis para processamento. "
        "Ensino é sempre obrigatório. Outras seções (Pesquisa, Extensão, Administrativo) só devem ser "
        "processadas se houver conteúdo específico planejado no PIT."
    ),
    backstory="Especialista em análise documental que identifica com precisão quais seções do relatório devem ser processadas baseado no conteúdo do PIT.",
    verbose=True,
    llm=client
)

planner = Agent(
    name="Planejador Acadêmico",
    role="Estruturador e Organizador Mestre do Relatório Acadêmico",
    goal=(
        "Analisar profundamente o Plano Individual de Trabalho (PIT) para identificar **todas as metas, atividades e entregas prometidas** pelo docente. "
        "Estruturar essas informações de forma lógica e categorizada por área (Ensino, Pesquisa, Extensão, Administrativo-Pedagógicas e Complementos/Observações). "
        "IMPORTANTE: Identificar quais seções têm conteúdo no PIT para determinar o que deve ser processado no relatório final. "
        "A seção de Ensino é OBRIGATÓRIA e sempre deve ser processada, mesmo que não haja documentos na pasta ENSINO. "
        "**Não invente informações, utilize apenas o que está explicitamente no PIT. Sempre que possível, cite trechos do documento.**"
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
            "Analisar as atividades de ensino sempre, independentemente da existência de documentos na pasta ENSINO. "
            "Se houver documentos na pasta ENSINO, priorizar essas informações e fazer merge com o planejamento. "
            "Se não houver documentos, basear-se exclusivamente no planejamento do PIT para criar um relatório "
            "confirmando que as atividades de ensino foram executadas conforme planejado, listando disciplinas e atividades. "
            "**Não invente informações, utilize apenas dados do planejamento PIT e/ou documentos da pasta ENSINO.**"
        ),
        backstory="Um veterano em educação, com expertise em pedagogia e análise de currículos. Seu foco é desvendar a essência das atividades de ensino e seu real impacto.",
        verbose=True,
        llm=client
    ),
    "research": Agent(
        name="Pesquisador de Pesquisa",
        role="Especialista em Produção e Análise Científica com Foco em Impacto",
        goal=(
            "Processar atividades de pesquisa APENAS se houver conteúdo na seção de pesquisa do PIT. "
            "Se houver documentos na pasta PESQUISA, priorizar essas informações e fazer merge com o planejamento. "
            "Se não houver documentos na pasta, basear-se exclusivamente no planejamento do PIT. "
            "**Não invente informações, utilize apenas dados do planejamento PIT e/ou documentos da pasta PESQUISA.**"
        ),
        backstory="Um pesquisador experiente, com olhar apurado para a produção acadêmica e suas métricas de impacto. Ele garante que cada descoberta seja devidamente documentada.",
        verbose=True,
        llm=client
    ),
    "extension": Agent(
        name="Pesquisador de Extensão",
        role="Analista de Projetos de Extensão Universitária e Engajamento Comunitário",
        goal=(
            "Processar atividades de extensão APENAS se houver conteúdo na seção de extensão do PIT. "
            "Se houver documentos na pasta EXTENSAO, priorizar essas informações e fazer merge com o planejamento. "
            "Se não houver documentos na pasta, basear-se exclusivamente no planejamento do PIT. "
            "**Não invente informações, utilize apenas dados do planejamento PIT e/ou documentos da pasta EXTENSAO.**"
        ),
        backstory="Com um histórico em projetos sociais e universitários, este agente é perito em traduzir as ações de extensão em narrativas de impacto real na comunidade.",
        verbose=True,
        llm=client
    ),
    "admin": Agent(
        name="Pesquisador Administrativo",
        role="Analista de Processos Administrativo-Pedagógicos e Governança Acadêmica",
        goal=(
            "Processar atividades administrativo-pedagógicas APENAS se houver conteúdo na seção administrativa do PIT. "
            "Se houver documentos na pasta ADMINISTRATIVO_PEDAGOGICO, priorizar essas informações e fazer merge com o planejamento. "
            "Se não houver documentos na pasta, basear-se exclusivamente no planejamento do PIT. "
            "**Não invente informações, utilize apenas dados do planejamento PIT e/ou documentos da pasta ADMINISTRATIVO_PEDAGOGICO.**"
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
        - Atividades de Ensino (OBRIGATÓRIO - sempre tem conteúdo)
        - Atividades de Pesquisa (OPCIONAL - só processar se houver conteúdo no PIT)
        - Atividades de Extensão (OPCIONAL - só processar se houver conteúdo no PIT)
        - Atividades Administrativo-Pedagógicas (OPCIONAL - só processar se houver conteúdo no PIT)
        - Complemento/Observações
        
        Para cada seção, indique claramente:
        1. Se a seção tem conteúdo no PIT (SIM/NÃO)
        2. Se tem conteúdo, liste as atividades/metas planejadas
        3. Para Ensino: sempre listar as disciplinas ministradas
        
        FORMATO OBRIGATÓRIO para o arquivo `Relatorio_Final/planejamento.txt`:
        
        # ANÁLISE DO PIT
        
        ## ENSINO: SIM
        - Lista das disciplinas/atividades de ensino encontradas no PIT
        
        ## PESQUISA: SIM/NÃO
        - Se SIM: lista das atividades de pesquisa encontradas no PIT
        - Se NÃO: indicar que não há conteúdo
        
        ## EXTENSÃO: SIM/NÃO
        - Se SIM: lista das atividades de extensão encontradas no PIT
        - Se NÃO: indicar que não há conteúdo
        
        ## ADMINISTRATIVO: SIM/NÃO
        - Se SIM: lista das atividades administrativo-pedagógicas encontradas no PIT
        - Se NÃO: indicar que não há conteúdo
        
        **Não invente informações, utilize apenas o que está explicitamente no PIT.**
    """),
    expected_output="Arquivo `./Relatorio_Final/planejamento.txt` contendo as promessas do PIT organizadas por seção, indicando quais seções têm conteúdo para processamento.",
    agent=planner,
    output_file="./Relatorio_Final/planejamento.txt",
    tools=[verificar_palavra_no_pdf]
)

# Task para análise de seções aplicáveis
section_analysis_task = Task(
    description=dedent("""
        Com base no arquivo `Relatorio_Final/planejamento.txt`, determine quais seções devem ser processadas:
        
        1. **ENSINO**: Sempre aplicável (obrigatório)
        2. **PESQUISA**: Aplicável apenas se houver atividades de pesquisa planejadas no PIT
        3. **EXTENSÃO**: Aplicável apenas se houver atividades de extensão planejadas no PIT  
        4. **ADMINISTRATIVO**: Aplicável apenas se houver atividades administrativo-pedagógicas planejadas no PIT
        
        INSTRUÇÕES:
        - Leia o arquivo `Relatorio_Final/planejamento.txt` 
        - Procure pelas seções "## ENSINO:", "## PESQUISA:", "## EXTENSÃO:", "## ADMINISTRATIVO:"
        - Se uma seção mostra "SIM", ela deve ser processada
        - Se uma seção mostra "NÃO", ela não deve ser processada
        
        Crie um arquivo `Relatorio_Final/secoes_aplicaveis.txt` com o seguinte formato:
        ENSINO: SIM
        PESQUISA: SIM/NÃO
        EXTENSAO: SIM/NÃO
        ADMINISTRATIVO: SIM/NÃO
        
        Baseie-se EXCLUSIVAMENTE no conteúdo do planejamento. Não invente informações.
    """),
    expected_output="Arquivo `Relatorio_Final/secoes_aplicaveis.txt` indicando quais seções devem ser processadas.",
    agent=section_analyzer,
    output_file="Relatorio_Final/secoes_aplicaveis.txt",
    tools=[]
)


research_tasks = []
sections = {
    "teaching": "./ENSINO/ensino.pdf",
    "research": "./PESQUISA/pesquisa.pdf", 
    "extension": "./EXTENSAO/extensao.pdf",
    "admin": "./ADMINISTRATIVO_PEDAGOGICO/admin.pdf"
}

# Task específica para Ensino (sempre executada)
research_tasks.append(Task(
    description=dedent("""
        ENSINO é uma seção OBRIGATÓRIA e sempre deve ser processada.
        
        1. Primeiro, leia o planejamento em `Relatorio_Final/planejamento.txt` e extraia as informações de ensino.
        2. Verifique se existe o arquivo `./ENSINO/ensino.pdf` e se tem conteúdo.
        3. Se houver conteúdo na pasta ENSINO, PRIORIZE essas informações e faça merge com o planejamento.
        4. Se não houver conteúdo na pasta ENSINO, baseie-se EXCLUSIVAMENTE no planejamento do PIT.
        
        Seu relatório deve:
        - Listar todas as disciplinas ministradas (sempre presente no PIT)
        - Confirmar que as atividades foram executadas conforme planejado
        - Se houver documentos na pasta, detalhar evidências e resultados específicos
        - Manter tom de confirmação: "As atividades de ensino foram executadas conforme planejado..."
        
        **Não invente informações. Use apenas dados do planejamento PIT e/ou documentos da pasta ENSINO.**
    """),
    expected_output="Arquivo `Relatorio_Final/teaching.txt` com relatório completo de ensino, baseado no planejamento e/ou documentos da pasta.",
    agent=research_agents["teaching"],
    output_file="Relatorio_Final/teaching.txt",
    tools=[verificar_palavra_no_pdf]
))

# Tasks condicionais para outras seções
for section, file_path in [(k, v) for k, v in sections.items() if k != "teaching"]:
    research_tasks.append(Task(
        description=dedent(f"""
            Esta seção ({section.upper()}) é OPCIONAL e deve ser processada APENAS se houver conteúdo no PIT.
            
            1. Primeiro, leia o planejamento em `Relatorio_Final/planejamento.txt` e verifique se a seção de {section.upper()} tem conteúdo.
            2. Se NÃO houver conteúdo no PIT para {section.upper()}, escreva: "Seção não aplicável - sem atividades planejadas no PIT."
            3. Se houver conteúdo no PIT:
               a. Verifique se existe o arquivo `{file_path}` e se tem conteúdo
               b. Se houver documentos na pasta, PRIORIZE essas informações e faça merge com o planejamento
               c. Se não houver documentos na pasta, baseie-se EXCLUSIVAMENTE no planejamento do PIT
            
            **Não invente informações. Use apenas dados do planejamento PIT e/ou documentos da pasta {section.upper()}.**
        """),
        expected_output=f"Arquivo `Relatorio_Final/{section}.txt` com relatório da seção {section.upper()} (se aplicável) ou indicação de não aplicabilidade.",
        agent=research_agents[section],
        output_file=f"Relatorio_Final/{section}.txt",
        tools=[verificar_palavra_no_pdf]
    ))

writing_task = Task(
    description=dedent("""
        Sua missão é compilar o relatório acadêmico final baseado nos seguintes arquivos:
        - Relatorio_Final/planejamento.txt (promessas do PIT)
        - Relatorio_Final/secoes_aplicaveis.txt (quais seções processar)
        - Relatorio_Final/teaching.txt (sempre presente - ensino é obrigatório)
        - Relatorio_Final/research.txt (condicional)
        - Relatorio_Final/extension.txt (condicional)
        - Relatorio_Final/admin.txt (condicional)

        REGRAS IMPORTANTES:
        1. **Consulte primeiro** o arquivo `secoes_aplicaveis.txt` para saber quais seções incluir
        2. **Ensino**: Sempre incluir seção completa (é obrigatório)
        3. **Outras seções**: Só incluir se marcadas como "SIM" no arquivo de seções aplicáveis
        4. **Não invente informações**: Use apenas o que está nos arquivos
        5. **Priorize documentos das pastas**: Quando há informações do PIT + documentos das pastas, priorize os documentos mas mencione o planejamento

        Estruture o relatório em Markdown com:
        - Introdução contextualizando o período
        - Seção de Ensino (sempre presente)
        - Seções condicionais (Pesquisa, Extensão, Administrativo) apenas se aplicáveis
        - Conclusão resumindo as principais realizações
        
        Para cada seção incluída, faça referência explícita ao que foi planejado no PIT e o que foi executado.
    """),
    expected_output="Arquivo `Relatorio_Final/relatorio_academico.md` com relatório completo, incluindo apenas seções aplicáveis baseadas no PIT.",
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
    agents=[planner, section_analyzer, *research_agents.values(), writer, reviewer_research, reviewer_report, evaluator_textual, evaluator_metrics],
    tasks=[planning_task, section_analysis_task, *research_tasks, writing_task, review_final_report_task, textual_review_task,
        academic_metrics_task],
    process=Process.sequential,
    verbose=True,
    llm=client
)

if __name__ == "__main__":
    crew.kickoff()