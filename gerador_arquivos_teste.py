import os

# Estrutura de diretórios
folders = [
    "Planejamento",
    "ENSINO",
    "PESQUISA",
    "EXTENSAO",
    "ADMINISTRATIVO_PEDAGOGICO",
    "Relatorio_Final"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Conteúdos de teste
files_content = {
    "Planejamento/PIT.pdf": """Metas e atividades planejadas:

Ensino:
- Ministrar as disciplinas X e Y no curso de Z.
- Participar de reuniões pedagógicas.

Pesquisa:
- Conduzir projeto de iniciação científica sobre inteligência artificial.
- Submeter artigo para conferência ABC.

Extensão:
- Coordenar projeto de extensão “Tecnologia e Comunidade”.
- Organizar oficinas abertas ao público.

Administrativo-Pedagógicas:
- Participar de colegiados e comissões.
- Apoiar organização de eventos acadêmicos.

Observações:
- Algumas atividades estão condicionadas a liberação orçamentária.
""",
    "ENSINO/ensino.txt": """- Disciplinas X e Y foram ministradas.
- Participação nas reuniões pedagógicas documentada.
- Participação extra em bancas de TCC.""",
    "PESQUISA/pesquisa.txt": """- Projeto de iniciação científica iniciado e em andamento.
- Artigo submetido à conferência ABC.
- Apresentação em evento local não previsto inicialmente.""",
    "EXTENSAO/extensao.txt": """- Projeto “Tecnologia e Comunidade” realizado com 3 oficinas.
- Oficinas abertas ao público realizadas em parceria com ONG.""",
    "ADMINISTRATIVO_PEDAGOGICO/admin.txt": """- Participação em 4 reuniões de colegiado.
- Colaboração na organização do Seminário Interno 2024."""
}

for path, content in files_content.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Arquivos de teste criados com sucesso!")
