from itertools import combinations

from model import No, Tempos


def gerar_sucessores(no_atual: No) -> list[No]:
    sucessores = []
    estado_atual = no_atual.estado
    lado_tocha = estado_atual[4]  # false = origem, true = destino

    # identifica quais indices estão do lado da tocha
    pessoas_disponiveis = []
    for i in range(4):
        if estado_atual[i] == lado_tocha:
            pessoas_disponiveis.append(i)

    # monta as possibilidades da travessia
    movimentos = []

    # possibilidade de ir 1 pessoa sozinha
    for p in pessoas_disponiveis:
        movimentos.append([p])

    # possibilidade de irem 2 pessoas juntas
    for p1, p2 in combinations(pessoas_disponiveis, 2):
        movimentos.append([p1, p2])

    # executa cada uma movimento possível para gerar novos nós
    for movimento in movimentos:
        # transforma a tupla em lista temporariamente para alterarmos os valores
        novo_estado = list(estado_atual)

        # inverte o lado da tocha
        novo_estado[4] = not lado_tocha

        # inverte o lado das pessoas que estão atravessando e descobre o custo do movimento
        custo_movimento = 0
        nome_pessoas = []
        for p in movimento:
            novo_estado[p] = not lado_tocha
            # regra: o tempo de travessia é sempre o da pessoa mais lenta
            custo_movimento = max(custo_movimento, Tempos[p])
            nome_pessoas.append(str(Tempos[p]))

        # formata a ação em texto para histórico
        direcao = "ida" if not lado_tocha else "volta"
        texto_acao = f"Pessoas {', '.join(nome_pessoas)} atravessam {direcao}"

        # junta tudo em um novo nó e adiciona à lista
        novo_no = No(
            estado=tuple(novo_estado),  # converte de volta para tupla
            pai=no_atual,
            acao=texto_acao,
            custo_acumulado=no_atual.custo_acumulado + custo_movimento,
        )
        sucessores.append(novo_no)

    return sucessores
