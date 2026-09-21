import heapq
from collections import deque

from model import Estado, No, ResultadoBusca, Tempos
from sucessor import gerar_sucessores

OBJETIVO: Estado = (True, True, True, True, True)
LIMITE_EXPANSOES_PADRAO = (
    100_000  # limite recomendado no enunciado pra evitar s buscas longas demais
)


def busca_bfs(
    estado_inicial: Estado, limite_expansoes: int = LIMITE_EXPANSOES_PADRAO
) -> ResultadoBusca:
    """Busca em Largura (BFS) - Usa fila (FIFO)"""
    no_raiz = No(estado=estado_inicial)
    fronteira = deque([no_raiz])
    visitados = {estado_inicial}
    nos_expandidos = 0

    while fronteira:
        # tira sempre do começo da fila (FIFO)
        no_atual = fronteira.popleft()

        # teste de objetivo:
        if no_atual.estado == OBJETIVO:
            return ResultadoBusca(no_atual, nos_expandidos)

        if nos_expandidos >= limite_expansoes:
            return ResultadoBusca(None, nos_expandidos, limite_atingido=True)

        # expansão
        # só conta como expandido se sucessor for gerado
        nos_expandidos += 1
        for filho in gerar_sucessores(no_atual):
            if filho.estado not in visitados:
                visitados.add(filho.estado)
                fronteira.append(filho)  # entra no final da fila

    return ResultadoBusca(None, nos_expandidos)


def busca_dfs(
    estado_inicial: Estado, limite_expansoes: int = LIMITE_EXPANSOES_PADRAO
) -> ResultadoBusca:
    """Busca em Profundidade (DFS) - Usa pilha (LIFO)"""
    no_raiz = No(estado=estado_inicial)
    fronteira = [no_raiz]
    visitados = {estado_inicial}
    nos_expandidos = 0

    while fronteira:
        # tira sempre do final da fila (LIFO)
        no_atual = fronteira.pop()

        # teste de objetivo:
        if no_atual.estado == OBJETIVO:
            return ResultadoBusca(no_atual, nos_expandidos)

        if nos_expandidos >= limite_expansoes:
            return ResultadoBusca(None, nos_expandidos, limite_atingido=True)

        nos_expandidos += 1
        # expansão
        for filho in gerar_sucessores(no_atual):
            if filho.estado not in visitados:
                visitados.add(filho.estado)
                fronteira.append(filho)  # entra no TOPO da fila

    return ResultadoBusca(None, nos_expandidos)


def busca_custo_minimo(
    estado_inicial: Estado, limite_expansoes: int = LIMITE_EXPANSOES_PADRAO
) -> ResultadoBusca:
    """Busca de Custo Mínimo (Custo Uniforme / UCS) - Usa fila de prioridade"""
    no_raiz = No(estado=estado_inicial)
    contador = 0
    fronteira = [(no_raiz.custo_acumulado, contador, no_raiz)]
    melhor_custo = {estado_inicial: 0}
    nos_expandidos = 0

    while fronteira:
        # tira sempre o nó com menor custo acumulado da fila de prioridade
        custo_atual, _, no_atual = heapq.heappop(fronteira)

        # se já encontramos um caminho melhor para esse estado, descarta
        if custo_atual > melhor_custo.get(no_atual.estado, float("inf")):
            continue

        # teste de objetivo:
        if no_atual.estado == OBJETIVO:
            return ResultadoBusca(no_atual, nos_expandidos)

        if nos_expandidos >= limite_expansoes:
            return ResultadoBusca(None, nos_expandidos, limite_atingido=True)

        nos_expandidos += 1
        # expansão
        for filho in gerar_sucessores(no_atual):
            if (
                filho.estado not in melhor_custo
                or filho.custo_acumulado < melhor_custo[filho.estado]
            ):
                melhor_custo[filho.estado] = filho.custo_acumulado
                contador += 1
                heapq.heappush(fronteira, (filho.custo_acumulado, contador, filho))

    return ResultadoBusca(None, nos_expandidos)


def heuristica(estado: Estado) -> int:
    """Calcula a heurística admissível para o problema"""
    pessoas_na_origem = []

    # pega apenas os índices de 0 a 3, ignorando o índice 4 que representa a tocha
    for i in range(4):
        if estado[i] is False:
            pessoas_na_origem.append(Tempos[i])

    if not pessoas_na_origem:
        return 0

    # retorna o tempo da pessoa mais lenta que ainda está na origem (false)
    return max(pessoas_na_origem)


def busca_a_estrela(
    estado_inicial: Estado, limite_expansoes: int = LIMITE_EXPANSOES_PADRAO
) -> ResultadoBusca:
    """Busca A* - Usa fila de prioridade ordenada pelo tempo gasto somado à previsão de tempo restante"""
    no_raiz = No(estado=estado_inicial)
    contador = 0

    h_inicial = heuristica(estado_inicial)
    f_inicial = no_raiz.custo_acumulado + h_inicial

    fronteira = [(f_inicial, contador, no_raiz)]

    melhor_custo = {estado_inicial: no_raiz.custo_acumulado}
    nos_expandidos = 0

    while fronteira:
        # sempre tira o nó com a menor estimativa de tempo total da fila de prioridade
        f_atual, _, no_atual = heapq.heappop(fronteira)

        # se já encontrar um caminho com custo real menor, descarta
        if no_atual.custo_acumulado > melhor_custo.get(no_atual.estado, float("inf")):
            continue

        # teste de objetivo:
        if no_atual.estado == OBJETIVO:
            return ResultadoBusca(no_atual, nos_expandidos)

        if nos_expandidos >= limite_expansoes:
            return ResultadoBusca(None, nos_expandidos, limite_atingido=True)

        # expansão
        nos_expandidos += 1
        for filho in gerar_sucessores(no_atual):
            # verifica se nunca esteve nesse estado ou se o caminho atual levou menos tempo do que o anterior
            if (
                filho.estado not in melhor_custo
                or filho.custo_acumulado < melhor_custo[filho.estado]
            ):
                melhor_custo[filho.estado] = filho.custo_acumulado

                # calcula a nova previsão de tempo total para o nó filho
                h_filho = heuristica(filho.estado)
                f_filho = filho.custo_acumulado + h_filho

                contador += 1
                heapq.heappush(fronteira, (f_filho, contador, filho))
    return ResultadoBusca(None, nos_expandidos)
