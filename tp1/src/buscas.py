from collections import deque
import heapq

from model import Estado, No
from sucessor import gerar_sucessores

OBJETIVO: Estado = (True, True, True, True, True)


def busca_bfs(estado_inicial: Estado) -> No | None:
    """Busca em Largura (BFS) - Usa fila (FIFO)"""
    no_raiz = No(estado=estado_inicial)
    fronteira = deque([no_raiz])
    visitados = {estado_inicial}

    while fronteira:
        # tira sempre do começo da fila (FIFO)
        no_atual = fronteira.popleft()

        # teste de objetivo:
        if no_atual.estado == OBJETIVO:
            return no_atual

        # expansão
        for filho in gerar_sucessores(no_atual):
            if filho.estado not in visitados:
                visitados.add(filho.estado)
                fronteira.append(filho)  # entra no final da fila

    return None


def busca_dfs(estado_inicial: Estado) -> No | None:
    """Busca em Profundidade (BFS) - Usa pilha (LIFO)"""
    no_raiz = No(estado=estado_inicial)
    fronteira = [no_raiz]
    visitados = {estado_inicial}

    while fronteira:
        # tira sempre do final da fila (LIFO)
        no_atual = fronteira.pop()

        # teste de objetivo:
        if no_atual.estado == OBJETIVO:
            return no_atual

        # expansão
        for filho in gerar_sucessores(no_atual):
            if filho.estado not in visitados:
                visitados.add(filho.estado)
                fronteira.append(filho)  # entra no TOPO da fila

    return None


def busca_custo_minimo(estado_inicial: Estado) -> No | None:
    """Busca de Custo Mínimo (Custo Uniforme / UCS) - Usa fila de prioridade"""
    no_raiz = No(estado=estado_inicial)
    contador = 0
    fronteira = [(no_raiz.custo_acumulado, contador, no_raiz)]
    melhor_custo = {estado_inicial: 0}

    while fronteira:
        # tira sempre o nó com menor custo acumulado da fila de prioridade
        custo_atual, _, no_atual = heapq.heappop(fronteira)

        # se já encontramos um caminho melhor para esse estado, descarta
        if custo_atual > melhor_custo.get(no_atual.estado, float("inf")):
            continue

        # teste de objetivo:
        if no_atual.estado == OBJETIVO:
            return no_atual

        # expansão
        for filho in gerar_sucessores(no_atual):
            if (
                filho.estado not in melhor_custo
                or filho.custo_acumulado < melhor_custo[filho.estado]
            ):
                melhor_custo[filho.estado] = filho.custo_acumulado
                contador += 1
                heapq.heappush(fronteira, (filho.custo_acumulado, contador, filho))

    return None
