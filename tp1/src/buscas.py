from collections import deque

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
