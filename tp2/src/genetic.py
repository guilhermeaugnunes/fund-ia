"""Algoritmo Genético para otimização de permutações.

Projetado conforme as diretrizes do Trabalho Prático 2 de Fundamentos de IA (UFMG):
- Arquitetura desacoplada: Algoritmo != Problema (Seção 2.1).
- Orçamento estrito de avaliações com ContadorAvaliacoes (Seção 3 e 6).
- Operadores: Seleção por Torneio / Roleta, Order Crossover (OX), Mutação por Troca (Swap) e Elitismo.
- Monitoramento de diversidade da população e curva da média de fitness da população.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

from local_search import ContadorAvaliacoes, HistoricoMelhor


def cruzamento_ox(pai1: list[int], pai2: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    """Order Crossover (OX) para permutações de elementos únicos.
    
    Preserva a ordem e a posição relativa de subsequências, garantindo que nenhum
    ativo seja duplicado ou omitido. Gera dois filhos complementares.
    """
    n = len(pai1)
    if n < 2:
        return pai1.copy(), pai2.copy()

    # Seleciona dois pontos de corte aleatórios distintos
    pt1, pt2 = sorted(rng.sample(range(n), 2))

    def _ox_unidirecional(p1: list[int], p2: list[int]) -> list[int]:
        filho = [None] * n
        # Copia o segmento do primeiro pai
        filho[pt1 : pt2 + 1] = p1[pt1 : pt2 + 1]
        presentes = set(filho[pt1 : pt2 + 1])

        # Preenche as posições restantes a partir de p2 após pt2
        idx_filho = (pt2 + 1) % n
        idx_p2 = (pt2 + 1) % n

        for _ in range(n):
            alelo = p2[idx_p2]
            if alelo not in presentes:
                filho[idx_filho] = alelo
                presentes.add(alelo)
                idx_filho = (idx_filho + 1) % n
            idx_p2 = (idx_p2 + 1) % n

        return filho  # type: ignore

    filho1 = _ox_unidirecional(pai1, pai2)
    filho2 = _ox_unidirecional(pai2, pai1)
    return filho1, filho2


def mutacao_swap(solucao: list[int], taxa_mutacao: float, rng: random.Random) -> list[int]:
    """Aplica mutação por troca (Swap) com probabilidade taxa_mutacao."""
    mutado = solucao.copy()
    if rng.random() < taxa_mutacao:
        n = len(mutado)
        if n >= 2:
            i, j = rng.sample(range(n), 2)
            mutado[i], mutado[j] = mutado[j], mutado[i]
    return mutado


def selecao_torneio(populacao: list[tuple[list[int], float]], tamanho_torneio: int, rng: random.Random) -> list[int]:
    """Seleção por torneio para minimização (menor custo vence).
    
    A seleção por torneio depende apenas da ordem relativa dos custos, sendo invariante
    a translações ou escalas da função h, preservando a pressão seletiva sem distorções.
    """
    competidores = rng.sample(populacao, min(tamanho_torneio, len(populacao)))
    vencedor = min(competidores, key=lambda ind: ind[1])
    return vencedor[0]


def selecao_roleta_minimizacao(populacao: list[tuple[list[int], float]], rng: random.Random) -> list[int]:
    """Seleção proporcional à aptidão (roleta) adaptada para minimização.
    
    Converte o custo h em aptidão positiva invertida: fitness_i = max(h) - h_i + eps.
    """
    custos = [c for _, c in populacao]
    max_c = max(custos)
    min_c = min(custos)
    delta = max_c - min_c
    eps = (delta / len(populacao)) if delta > 0 else 1.0

    aptidoes = [(max_c - c) + eps for c in custos]
    total_aptidao = sum(aptidoes)

    limiar = rng.random() * total_aptidao
    acumulada = 0.0
    for (individuo, _), apt in zip(populacao, aptidoes):
        acumulada += apt
        if acumulada >= limiar:
            return individuo
    return populacao[-1][0]


def calcular_diversidade(populacao: list[tuple[list[int], float]]) -> int:
    """Calcula o número de cromossomos únicos na população."""
    return len(set(tuple(ind) for ind, _ in populacao))


def genetic_algorithm(
    problema: Any,
    orcamento: int,
    tamanho_populacao: int = 100,
    taxa_cruzamento: float = 0.9,
    taxa_mutacao: float = 0.1,
    metodo_selecao: str = "torneio",
    tamanho_torneio: int = 3,
    elitismo: int = 1,
    seed: int | None = None,
    intervalo_historico: int = 1000,
) -> dict[str, Any]:
    """Algoritmo Genético com orçamento exato e monitoramento populacional.

    Parâmetros:
      - problema: objeto com métodos solucao_aleatoria() e avaliar(solucao).
      - orcamento: número máximo estrito de avaliações de h permitidas.
      - tamanho_populacao: tamanho da população N (default: 100).
      - taxa_cruzamento: probabilidade de cruzamento Pc (default: 0.9).
      - taxa_mutacao: probabilidade de mutação Pm por indivíduo (default: 0.1).
      - metodo_selecao: 'torneio' ou 'roleta'.
      - tamanho_torneio: k competidores no torneio (default: 3).
      - elitismo: número de melhores indivíduos preservados para a próxima geração (default: 1).
      - seed: semente pseudoaleatória para reprodutibilidade.
      - intervalo_historico: frequência de amostragem no histórico de convergência.
    """
    if tamanho_populacao < 2:
        raise ValueError("tamanho_populacao deve ser >= 2")
    if not (0.0 <= taxa_cruzamento <= 1.0):
        raise ValueError("taxa_cruzamento deve estar entre 0.0 e 1.0")
    if not (0.0 <= taxa_mutacao <= 1.0):
        raise ValueError("taxa_mutacao deve estar entre 0.0 e 1.0")
    if metodo_selecao not in ("torneio", "roleta"):
        raise ValueError("metodo_selecao deve ser 'torneio' ou 'roleta'")
    if tamanho_torneio < 1:
        raise ValueError("tamanho_torneio deve ser >= 1")
    if elitismo < 0:
        raise ValueError("elitismo deve ser >= 0")

    rng = random.Random(seed)
    contador = ContadorAvaliacoes(problema, orcamento)
    hist = HistoricoMelhor(intervalo_historico)

    # 1. INICIALIZAÇÃO DA POPULAÇÃO
    # Para orçamentos muito pequenos, restringe a população inicial ao orçamento restante
    pop_size = min(tamanho_populacao, contador.disponiveis)
    populacao: list[tuple[list[int], float]] = []

    melhor_solucao: list[int] = []
    melhor_fitness = math.inf

    for _ in range(pop_size):
        solucao = sorted(problema.solucao_aleatoria())
        rng.shuffle(solucao)
        custo = contador.avaliar(solucao)
        populacao.append((solucao, custo))

        if custo < melhor_fitness:
            melhor_solucao, melhor_fitness = solucao.copy(), custo
        hist.registrar(contador, melhor_fitness)

    if not populacao:
        raise RuntimeError("Orçamento insuficiente para inicializar a população.")

    # Registro de métricas populacionais ao longo das gerações
    curva_populacao: list[dict[str, Any]] = []
    geracoes = 0

    def registrar_estatisticas_geracao(ger: int):
        custos = [c for _, c in populacao]
        media_h = statistics.mean(custos)
        div = calcular_diversidade(populacao)
        curva_populacao.append({
            "geracao": ger,
            "avaliacoes": contador.avaliacoes,
            "melhor_fitness": melhor_fitness,
            "media_fitness_populacao": media_h,
            "cromossomos_distintos": div,
            "taxa_diversidade": div / len(populacao),
        })

    registrar_estatisticas_geracao(geracoes)

    # 2. CICLO EVOLUTIVO
    while contador.disponiveis > 0:
        geracoes += 1

        # Ordena a população por aptidão (menor custo primeiro) para elitismo
        populacao.sort(key=lambda ind: ind[1])
        nova_populacao: list[tuple[list[int], float]] = []

        # Aplica Elitismo: preserva os E melhores diretamente
        n_elite = min(elitismo, len(populacao))
        for i in range(n_elite):
            nova_populacao.append(populacao[i])

        # Gera descendentes até completar a nova população
        while len(nova_populacao) < len(populacao) and contador.disponiveis > 0:
            # Seleção de dois pais
            if metodo_selecao == "torneio":
                pai1 = selecao_torneio(populacao, tamanho_torneio, rng)
                pai2 = selecao_torneio(populacao, tamanho_torneio, rng)
            else:
                pai1 = selecao_roleta_minimizacao(populacao, rng)
                pai2 = selecao_roleta_minimizacao(populacao, rng)

            # Cruzamento (OX)
            if rng.random() < taxa_cruzamento:
                filho1, filho2 = cruzamento_ox(pai1, pai2, rng)
            else:
                filho1, filho2 = pai1.copy(), pai2.copy()

            # Mutação (Swap)
            filho1 = mutacao_swap(filho1, taxa_mutacao, rng)
            filho2 = mutacao_swap(filho2, taxa_mutacao, rng)

            # Avaliação dos filhos respeitando estritamente o orçamento
            for filho in (filho1, filho2):
                if len(nova_populacao) < len(populacao) and contador.disponiveis > 0:
                    custo_filho = contador.avaliar(filho)
                    nova_populacao.append((filho, custo_filho))

                    if custo_filho < melhor_fitness:
                        melhor_solucao, melhor_fitness = filho.copy(), custo_filho
                    hist.registrar(contador, melhor_fitness)
                else:
                    break

        populacao = nova_populacao
        registrar_estatisticas_geracao(geracoes)

    hist.registrar(contador, melhor_fitness, forcar=True)

    custos_finais = [c for _, c in populacao]
    return {
        "algoritmo": "genetic_algorithm",
        "seed": seed,
        "melhor_solucao": melhor_solucao,
        "melhor_fitness": melhor_fitness,
        "avaliacoes": contador.avaliacoes,
        "historico": hist.dados,
        "metricas": {
            "tamanho_populacao": len(populacao),
            "taxa_cruzamento": taxa_cruzamento,
            "taxa_mutacao": taxa_mutacao,
            "metodo_selecao": metodo_selecao,
            "tamanho_torneio": tamanho_torneio,
            "elitismo": elitismo,
            "geracoes_completadas": geracoes,
            "diversidade_final": calcular_diversidade(populacao),
            "media_fitness_final": statistics.mean(custos_finais),
            "desvio_fitness_final": statistics.stdev(custos_finais) if len(custos_finais) > 1 else 0.0,
            "curva_populacao": curva_populacao,
        },
    }
