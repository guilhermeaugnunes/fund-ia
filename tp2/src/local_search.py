# Busca local independente da implementação do problema

from __future__ import annotations

import math
import random
import statistics
from typing import Any


OPERADORES = ("swap", "insertion", "2opt")
ESTRATEGIAS = ("best", "first")


class ContadorAvaliacoes:
    """Aplica o mesmo orçamento à avaliação normal e à incremental."""

    def __init__(self, problema: Any, orcamento: int):
        if not isinstance(orcamento, int) or isinstance(orcamento, bool) or orcamento < 1:
            raise ValueError("O orçamento deve ser inteiro positivo")
        self.problema = problema
        self.orcamento = orcamento
        self.avaliacoes = 0

    def avaliar(self, solucao: list[int]) -> float:
        if self.avaliacoes >= self.orcamento:
            raise RuntimeError("Orçamento de avaliações esgotado")
        valor = self.problema.avaliar(solucao)
        self.avaliacoes += 1
        return float(valor)

    def registrar_incremental(self, valor: float) -> float:
        if self.avaliacoes >= self.orcamento:
            raise RuntimeError("Orçamento de avaliações esgotado")
        self.avaliacoes += 1
        return float(valor)

    @property
    def disponiveis(self) -> int:
        return self.orcamento - self.avaliacoes

# Histórico compacto
class HistoricoMelhor:
    def __init__(self, intervalo: int = 1000):
        if intervalo < 1:
            raise ValueError("intervalo deve ser positivo")
        self.intervalo = intervalo
        self.dados: list[tuple[int, float]] = []

    def registrar(self, contador: ContadorAvaliacoes, melhor: float, forcar: bool = False):
        n = contador.avaliacoes
        if forcar or n == 1 or n % self.intervalo == 0 or n == contador.orcamento:
            if self.dados and self.dados[-1][0] == n:
                self.dados[-1] = (n, float(melhor))
            else:
                self.dados.append((n, float(melhor)))


def _validar_operador(operador: str):
    if operador not in OPERADORES:
        raise ValueError(f"Operador inválido: {operador}. Use {OPERADORES}")


def _movimentos(n: int, operador: str):
    """Indices de vizinhos distintos; insertion usa (n-1)^2 movimentos.
    Na inserção de adjacentes, mover i para i-1 equivale a mover i-1 para i.
    Descartamos a primeira representação para não avaliar duplicatas.
    """
    if operador == "swap":
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j
    elif operador == "2opt":
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j
    else:
        for i in range(n):
            for j in range(n):
                if j != i and j != i - 1:
                    yield i, j


def _aplicar_movimento(solucao: list[int], operador: str, i: int, j: int) -> list[int]:
    vizinho = solucao.copy()
    if operador == "swap":
        vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
    elif operador == "2opt":
        vizinho[i:j + 1] = reversed(vizinho[i:j + 1])
    else:
        vizinho.insert(j, vizinho.pop(i))
    return vizinho


def tamanho_vizinhanca(n: int, operador: str) -> int:
    """Número de vizinhos distintoss para permutações com elementos únicos"""
    _validar_operador(operador)
    if n < 2:
        return 0
    if operador == "insertion":
        return (n - 1) ** 2
    return n * (n - 1) // 2


class AvaliacaoIncrementalSwap:
    def __init__(self, problema: Any, solucao: list[int]):
        self.problema = problema
        self.solucao = solucao
        self.tempo_prefixo = [0.0] * (len(solucao) + 1)
        self.custo_prefixo = [0.0] * (len(solucao) + 1)
        base = problema.base
        ativos = problema.dict_ativos
        for k, ativo_id in enumerate(solucao):
            a = ativos[ativo_id]
            tempo = round(math.hypot(a['x'] - base[0], a['y'] - base[1]))
            self.tempo_prefixo[k + 1] = self.tempo_prefixo[k] + tempo + a['p']
            self.custo_prefixo[k + 1] = self.custo_prefixo[k] + a['w'] * max(0.0, self.tempo_prefixo[k + 1] - a['d'])
            base = (a['x'], a['y'])

    def avaliar_troca(self, i: int, j: int) -> float:
        if i > j:
            i, j = j, i
        seq = self.solucao
        dados = self.problema.dict_ativos
        tempo = self.tempo_prefixo[i]
        custo = self.custo_prefixo[i]
        if i:
            anterior = dados[seq[i - 1]]
            ponto = (anterior['x'], anterior['y'])
        else:
            ponto = self.problema.base
        for k in range(i, len(seq)):
            ativo_id = seq[j] if k == i else seq[i] if k == j else seq[k]
            a = dados[ativo_id]
            destino = (a['x'], a['y'])
            tempo += round(math.hypot(destino[0] - ponto[0], destino[1] - ponto[1])) + a['p']
            custo += a['w'] * max(0.0, tempo - a['d'])
            ponto = destino
        return custo


def greedy_random_restart(problema: Any, orcamento: int, operador: str = "swap",
                          estrategia: str = "best", seed: int | None = None,
                          intervalo_historico: int = 1000,
                          incremental_swap: bool = True) -> dict[str, Any]:
   
    """Descida gulosa best/first com reinícios aleatórios e orçamento exato.
    Mesmo uma varredura interrompida no limite pode gerar melhorias. 
    Só contamos um ótimo local depois de terminar a varredura da vizinhança sem melhora.
    """
    _validar_operador(operador)
    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"Estratégia inválida: {estrategia}. Use {ESTRATEGIAS}")
    rng = random.Random(seed)
    contador = ContadorAvaliacoes(problema, orcamento)
    hist = HistoricoMelhor(intervalo_historico)
    melhor_global: list[int] = []
    melhor_fitness = math.inf
    passos = 0
    reinicios = 0
    otimos_locais: set[tuple[int, ...]] = set()
    valores_otimos: list[float] = []
    usando_incremental = bool(incremental_swap and operador == "swap" and
                              hasattr(problema, "dict_ativos") and hasattr(problema, "base"))

    while contador.disponiveis:
        corrente = sorted(problema.solucao_aleatoria())
        # Usa rng isolado, reproduzível e sem modificar o random global do processo
        rng.shuffle(corrente)
        custo = contador.avaliar(corrente)
        reinicios += 1
        if custo < melhor_fitness:
            melhor_global, melhor_fitness = corrente.copy(), custo
        hist.registrar(contador, melhor_fitness)

        while contador.disponiveis:
            cache = AvaliacaoIncrementalSwap(problema, corrente) if usando_incremental else None
            indices = list(_movimentos(len(corrente), operador))
            if estrategia == "first":
                rng.shuffle(indices)
            candidato = None
            custo_candidato = custo
            varredura_completa = True
            for i, j in indices:
                if not contador.disponiveis:
                    varredura_completa = False
                    break
                if cache is not None:
                    valor = contador.registrar_incremental(cache.avaliar_troca(i, j))
                    vizinho = None
                else:
                    vizinho = _aplicar_movimento(corrente, operador, i, j)
                    valor = contador.avaliar(vizinho)
                if valor < custo_candidato:
                    candidato = vizinho if vizinho is not None else _aplicar_movimento(corrente, operador, i, j)
                    custo_candidato = valor
                    if estrategia == "first":
                        if valor < melhor_fitness:
                            melhor_global, melhor_fitness = candidato.copy(), valor
                        hist.registrar(contador, melhor_fitness)
                        break
                # Na estratégia best, só registramos o melhor global após comparar
                if valor < melhor_fitness:
                    melhor_global = (vizinho if vizinho is not None else
                                     _aplicar_movimento(corrente, operador, i, j))
                    melhor_global = melhor_global.copy()
                    melhor_fitness = valor
                hist.registrar(contador, melhor_fitness)

            if candidato is not None:
                corrente, custo = candidato, custo_candidato
                passos += 1
            elif varredura_completa:
                otimos_locais.add(tuple(corrente))
                valores_otimos.append(custo)
                break
            else:
                break
        # Se orçamento acaba, while externo termina. Se ótimo local, reinicia

    hist.registrar(contador, melhor_fitness, forcar=True)
    return {
        "algoritmo": "greedy_random_restart",
        "operador": operador,
        "estrategia": estrategia,
        "seed": seed,
        "melhor_solucao": melhor_global,
        "melhor_fitness": melhor_fitness,
        "avaliacoes": contador.avaliacoes,
        "historico": hist.dados,
        "metricas": {
            "passos_melhoria": passos,
            "inicializacoes": reinicios,
            "random_restarts": max(0, reinicios - 1),
            "otimos_locais_distintos": len(otimos_locais),
            "otimos_locais_fitness": valores_otimos,
            "incremental_swap": usando_incremental,
        },
    }


def calibrar_temperatura(problema: Any, contador: ContadorAvaliacoes,
                         rng: random.Random, operador: str = "swap",
                         amostras: int = 100, aceitacao_inicial: float = 0.80,
                         aceitacao_final: float = 0.01) -> dict[str, float]:
    """Estima escala de delta h de movimentos piores, contando todas as avaliações

    Conta uma avaliação inicial e cada vizinho sorteado. Não usa orçamento extra.
    T0 = -mediana(delta h positivos)/ln(0.8); Tf = -mediana(delta h positivos)/ln(0.01).
    """
    
    _validar_operador(operador)
    if not 0 < aceitacao_final < aceitacao_inicial < 1:
        raise ValueError("Probabilidades: 0 < final < inicial < 1")
    solucao = sorted(problema.solucao_aleatoria())
    rng.shuffle(solucao)
    atual = contador.avaliar(solucao)
    melhor_amostra, melhor_amostra_fitness = solucao.copy(), atual
    diferencas = []
    for _ in range(min(amostras, contador.disponiveis)):
        vizinho = _vizinho_aleatorio(problema, solucao, operador, rng)
        custo_vizinho = contador.avaliar(vizinho)
        if custo_vizinho < melhor_amostra_fitness:
            melhor_amostra, melhor_amostra_fitness = vizinho.copy(), custo_vizinho
        if custo_vizinho > atual:
            diferencas.append(custo_vizinho - atual)
        solucao, atual = vizinho, custo_vizinho
    escala = statistics.median(diferencas) if diferencas else 1.0
    return {
        "T0": -escala / math.log(aceitacao_inicial),
        "Tf": -escala / math.log(aceitacao_final),
        "escala_delta": escala,
        "amostras_piores": len(diferencas),
        "melhor_amostra": melhor_amostra,
        "melhor_amostra_fitness": melhor_amostra_fitness,
    }


def _vizinho_aleatorio(problema: Any, solucao: list[int], operador: str,
                       rng: random.Random) -> list[int]:
    """Usa rng próprio para reprodutibilidade, sem acessar o random global"""
    n = len(solucao)
    if n < 2:
        raise ValueError("É necessário ter pelo menos dois ativos")
    if operador == "swap":
        i, j = rng.sample(range(n), 2)
    elif operador == "2opt":
        i, j = sorted(rng.sample(range(n), 2))
    else:
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
    return _aplicar_movimento(solucao, operador, i, j)


def simulated_annealing(problema: Any, orcamento: int, operador: str = "swap",
                        seed: int | None = None, intervalo_historico: int = 1000,
                        T0: float | None = None, Tf: float | None = None,
                        amostras_calibracao: int = 100, blocos: int = 20) -> dict[str, Any]:
    """SA com resfriamento geométrico e orçamento exato.

    O gasto da calibração é contado. O gráfico de piores aceitações é dado
    por metricas['taxas_piores_por_bloco'], em faixas de avaliações.
    """
    _validar_operador(operador)
    if blocos < 1:
        raise ValueError("blocos deve ser >= 1")
    contador = ContadorAvaliacoes(problema, orcamento)
    rng = random.Random(seed)
    hist = HistoricoMelhor(intervalo_historico)
    calibracao = None
    if (T0 is None) != (Tf is None):
        raise ValueError("Forneça T0 e Tf juntos, ou deixe os dois automáticos")
    if T0 is None:
        # Reserva ao menos uma avaliação para inicializar a busca.
        calibracao = calibrar_temperatura(
            problema, contador, rng, operador,
            amostras=min(amostras_calibracao, max(0, orcamento - 2)),
        ) if orcamento >= 2 else {"T0": 1., "Tf": .01, "escala_delta": 1., "amostras_piores": 0}
        T0, Tf = calibracao['T0'], calibracao['Tf']
    if not math.isfinite(T0) or not math.isfinite(Tf) or T0 <= Tf or Tf <= 0:
        raise ValueError("É necessário T0 > Tf > 0 (finitos)")

    corrente = sorted(problema.solucao_aleatoria())
    rng.shuffle(corrente)
    custo = contador.avaliar(corrente)
    melhor_solucao, melhor_fitness = corrente.copy(), custo
    if calibracao is not None and "melhor_amostra_fitness" in calibracao:
        if calibracao["melhor_amostra_fitness"] < melhor_fitness:
            melhor_fitness = calibracao["melhor_amostra_fitness"]
            melhor_solucao = calibracao["melhor_amostra"].copy()
    hist.registrar(contador, melhor_fitness, forcar=True)
    inicio_busca = contador.avaliacoes
    passos_restantes = contador.disponiveis
    alpha = (Tf / T0) ** (1 / max(1, passos_restantes - 1))
    propostas_piores = [0] * blocos
    aceitas_piores = [0] * blocos
    propostas_total = [0] * blocos
    aceitas_total = [0] * blocos

    for passo in range(passos_restantes):
        T = T0 * alpha ** passo
        vizinho = _vizinho_aleatorio(problema, corrente, operador, rng)
        custo_vizinho = contador.avaliar(vizinho)
        delta = custo_vizinho - custo
        faixa = min(blocos - 1, passo * blocos // passos_restantes)
        propostas_total[faixa] += 1
        if delta > 0:
            propostas_piores[faixa] += 1
        aceita = delta <= 0 or rng.random() < math.exp(-delta / T)
        if aceita:
            corrente, custo = vizinho, custo_vizinho
            aceitas_total[faixa] += 1
            if delta > 0:
                aceitas_piores[faixa] += 1
            if custo < melhor_fitness:
                melhor_solucao, melhor_fitness = corrente.copy(), custo
        hist.registrar(contador, melhor_fitness)

    hist.registrar(contador, melhor_fitness, forcar=True)
    taxas = []
    for i in range(blocos):
        taxas.append({
            "bloco": i + 1,
            "avaliacoes_inicio": inicio_busca + (i * passos_restantes // blocos),
            "avaliacoes_fim": inicio_busca + ((i + 1) * passos_restantes // blocos),
            "piores_propostas": propostas_piores[i],
            "piores_aceitas": aceitas_piores[i],
            "taxa_aceitacao_piores": (
                aceitas_piores[i] / propostas_piores[i] if propostas_piores[i] else None
            ),
        })
    return {
        "algoritmo": "simulated_annealing", "operador": operador,
        "seed": seed, "melhor_solucao": melhor_solucao,
        "melhor_fitness": melhor_fitness,
        "avaliacoes": contador.avaliacoes, "historico": hist.dados,
        "metricas": {
            "T0": T0, "Tf": Tf, "alpha": alpha, "calibracao": calibracao,
            "taxas_piores_por_bloco": taxas,
            "piores_propostas": sum(propostas_piores),
            "piores_aceitas": sum(aceitas_piores),
            "taxa_aceitacao_piores_global": (
                sum(aceitas_piores) / sum(propostas_piores)
                if sum(propostas_piores) else None
            ),
            "taxa_aceitacao_total": sum(aceitas_total) / sum(propostas_total) if sum(propostas_total) else None,
        },
    }
