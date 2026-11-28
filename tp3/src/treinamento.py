from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from src.agente import (
    NOMES_ACOES,
    AgenteQLearning,
    criar_agenda_epsilon,
    decodificar_estado,
    formatar_estado,
)
from src.ambiente import Ambiente


def executar_episodio(
    env: Ambiente,
    agente: AgenteQLearning,
    epsilon: float,
    horizonte: int = 1000,
) -> dict[str, float | int]:
    """Executa um episódio completo de horizonte fixo H e atualiza a Q-table."""
    estado = env.reset()
    recompensa_total = 0.0
    lixos_coletados = 0
    coletas_invalidas = 0
    movs_invalidos = 0

    for _ in range(horizonte):
        acao = agente.selecionar_acao(estado, epsilon)
        prox_estado, recompensa, info = env.step(acao)
        agente.atualizar(estado, acao, recompensa, prox_estado)

        recompensa_total += recompensa
        if info["coletou"]:
            lixos_coletados += 1
        if info["coleta_invalida"]:
            coletas_invalidas += 1
        if info["mov_invalido"]:
            movs_invalidos += 1

        estado = prox_estado

    return {
        "recompensa_total": recompensa_total,
        "lixos_coletados": lixos_coletados,
        "coletas_invalidas": coletas_invalidas,
        "movs_invalidos": movs_invalidos,
        "epsilon": float(epsilon),
    }


def treinar_marvin(
    env: Ambiente,
    agente: AgenteQLearning,
    agenda_epsilon: Callable[[int], float],
    n_episodios: int = 500,
    horizonte: int = 1000,
) -> dict[str, Any]:
    """Treina Marvin por K episódios e retorna o histórico de métricas por episódio."""
    recompensas: list[float] = []
    lixos_coletados: list[int] = []
    coletas_invalidas: list[int] = []
    movs_invalidos: list[int] = []
    epsilons: list[float] = []

    for k in range(1, n_episodios + 1):
        eps_k = agenda_epsilon(k)
        res = executar_episodio(env, agente, epsilon=eps_k, horizonte=horizonte)

        recompensas.append(float(res["recompensa_total"]))
        lixos_coletados.append(int(res["lixos_coletados"]))
        coletas_invalidas.append(int(res["coletas_invalidas"]))
        movs_invalidos.append(int(res["movs_invalidos"]))
        epsilons.append(float(res["epsilon"]))

    return {
        "recompensas": np.array(recompensas, dtype=np.float64),
        "lixos_coletados": np.array(lixos_coletados, dtype=np.int64),
        "coletas_invalidas": np.array(coletas_invalidas, dtype=np.int64),
        "movs_invalidos": np.array(movs_invalidos, dtype=np.int64),
        "epsilons": np.array(epsilons, dtype=np.float64),
        "fracao_visitada": agente.fracao_visitada(),
    }


def media_movel(valores: Sequence[float] | np.ndarray, janela: int = 20) -> np.ndarray:
    """Calcula a média móvel com janela variável no início para manter o mesmo tamanho."""
    arr = np.asarray(valores, dtype=np.float64)
    if janela < 1:
        raise ValueError("A janela da média móvel deve ser >= 1.")
    if arr.size == 0:
        return arr.copy()

    resultado = np.empty_like(arr)
    soma_acum = np.cumsum(arr)
    for i in range(arr.size):
        inicio = i - janela
        soma = soma_acum[i] - (soma_acum[inicio] if inicio >= 0 else 0.0)
        qtd = min(i + 1, janela)
        resultado[i] = soma / qtd
    return resultado


def executar_varredura_epsilon(
    epsilons: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.3, 1.0),
    alpha: float = 0.1,
    gamma: float = 0.9,
    n_episodios: int = 500,
    horizonte: int = 1000,
    seed_env: int | None = 42,
    seed_agente: int | None = 42,
) -> list[dict[str, Any]]:
    """Executa o experimento da Tarefa 4(a) mantendo alpha, gamma, K e H fixos."""
    resultados: list[dict[str, Any]] = []
    for eps in epsilons:
        env = Ambiente(seed=seed_env)
        agente = AgenteQLearning(alpha=alpha, gamma=gamma, seed=seed_agente)
        agenda = criar_agenda_epsilon("constante", epsilon_const=eps)
        hist = treinar_marvin(
            env,
            agente,
            agenda,
            n_episodios=n_episodios,
            horizonte=horizonte,
        )
        recomps = hist["recompensas"]
        resultados.append(
            {
                "epsilon": float(eps),
                "recompensa_media_ultimos_50": float(np.mean(recomps[-50:])),
                "melhor_episodio": float(np.max(recomps)),
                "pct_qtable_visitada": float(hist["fracao_visitada"] * 100.0),
                "historico": hist,
                "agente": agente,
            }
        )
    return resultados


def comparar_agendas_epsilon(
    alpha: float = 0.1,
    gamma: float = 0.9,
    n_episodios: int = 500,
    horizonte: int = 1000,
    epsilon_0: float = 1.0,
    epsilon_min: float = 0.01,
    decaimento: float = 0.99,
    epsilon_const: float = 0.1,
    seed_env: int | None = 42,
    seed_agente: int | None = 42,
) -> dict[str, dict[str, Any]]:
    """Executa o experimento da Tarefa 4(b) comparando constante, exponencial e harmônica."""
    agendas = {
        "constante": criar_agenda_epsilon("constante", epsilon_const=epsilon_const),
        "exponencial": criar_agenda_epsilon(
            "exponencial",
            epsilon_0=epsilon_0,
            epsilon_min=epsilon_min,
            decaimento=decaimento,
        ),
        "harmonica": criar_agenda_epsilon(
            "harmonica",
            epsilon_min=epsilon_min,
        ),
    }

    comparacao: dict[str, dict[str, Any]] = {}
    for nome, agenda in agendas.items():
        env = Ambiente(seed=seed_env)
        agente = AgenteQLearning(alpha=alpha, gamma=gamma, seed=seed_agente)
        hist = treinar_marvin(
            env,
            agente,
            agenda,
            n_episodios=n_episodios,
            horizonte=horizonte,
        )
        comparacao[nome] = {"historico": hist, "agente": agente}
    return comparacao


def extrair_tabela_politica(agente: AgenteQLearning) -> list[dict[str, Any]]:
    """Extrai o mapeamento estado -> ação para análise da política aprendida (Tarefa 5)."""
    politica = agente.obter_politica()
    tabela: list[dict[str, Any]] = []
    for estado_idx in range(agente.n_estados):
        acao_idx = int(politica[estado_idx])
        c_atual, c_n, c_s, c_l, c_o = decodificar_estado(estado_idx)
        tabela.append(
            {
                "estado_idx": estado_idx,
                "quintupla": (c_atual, c_n, c_s, c_l, c_o),
                "estado_str": formatar_estado(estado_idx),
                "acao_idx": acao_idx,
                "acao_nome": NOMES_ACOES[acao_idx],
                "q_max": float(agente.q_table[estado_idx, acao_idx]),
                "visitas_estado": int(np.sum(agente.visitas[estado_idx])),
            }
        )
    return tabela
