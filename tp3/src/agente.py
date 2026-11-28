from collections.abc import Callable

import numpy as np

N_ESTADOS = 162
N_ACOES = 7

NOMES_ACOES: tuple[str, ...] = (
    "norte",
    "sul",
    "leste",
    "oeste",
    "aleatoria",
    "parado",
    "coletar",
)

NOMES_CELULAS: tuple[str, ...] = ("vazio", "lixo", "parede")


def codificar_estado(c_atual: int, c_n: int, c_s: int, c_l: int, c_o: int) -> int:
    """Codifica a quíntupla perceptual (c_atual, c_N, c_S, c_L, c_O) no índice 0..161."""
    if c_atual not in (0, 1):
        raise ValueError(f"c_atual inválido: {c_atual}. Esperado 0 (vazio) ou 1 (lixo).")
    for nome, valor in (("c_N", c_n), ("c_S", c_s), ("c_L", c_l), ("c_O", c_o)):
        if valor not in (0, 1, 2):
            raise ValueError(
                f"{nome} inválido: {valor}. Esperado 0 (vazio), 1 (lixo) ou 2 (parede)."
            )
    return c_atual * 81 + c_n * 27 + c_s * 9 + c_l * 3 + c_o


def decodificar_estado(estado_idx: int) -> tuple[int, int, int, int, int]:
    """Converte um índice de estado (0..161) na quíntupla (c_atual, c_N, c_S, c_L, c_O)."""
    if not (0 <= estado_idx < N_ESTADOS):
        raise ValueError(f"Índice de estado fora do intervalo [0, 161]: {estado_idx}")

    c_atual, resto = divmod(estado_idx, 81)
    c_n, resto = divmod(resto, 27)
    c_s, resto = divmod(resto, 9)
    c_l, c_o = divmod(resto, 3)
    return c_atual, c_n, c_s, c_l, c_o


def formatar_estado(estado_idx: int) -> str:
    """Retorna representação legível da quíntupla perceptual para análise da política."""
    quintupla = decodificar_estado(estado_idx)
    nomes = ", ".join(NOMES_CELULAS[c] for c in quintupla)
    return f"({nomes})"


def criar_agenda_epsilon(
    tipo: str,
    epsilon_0: float = 1.0,
    epsilon_min: float = 0.01,
    decaimento: float = 0.99,
    epsilon_const: float = 0.1,
) -> Callable[[int], float]:
    """Cria uma função f(k) -> epsilon_k para o episódio k (1-indexado)."""
    if tipo == "constante":
        return lambda _k: float(epsilon_const)
    if tipo == "exponencial":
        return lambda k: float(max(epsilon_min, epsilon_0 * (decaimento**k)))
    if tipo == "harmonica":
        return lambda k: float(max(epsilon_min, 1.0 / max(1, k)))
    raise ValueError(
        f"Agenda desconhecida: '{tipo}'. Use 'constante', 'exponencial' ou 'harmonica'."
    )


class AgenteQLearning:
    """Agente tabular Q-Learning com política epsilon-greedy e desempate aleatório."""

    def __init__(
        self,
        n_estados: int = N_ESTADOS,
        n_acoes: int = N_ACOES,
        alpha: float = 0.1,
        gamma: float = 0.9,
        q0: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.n_estados = n_estados
        self.n_acoes = n_acoes
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q0 = float(q0)

        self.q_table = np.full((n_estados, n_acoes), self.q0, dtype=np.float64)
        self.visitas = np.zeros((n_estados, n_acoes), dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def _argmax_aleatorio(self, valores: np.ndarray) -> int:
        """Retorna um índice de valor máximo, sorteando uniformemente em caso de empate."""
        melhores = np.flatnonzero(valores == np.max(valores))
        return int(self.rng.choice(melhores))

    def selecionar_acao(self, estado: int, epsilon: float) -> int:
        """Seleciona uma ação segundo a política epsilon-greedy (Eq. 4)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_acoes))
        return self._argmax_aleatorio(self.q_table[estado])

    def atualizar(
        self,
        estado: int,
        acao: int,
        recompensa: float,
        proximo_estado: int,
    ) -> float:
        """Atualiza Q(s, a) pela regra TD do Q-Learning (Eq. 2) e registra a visita."""
        melhor_futuro = float(np.max(self.q_table[proximo_estado]))
        alvo = recompensa + self.gamma * melhor_futuro
        erro_td = alvo - float(self.q_table[estado, acao])

        self.q_table[estado, acao] += self.alpha * erro_td
        self.visitas[estado, acao] += 1
        return erro_td

    def fracao_visitada(self) -> float:
        """Retorna a fração de pares (s, a) da Q-table visitados ao menos uma vez."""
        return float(np.count_nonzero(self.visitas) / self.q_table.size)

    def obter_politica(self) -> np.ndarray:
        """Extrai a política gulosa atual pi(s) = argmax_a Q(s, a) para todos os estados."""
        return np.array(
            [self._argmax_aleatorio(self.q_table[s]) for s in range(self.n_estados)],
            dtype=np.int64,
        )
