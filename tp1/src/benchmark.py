from collections.abc import Callable
from statistics import mean, stdev
from time import perf_counter_ns

from buscas import busca_a_estrela, busca_bfs, busca_custo_minimo, busca_dfs
from model import Estado, ResultadoBusca

ESTADO_INICIAL: Estado = (False, False, False, False, False)
REPETICOES = 10000 # mesmo com 90000 repetições, o tempo varia bem pouco entre as execuções.
AQUECIMENTO = 500 # só pra não considerar o tempo das  primeiras execuções que são mais lentas por causa de cache, carregamento do código e etc, 
# testado com 500, 0, 1000 e 5000 de aquecimento, mas pouca variação pra esse problema que é muito simples

AlgoritmoBusca = Callable[[Estado], ResultadoBusca]

ALGORITMOS: tuple[tuple[str, AlgoritmoBusca], ...] = (
    ("BFS", busca_bfs),
    ("DFS", busca_dfs),
    ("UCS", busca_custo_minimo),
    ("A*", busca_a_estrela),
)


def executar_benchmark(nome: str, algoritmo: AlgoritmoBusca) -> dict[str, float | str]:
    """
    Executa uma busca repetidamente e resume as métricas da Tarefa4
    Benchmark pro algoritmo de busca fornecido.
    """
    for _ in range(AQUECIMENTO):
        algoritmo(ESTADO_INICIAL)

    tempos_ms: list[float] = []
    custos: list[int] = []
    expansoes: list[int] = []

    for _ in range(REPETICOES):
        inicio = perf_counter_ns()
        resultado = algoritmo(ESTADO_INICIAL)
        fim = perf_counter_ns()

        if resultado.no_final is None:
            motivo = (
                "limite de expansões atingido"
                if resultado.limite_atingido
                else "fronteira esgotada"
            )
            raise RuntimeError(f"{nome} não encontrou solução: {motivo}.")

        tempos_ms.append((fim - inicio) / 1_000_000)
        custos.append(resultado.no_final.custo_acumulado)
        expansoes.append(resultado.nos_expandidos)

    return {
        "nome": nome,
        "custo_medio": mean(custos),
        "expansoes_medias": mean(expansoes),
        "tempo_medio_ms": mean(tempos_ms),
        "desvio_tempo_ms": stdev(tempos_ms),
    }


def main() -> None:
    resultados = [executar_benchmark(nome, algoritmo) for nome, algoritmo in ALGORITMOS]

    print(f"{REPETICOES} execuções pra cada algoritmo, com {AQUECIMENTO} aquecimentos")
    print(
        "| Método | Custo médio (min) | Nós expandidos médios | "
        "Tempo médio (ms) | Desvio padrão (ms) |"
    )
    print("|---|---:|---:|---:|---:|")

    for resultado in resultados:
        print(
            f"| {resultado['nome']} "
            f"| {resultado['custo_medio']:.2f} "
            f"| {resultado['expansoes_medias']:.2f} "
            f"| {resultado['tempo_medio_ms']:.6f} "
            f"| {resultado['desvio_tempo_ms']:.6f} |"
        )


if __name__ == "__main__":
    main()
