from buscas import busca_bfs, busca_dfs
from model import Estado, No


def imprimir_caminho(no_final: No, nome_busca: str):
    """Refaz o caminho do nó final até a raiz e imprime no terminal"""
    if not no_final:
        print("\n[[nome_busca]] Nenhuma solução encontrada.")
        return

    # refaz o caminho de trás pra frente usando o ponteiro 'pai'
    caminho = []
    no_atual = no_final
    while no_atual is not None:
        caminho.append(no_atual)
        no_atual = no_atual.pai

    # inverte a lista para ficar da origem até o destino
    caminho.reverse()

    print(f"\n--- Solução usando {nome_busca} ---")
    print(f"Passos totais: {len(caminho) - 1}")
    print(f"Tempo total gasto: {no_final.custo_acumulado} minutos\n")


if __name__ == "__main__":
    estado_inicial: Estado = (False, False, False, False, False)

    # roda o bfs
    print("Executando Busca em Largura (BFS)")
    resultado_bfs = busca_bfs(estado_inicial)
    imprimir_caminho(resultado_bfs, "BFS")

    print("\n" + "-" * 50 + "\n")

    # Roda o dfs
    print("Executando Busca em Profundidade (DFS)...")
    resultado_dfs = busca_dfs(estado_inicial)
    imprimir_caminho(resultado_dfs, "DFS")
