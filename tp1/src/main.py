from buscas import busca_bfs, busca_dfs, busca_custo_minimo, busca_a_estrela
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

    # imprime os detalhes do trajeto e o passo a passo de execucao
    print("Detalhes do trajeto:")
    for i, no in enumerate(caminho):
        if i == 0:
            print(f"Estado Inicial: Todos na margem de origem {no.estado}")
        else:
            print(f"Passo {i}: {no.acao}")
            print(
                f"  └> Como ficou a ponte: {no.estado} | Relógio marcando: {no.custo_acumulado} min"
            )


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
    
    print("\n" + "-" * 50 + "\n")

    # Roda custo mínimo
    print("Executando Busca de Custo Mínimo (Custo Uniforme / UCS)")
    resultado_ucs = busca_custo_minimo(estado_inicial)
    imprimir_caminho(resultado_ucs, "UCS")

    print("\n" + "-" * 50 + "\n")

    # Roda A*
    print("Executando Busca A*")
    resultado_a_estrela = busca_a_estrela(estado_inicial)
    imprimir_caminho(resultado_a_estrela, "A*")