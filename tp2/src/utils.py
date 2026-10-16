import math
import os


def carregar_instancia(nome_arquivo, pasta="instancias"):
    """Lê o arquivo de instância dentro de uma pasta específica e retorna os dados."""
    caminho_completo = os.path.join(pasta, nome_arquivo)

    if not os.path.exists(caminho_completo):
        raise FileNotFoundError(f"Arquivo não encontrado em: {caminho_completo}")

    with open(caminho_completo, "r", encoding="utf-8") as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    n = int(linhas[0])
    base_x, base_y = map(float, linhas[1].split())

    dict_ativos = {}
    for i in range(2, n + 2):
        partes = linhas[i].split()
        ativo_id = i - 1
        dict_ativos[ativo_id] = {
            "id": ativo_id,
            "x": float(partes[0]),
            "y": float(partes[1]),
            "p": float(partes[2]),  # Duração do serviço
            "d": float(partes[3]),  # Prazo regulatório (deadline)
            "w": float(partes[4]),  # Peso de criticidade
        }

    return n, (base_x, base_y), dict_ativos


def calcular_tempo_deslocamento(p1, p2):
    """Calcula o tempo de deslocamento arredondado entre dois pontos"""
    distancia = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return round(distancia)


def avaliar_fitness(permutacao, base, dict_ativos):
    """Calcula o atraso ponderado total h(pi) para uma dada rota de forma otimizada"""
    c_atual = 0.0
    ponto_atual = base
    custo_total = 0.0

    for ativo_id in permutacao:
        ativo = dict_ativos[ativo_id]
        ponto_destino = (ativo["x"], ativo["y"])

        t_ij = calcular_tempo_deslocamento(ponto_atual, ponto_destino)
        c_atual = c_atual + t_ij + ativo["p"]

        atraso = max(0.0, c_atual - ativo["d"])
        custo_total += ativo["w"] * atraso

        ponto_atual = ponto_destino

    return custo_total


# --- Bloco de Teste e Validação (T1) ---
if __name__ == "__main__":
    try:
        n, base, dict_ativos = carregar_instancia(
            "manutencao_pequena.txt", pasta="tp2/instancias"
        )

        permutacao_identidade = list(dict_ativos.keys())
        h_valor = avaliar_fitness(permutacao_identidade, base, dict_ativos)

        print(f"Número de ativos (n): {n}")
        print(f"Fitness para a permutação identidade h(pi) = {h_valor}")

        esperado = 4366
        if h_valor == esperado:
            print(
                "Sucesso! O valor bateu com o esperado (4366). A função de fitness está validada!"
            )
        else:
            print(f"Atenção: Esperado {esperado}, mas obteve {h_valor}.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro ao executar o teste: {e}")
