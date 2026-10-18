"""Protocolo Experimental Geral e Unificado - TP2 IA (UFMG)

Executa as 9 combinações obrigatórias:
3 Algoritmos (Greedy Descent, Simulated Annealing, Genetic Algorithm)
x 3 Instâncias (Pequena, Média, Grande)

Garante o orçamento exato e as repetições independentes da Seção 3 do enunciado:
- Pequena (n=12):  50.000 avaliações, 30 execuções independentes
- Média   (n=50): 150.000 avaliações, 30 execuções independentes
- Grande (n=150): 300.000 avaliações, 10 execuções independentes

Salva todos os resultados em arquivos CSV estruturados.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import time
from typing import Any

from genetic import genetic_algorithm
from local_search import greedy_random_restart, simulated_annealing
from problem import ProblemaManutencaoAtivos

CONFIG_INSTANCIAS = {
    "pequena": {
        "arquivo": "manutencao_pequena.txt",
        "n": 12,
        "orcamento": 50_000,
        "repeticoes": 30,
        "otimo_referencia": 1875,
    },
    "media": {
        "arquivo": "manutencao_media.txt",
        "n": 50,
        "orcamento": 150_000,
        "repeticoes": 30,
        "otimo_referencia": 16332,
    },
    "grande": {
        "arquivo": "manutencao_grande.txt",
        "n": 150,
        "orcamento": 300_000,
        "repeticoes": 10,
        "otimo_referencia": 60417,
    },
}


def executar_protocolo(
    instancias: list[str],
    algoritmos: list[str],
    repeticoes_override: int | None = None,
    orcamento_override: int | None = None,
    diretorio_saida: pathlib.Path = pathlib.Path("tp2/resultados"),
    operador_busca_local: str = "swap",
    estrategia_greedy: str = "first",
    intervalo_historico: int = 1000,
) -> dict[str, list[dict[str, Any]]]:
    """Executa a bateria de experimentos e persiste em CSVs."""
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    pasta_instancias = pathlib.Path(__file__).resolve().parents[1] / "instancias"

    resultados_execucoes: list[dict[str, Any]] = []
    convergencia_geral: list[dict[str, Any]] = []
    metricas_populacao_ag: list[dict[str, Any]] = []

    print("=" * 80)
    print("INICIANDO PROTOCOLO EXPERIMENTAL - TP2 FUNDAMENTOS DE IA")
    print(f"Instâncias: {instancias} | Algoritmos: {algoritmos}")
    print(f"Diretório de saída: {diretorio_saida.resolve()}")
    print("=" * 80)

    for nome_instancia in instancias:
        cfg = CONFIG_INSTANCIAS[nome_instancia]
        orcamento = orcamento_override if orcamento_override is not None else cfg["orcamento"]
        n_repeticoes = repeticoes_override if repeticoes_override is not None else cfg["repeticoes"]

        print(f"\n>>> Carregando Instância: {nome_instancia.upper()} ({cfg['arquivo']})")
        print(f"    Orçamento: {orcamento} avaliações | Execuções: {n_repeticoes}")
        problema = ProblemaManutencaoAtivos(cfg["arquivo"], pasta=str(pasta_instancias))

        for algo in algoritmos:
            print(f"\n  --- Algoritmo: {algo.upper()} ---")
            custos_algo = []

            for rep in range(n_repeticoes):
                seed = 1000 + rep
                t_inicio = time.perf_counter()

                if algo == "greedy":
                    res = greedy_random_restart(
                        problema,
                        orcamento=orcamento,
                        operador=operador_busca_local,
                        estrategia=estrategia_greedy,
                        seed=seed,
                        intervalo_historico=intervalo_historico,
                    )
                elif algo == "sa":
                    res = simulated_annealing(
                        problema,
                        orcamento=orcamento,
                        operador=operador_busca_local,
                        seed=seed,
                        intervalo_historico=intervalo_historico,
                    )
                elif algo == "ag":
                    res = genetic_algorithm(
                        problema,
                        orcamento=orcamento,
                        tamanho_populacao=100,
                        taxa_cruzamento=0.9,
                        taxa_mutacao=0.1,
                        metodo_selecao="torneio",
                        tamanho_torneio=3,
                        elitismo=1,
                        seed=seed,
                        intervalo_historico=intervalo_historico,
                    )
                else:
                    raise ValueError(f"Algoritmo desconhecido: {algo}")

                duracao_seg = time.perf_counter() - t_inicio
                custos_algo.append(res["melhor_fitness"])

                # Registro da execução individual
                linha = {
                    "instancia": nome_instancia,
                    "n_ativos": cfg["n"],
                    "algoritmo": algo,
                    "seed": seed,
                    "orcamento": orcamento,
                    "avaliacoes_utilizadas": res["avaliacoes"],
                    "melhor_fitness": res["melhor_fitness"],
                    "tempo_segundos": round(duracao_seg, 4),
                    "otimo_referencia": cfg["otimo_referencia"],
                    "gap_relativo": (
                        round((res["melhor_fitness"] - cfg["otimo_referencia"]) / cfg["otimo_referencia"], 4)
                        if cfg["otimo_referencia"] > 0
                        else None
                    ),
                    "melhor_solucao": " ".join(map(str, res["melhor_solucao"])),
                }

                # Metricas específicas do algoritmo
                metricas = res.get("metricas", {})
                if algo == "greedy":
                    linha["greedy_passos"] = metricas.get("passos_melhoria")
                    linha["greedy_restarts"] = metricas.get("random_restarts")
                    linha["greedy_otimos_locais"] = metricas.get("otimos_locais_distintos")
                elif algo == "sa":
                    linha["sa_T0"] = metricas.get("T0")
                    linha["sa_Tf"] = metricas.get("Tf")
                    linha["sa_alpha"] = metricas.get("alpha")
                    linha["sa_taxa_aceitacao_piores"] = metricas.get("taxa_aceitacao_piores_global")
                elif algo == "ag":
                    linha["ag_geracoes"] = metricas.get("geracoes_completadas")
                    linha["ag_diversidade_final"] = metricas.get("diversidade_final")
                    linha["ag_media_fitness_final"] = metricas.get("media_fitness_final")

                resultados_execucoes.append(linha)

                # Histórico de convergência do melhor valor de h
                for aval, fit in res["historico"]:
                    convergencia_geral.append({
                        "instancia": nome_instancia,
                        "algoritmo": algo,
                        "seed": seed,
                        "avaliacoes": aval,
                        "melhor_fitness": fit,
                    })

                # Curva populacional detalhada do AG (média da população vs melhor)
                if algo == "ag" and "curva_populacao" in metricas:
                    for p_dado in metricas["curva_populacao"]:
                        metricas_populacao_ag.append({
                            "instancia": nome_instancia,
                            "seed": seed,
                            "geracao": p_dado["geracao"],
                            "avaliacoes": p_dado["avaliacoes"],
                            "melhor_fitness": p_dado["melhor_fitness"],
                            "media_fitness_populacao": p_dado["media_fitness_populacao"],
                            "cromossomos_distintos": p_dado["cromossomos_distintos"],
                            "taxa_diversidade": p_dado["taxa_diversidade"],
                        })

                print(
                    f"    [{algo.upper():6}] Rep {rep+1:2d}/{n_repeticoes} | Seed {seed} | "
                    f"Melhor h: {res['melhor_fitness']:.1f} | Evals: {res['avaliacoes']} | "
                    f"Tempo: {duracao_seg:.2f}s",
                    flush=True,
                )

            print(
                f"  >> Resumo {nome_instancia} {algo.upper()}: "
                f"Média = {statistics.mean(custos_algo):.2f} ± {statistics.stdev(custos_algo) if len(custos_algo) > 1 else 0:.2f} | "
                f"Mediana = {statistics.median(custos_algo):.2f} | Min = {min(custos_algo):.2f}"
            )

    # PERSISTÊNCIA EM ARQUIVOS CSV
    _salvar_csv(diretorio_saida / "resultados_gerais.csv", resultados_execucoes)
    _salvar_csv(diretorio_saida / "convergencia_geral.csv", convergencia_geral)
    if metricas_populacao_ag:
        _salvar_csv(diretorio_saida / "curvas_ag_populacao.csv", metricas_populacao_ag)

    print("\n" + "=" * 80)
    print("TODOS OS EXPERIMENTOS CONCLUÍDOS COM SUCESSO!")
    print(f"Arquivos gerados em: {diretorio_saida.resolve()}")
    print("=" * 80)

    return {
        "resultados": resultados_execucoes,
        "convergencia": convergencia_geral,
        "populacao_ag": metricas_populacao_ag,
    }


def executar_experimento_adicional_crossover(
    instancia: str = "media",
    repeticoes: int = 15,
    orcamento: int = 100_000,
    diretorio_saida: pathlib.Path = pathlib.Path("tp2/resultados"),
) -> list[dict[str, Any]]:
    """Experimento Adicional Obrigatório do AG (Seção 5 / Seção 6):
    
    Testa a hipótese sobre o papel da recombinação (crossover OX):
    'A recombinação consegue combinar bons trechos de dois pais em um filho melhor?'
    Compara o AG padrão (Pc = 0.9) vs AG puramente mutacional sem cruzamento (Pc = 0.0).
    """
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    pasta_instancias = pathlib.Path(__file__).resolve().parents[1] / "instancias"
    cfg = CONFIG_INSTANCIAS[instancia]
    problema = ProblemaManutencaoAtivos(cfg["arquivo"], pasta=str(pasta_instancias))

    print("\n" + "=" * 80)
    print("EXECUTANDO EXPERIMENTO ADICIONAL DO AG: PAPEL DO CROSSOVER OX")
    print(f"Instância: {instancia} | Orçamento: {orcamento} | Repetições: {repeticoes}")
    print("=" * 80)

    linhas_comparacao = []
    for modo, pc in [("AG_com_OX_Pc0.9", 0.9), ("AG_sem_OX_Pc0.0_apenas_mutacao", 0.0)]:
        print(f"\n>> Testando configuração: {modo}")
        custos = []
        for rep in range(repeticoes):
            seed = 5000 + rep
            t0 = time.perf_counter()
            res = genetic_algorithm(
                problema,
                orcamento=orcamento,
                tamanho_populacao=100,
                taxa_cruzamento=pc,
                taxa_mutacao=0.15,
                seed=seed,
            )
            dt = time.perf_counter() - t0
            custos.append(res["melhor_fitness"])
            linhas_comparacao.append({
                "configuracao": modo,
                "taxa_cruzamento_pc": pc,
                "seed": seed,
                "orcamento": orcamento,
                "melhor_fitness": res["melhor_fitness"],
                "tempo_segundos": round(dt, 4),
                "diversidade_final": res["metricas"]["diversidade_final"],
                "media_fitness_final": res["metricas"]["media_fitness_final"],
            })
            print(f"   Rep {rep+1:2d}/{repeticoes} | Seed {seed} | h = {res['melhor_fitness']:.1f} | Tempo: {dt:.2f}s")
        dp = statistics.stdev(custos) if len(custos) > 1 else 0.0
        print(f"   Média {modo}: {statistics.mean(custos):.2f} ± {dp:.2f} | Mediana: {statistics.median(custos):.2f}")

    _salvar_csv(diretorio_saida / "adicional_ag_crossover.csv", linhas_comparacao)
    print(f"\nResultado do experimento adicional salvo em: {diretorio_saida / 'adicional_ag_crossover.csv'}")
    return linhas_comparacao


def _salvar_csv(caminho: pathlib.Path, dados: list[dict[str, Any]]):
    if not dados:
        return
    # Coleta a união ordenada de todas as colunas presentes em qualquer registro
    colunas: list[str] = []
    vistas: set[str] = set()
    for item in dados:
        for chave in item.keys():
            if chave not in vistas:
                vistas.add(chave)
                colunas.append(chave)

    with caminho.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=colunas, delimiter=",", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dados)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execução do Protocolo Experimental Unificado (TP2)")
    parser.add_argument("--instancia", choices=("pequena", "media", "grande", "todas"), default="pequena")
    parser.add_argument("--algoritmo", choices=("greedy", "sa", "ag", "todos"), default="todos")
    parser.add_argument("--orcamento", type=int, default=None, help="Sobrescreve o orçamento padrão")
    parser.add_argument("--repeticoes", type=int, default=1, help="Número de execuções independentes")
    parser.add_argument("--protocolo", action="store_true", help="Executa com repetições e orçamentos do PDF (30/30/10)")
    parser.add_argument("--experimento-crossover", action="store_true", help="Executa o experimento adicional de validação do OX")
    parser.add_argument("--saida", type=pathlib.Path, default=pathlib.Path("tp2/resultados"))
    args = parser.parse_args()

    instancias_alvo = list(CONFIG_INSTANCIAS.keys()) if args.instancia == "todas" else [args.instancia]
    algos_alvo = ["greedy", "sa", "ag"] if args.algoritmo == "todos" else [args.algoritmo]

    if args.experimento_crossover:
        executar_experimento_adicional_crossover(
            instancia=instancias_alvo[0],
            repeticoes=args.repeticoes if not args.protocolo else 15,
            orcamento=args.orcamento or 100_000,
            diretorio_saida=args.saida,
        )
    else:
        executar_protocolo(
            instancias=instancias_alvo,
            algoritmos=algos_alvo,
            repeticoes_override=None if args.protocolo else args.repeticoes,
            orcamento_override=None if args.protocolo else args.orcamento,
            diretorio_saida=args.saida,
        )
