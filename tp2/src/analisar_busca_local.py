"""Analisa os CSVs dos experimentos de Busca Local (T2 e SA)
Comando de execução:
 python tp2/src/analisar_busca_local.py --entrada tp2/resultados_busca_local/t2_pequena --saida tp2/analises_busca_local/pequena
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
from collections import defaultdict


def ler_csv(caminho: pathlib.Path) -> list[dict[str, str]]:
    if not caminho.is_file():
        return []
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def salvar_csv(caminho: pathlib.Path, linhas: list[dict], colunas: list[str]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas)


def numero(linha: dict, chave: str) -> float | None:
    texto = linha.get(chave)
    if texto is None or str(texto).strip() == "":
        return None
    try:
        valor = float(texto)
        return valor if math.isfinite(valor) else None
    except (TypeError, ValueError):
        return None


def analisar_t2(registros: list[dict], saida: pathlib.Path) -> list[dict]:
    grupos = defaultdict(list)
    for linha in registros:
        if linha.get("algoritmo") == "greedy":
            grupos[(linha.get("instancia"), linha.get("operador"), linha.get("estrategia"))].append(linha)

    tabela = []
    for (inst, op, estrategia), linhas in sorted(grupos.items()):
        valores = [v for item in linhas if (v := numero(item, "melhor_fitness")) is not None]
        tempos = [v for item in linhas if (v := numero(item, "segundos")) is not None]
        passos = [v for item in linhas if (v := numero(item, "passos_melhoria")) is not None]
        reinicios = [v for item in linhas if (v := numero(item, "random_restarts")) is not None]
        otimos = [v for item in linhas if (v := numero(item, "otimos_locais_distintos")) is not None]
        avaliacoes = [v for item in linhas if (v := numero(item, "avaliacoes")) is not None]
        if not valores:
            continue
        tabela.append({
            "instancia": inst, "operador": op, "estrategia": estrategia,
            "repeticoes": len(valores),
            "fitness_medio": round(statistics.mean(valores), 3),
            "fitness_desvio_padrao": round(statistics.stdev(valores), 3) if len(valores)>1 else "",
            "fitness_mediana": round(statistics.median(valores), 3),
            "fitness_minimo": min(valores),
            "tempo_medio_s": round(statistics.mean(tempos), 4) if tempos else "",
            "passos_melhoria_medio": round(statistics.mean(passos), 2) if passos else "",
            "reinicios_medio": round(statistics.mean(reinicios), 2) if reinicios else "",
            "otimos_locais_distintos_medio": round(statistics.mean(otimos), 2) if otimos else "",
            "avaliacoes_min": min(avaliacoes) if avaliacoes else "",
            "avaliacoes_max": max(avaliacoes) if avaliacoes else "",
        })
    colunas = ["instancia", "operador", "estrategia", "repeticoes", "fitness_medio",
               "fitness_desvio_padrao", "fitness_mediana", "fitness_minimo", "tempo_medio_s",
               "passos_melhoria_medio", "reinicios_medio", "otimos_locais_distintos_medio",
               "avaliacoes_min", "avaliacoes_max"]
    salvar_csv(saida / "resumo_t2.csv", tabela, colunas)
    return tabela


def analisar_sa(registros: list[dict], taxas: list[dict], saida: pathlib.Path) -> tuple[list[dict], list[dict]]:
    # Uma linha por instância/seed, com parâmetros usados na execução.
    parametros = []
    for linha in registros:
        if linha.get("algoritmo") == "sa":
            parametros.append({k: linha.get(k, "") for k in
                               ["instancia", "seed", "operador", "melhor_fitness", "avaliacoes",
                                "T0", "Tf", "alpha", "escala_delta", "amostras_piores",
                                "taxa_aceitacao_piores_global", "piores_propostas", "piores_aceitas"]})
    cols_param = ["instancia", "seed", "operador", "melhor_fitness", "avaliacoes", "T0", "Tf",
                  "alpha", "escala_delta", "amostras_piores", "taxa_aceitacao_piores_global",
                  "piores_propostas", "piores_aceitas"]
    salvar_csv(saida / "parametros_sa.csv", parametros, cols_param)

    # Agrupa propostas/aceitações por bloco e instância. Taxa agrupada ponderada
    # pelo número de propostas piores (não média simples de percentuais).
    grupos = defaultdict(lambda: [0, 0, []])
    for linha in taxas:
        if linha.get("algoritmo") != "sa":
            continue
        key = (linha.get("instancia"), int(linha["bloco"]))
        prop = numero(linha, "piores_propostas")
        ace = numero(linha, "piores_aceitas")
        if prop is not None and ace is not None:
            grupos[key][0] += int(prop)
            grupos[key][1] += int(ace)
        x = numero(linha, "avaliacoes_fim")
        if x is not None:
            grupos[key][2].append(x)
    resumo = []
    for (inst, bloco), (propostas, aceitas, finais) in sorted(grupos.items()):
        resumo.append({"instancia": inst, "bloco": bloco,
                       "avaliacoes_fim_mediana": statistics.median(finais) if finais else "",
                       "piores_propostas": propostas, "piores_aceitas": aceitas,
                       "taxa_aceitacao_piores": (aceitas/propostas if propostas else "")})
    cols_resumo = ["instancia", "bloco", "avaliacoes_fim_mediana", "piores_propostas",
                   "piores_aceitas", "taxa_aceitacao_piores"]
    salvar_csv(saida / "resumo_aceitacao_sa.csv", resumo, cols_resumo)
    return parametros, resumo


def gerar_graficos(registros: list[dict], resumo_sa: list[dict], saida: pathlib.Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib não instalado: CSVs de resumo gerados, sem PNGs.")
        return False
    instancias = sorted(set(r["instancia"] for r in registros if r.get("instancia")))
    for inst in instancias:
        # Boxplot das seis configurações do greedy
        base = [r for r in registros if r.get("instancia") == inst and r.get("algoritmo") == "greedy"]
        chaves = [(op, est) for op in ("swap", "insertion", "2opt") for est in ("best", "first")]
        conjuntos = [[v for r in base if (r.get("operador"), r.get("estrategia")) == chave
                      if (v := numero(r, "melhor_fitness")) is not None] for chave in chaves]
        ativos = [(f"{op}\n{est}", vals) for (op, est), vals in zip(chaves, conjuntos) if vals]
        if ativos:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.boxplot([v for _, v in ativos], tick_labels=[n for n, _ in ativos])
            ax.set_title(f"T2 — Vizinhanças do Greedy | {inst}")
            ax.set_ylabel("Melhor fitness h (menor é melhor)")
            ax.grid(axis="y", alpha=.3)
            fig.tight_layout()
            fig.savefig(saida / f"boxplot_t2_{inst}.png", dpi=180)
            plt.close(fig)
        sa = [r for r in resumo_sa if r["instancia"] == inst and r["taxa_aceitacao_piores"] != ""]
        if sa:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot([float(r["avaliacoes_fim_mediana"]) for r in sa],
                    [100 * float(r["taxa_aceitacao_piores"]) for r in sa], marker="o")
            ax.set_title(f"SA — Aceitação de movimentos piores | {inst}")
            ax.set_xlabel("Avaliações de h")
            ax.set_ylabel("Taxa de aceitação de pioras (%)")
            ax.set_ylim(0, 100)
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(saida / f"aceitacao_sa_{inst}.png", dpi=180)
            plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Analisa experimentos T2 e SA já salvos em CSV")
    p.add_argument("--entrada", type=pathlib.Path, required=True, help="Pasta contendo os quatro CSVs gerados")
    p.add_argument("--saida", type=pathlib.Path, required=True, help="Pasta para resumos e gráficos")
    args = p.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    registros = ler_csv(args.entrada / "resultados_busca_local.csv")
    if not registros:
        p.error(f"Não encontrei resultados em {args.entrada / 'resultados_busca_local.csv'}")
    taxas = ler_csv(args.entrada / "aceitacao_sa.csv")
    tabela = analisar_t2(registros, args.saida)
    parametros, resumo_sa = analisar_sa(registros, taxas, args.saida)
    gerar_graficos(registros, resumo_sa, args.saida)
    print(f"Análise salva em: {args.saida.resolve()}")
    for linha in tabela:
        print(f"{linha['instancia']:7} {linha['operador']:9} {linha['estrategia']:5} "
              f"N={linha['repeticoes']:2} media={linha['fitness_medio']:10.2f} "
              f"min={linha['fitness_minimo']} tempo={linha['tempo_medio_s']}s")
    print(f"Registros SA: {len(parametros)}; blocos agregados: {len(resumo_sa)}")
    if any(t["repeticoes"] < 2 for t in tabela):
        print("ATENÇÃO: com apenas uma repetição não é possível avaliar a dispersão estatística.")


if __name__ == "__main__":
    main()