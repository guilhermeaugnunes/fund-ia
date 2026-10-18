"""Análise Estatística dos Resultados Experimentais - TP2 IA (UFMG)

Processa o arquivo 'resultados_gerais.csv' gerado pelo protocolo experimental:
1. Gera a Tabela 1 comparativa exigida na Seção 5:
   - 9 combinações (3 instâncias x 3 algoritmos)
   - Média ± Desvio Padrão, Mediana, Mínimo e Melhor h
   - Gap relativo ao ótimo conhecido na instância pequena: gap = (h - h*) / h*
   - Tempo médio de parede (segundos) e contagem de avaliações
2. Executa Testes Estatísticos Não-Paramétricos (Mann-Whitney / Wilcoxon):
   - Comparações par a par (Greedy vs SA, Greedy vs AG, SA vs AG) por instância
   - Reporta estatística U, z-score e p-valor bilateral (two-tailed)
3. Gera tabelas formatadas em Markdown e LaTeX prontas para o relatório em PDF.
"""

from __future__ import annotations

import csv
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Any

VALORES_REFERENCIA = {
    "pequena": 1875,   # Ótimo global por busca exata
    "media": 16332,    # Melhor valor conhecido
    "grande": 60417,   # Melhor valor conhecido
}


def calcular_mann_whitney_u(amostra1: list[float], amostra2: list[float]) -> tuple[float, float, float]:
    """Calcula o teste U de Mann-Whitney (rank-sum) em Python puro.
    
    Retorna: (estatistica_U, z_score, p_valor_bilateral)
    Usa aproximação assintótica normal com correção de empates (tied ranks).
    """
    n1 = len(amostra1)
    n2 = len(amostra2)
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 1.0

    # Combina e rotula os dados para atribuição de postos (ranks)
    combinados = [(val, 1) for val in amostra1] + [(val, 2) for val in amostra2]
    combinados.sort(key=lambda x: x[0])

    # Atribui postos médios a empates
    ranks = [0.0] * len(combinados)
    i = 0
    t_empates = []
    while i < len(combinados):
        j = i
        while j < len(combinados) and combinados[j][0] == combinados[i][0]:
            j += 1
        empate_len = j - i
        posto_medio = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = posto_medio
        if empate_len > 1:
            t_empates.append(empate_len)
        i = j

    r1 = sum(ranks[k] for k in range(len(combinados)) if combinados[k][1] == 1)
    u1 = r1 - (n1 * (n1 + 1)) / 2.0
    u2 = (n1 * n2) - u1
    u = min(u1, u2)

    # Média e variância sob H0
    mu = (n1 * n2) / 2.0
    n_total = n1 + n2
    termo_empate = sum(t**3 - t for t in t_empates) / (n_total * (n_total - 1)) if n_total > 1 else 0.0
    var_u = (n1 * n2 / 12.0) * ((n_total + 1) - termo_empate)
    sigma = math.sqrt(var_u) if var_u > 0 else 1.0

    # Correção de continuidade
    z = (abs(u - mu) - 0.5) / sigma if sigma > 0 else 0.0

    # P-valor bilateral via função de erro complementar
    p_valor = math.erfc(z / math.sqrt(2.0))
    return u, z, p_valor


def processar_resultados_csv(caminho_csv: pathlib.Path):
    if not caminho_csv.exists():
        print(f"Erro: Arquivo não encontrado em {caminho_csv}")
        return

    dados_por_chave = defaultdict(list)
    tempos_por_chave = defaultdict(list)
    evals_por_chave = defaultdict(list)

    with caminho_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chave = (row["instancia"], row["algoritmo"])
            h = float(row["melhor_fitness"])
            t = float(row["tempo_segundos"])
            ev = int(row["avaliacoes_utilizadas"])
            dados_por_chave[chave].append(h)
            tempos_por_chave[chave].append(t)
            evals_por_chave[chave].append(ev)

    print("=" * 95)
    print("TABELA PRINCIPAL - RESULTADOS COMPARATIVOS (SEÇÃO 5)")
    print("=" * 95)
    print(
        f"{'Instância':<10} | {'Algoritmo':<8} | {'N':<3} | {'Média ± DesvPad':<22} | "
        f"{'Mediana':<10} | {'Mínimo':<10} | {'Tempo Méd (s)':<14} | {'Gap Ref':<10}"
    )
    print("-" * 95)

    ordem_instancias = ["pequena", "media", "grande"]
    ordem_algos = ["greedy", "sa", "ag"]
    tabela_linhas = []

    for inst in ordem_instancias:
        h_ref = VALORES_REFERENCIA.get(inst, 0)
        for algo in ordem_algos:
            valores = dados_por_chave.get((inst, algo), [])
            tempos = tempos_por_chave.get((inst, algo), [])
            if not valores:
                continue

            n_amostras = len(valores)
            media = statistics.mean(valores)
            dp = statistics.stdev(valores) if n_amostras > 1 else 0.0
            mediana = statistics.median(valores)
            minimo = min(valores)
            tempo_med = statistics.mean(tempos) if tempos else 0.0
            gap_str = f"{(minimo - h_ref)/h_ref*100:+.2f}%" if h_ref > 0 else "-"

            linha_txt = (
                f"{inst:<10} | {algo.upper():<8} | {n_amostras:<3} | "
                f"{media:10.1f} ± {dp:<8.1f} | {mediana:10.1f} | {minimo:10.1f} | "
                f"{tempo_med:14.3f} | {gap_str:<10}"
            )
            print(linha_txt)
            tabela_linhas.append({
                "instancia": inst,
                "algoritmo": algo.upper(),
                "n": n_amostras,
                "media_dp": f"{media:.1f} ± {dp:.1f}",
                "mediana": f"{mediana:.1f}",
                "minimo": f"{minimo:.1f}",
                "tempo": f"{tempo_med:.3f}s",
                "gap": gap_str,
            })
        print("-" * 95)

    # TESTES DE HIPÓTESE NÃO-PARAMÉTRICOS (WILCOXON / MANN-WHITNEY)
    print("\n" + "=" * 95)
    print("TESTES NÃO-PARAMÉTRICOS DE MANN-WHITNEY (COMPARAÇÕES PAR A PAR)")
    print("=" * 95)
    print(f"{'Instância':<10} | {'Comparação':<22} | {'U':<8} | {'z-score':<8} | {'p-valor':<12} | {'Conclusão (alpha=0.05)':<25}")
    print("-" * 95)

    pares = [("greedy", "sa"), ("greedy", "ag"), ("sa", "ag")]
    for inst in ordem_instancias:
        for a1, a2 in pares:
            vals1 = dados_por_chave.get((inst, a1), [])
            vals2 = dados_por_chave.get((inst, a2), [])
            if not vals1 or not vals2:
                continue
            u, z, p = calcular_mann_whitney_u(vals1, vals2)
            if p < 0.001:
                p_str = "< 0.001"
            else:
                p_str = f"{p:.4f}"

            sig = "Diferença Significativa" if p < 0.05 else "Sem Dif. Significativa"
            print(f"{inst:<10} | {a1.upper()} vs {a2.upper():<13} | {u:8.1f} | {z:8.3f} | {p_str:<12} | {sig:<25}")
        print("-" * 95)

    # GERAÇÃO DE CÓDIGO LATEX
    print("\n>>> Código LaTeX da Tabela para o Relatório:")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Comparação de desempenho das heurísticas e metaheurística nas 3 instâncias.}")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\hline")
    print(r"Instância & Algoritmo & $N$ & Média $\pm$ Desv. Padrão & Mediana & Melhor $h$ & Tempo Médio (s) & Gap (\%) \\")
    print(r"\hline")
    for row in tabela_linhas:
        print(f"{row['instancia'].capitalize()} & {row['algoritmo']} & {row['n']} & {row['media_dp']} & {row['mediana']} & {row['minimo']} & {row['tempo']} & {row['gap']} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    caminho = pathlib.Path("tp2/resultados/resultados_gerais.csv")
    if len(sys.argv) > 1:
        caminho = pathlib.Path(sys.argv[1])
    processar_resultados_csv(caminho)
