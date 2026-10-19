"""Análise estatística dos resultados de experimentos de otimização combinatória.

Produz tabela comparativa de desempenho (média, desvio padrão, mediana, mínimo, máximo,
gap relativo ao ótimo de referência) e testes não-paramétricos de Mann-Whitney entre pares
de algoritmos. Inclui análise do efeito do operador de recombinação no Algoritmo Genético.

Exporta relatório em texto simples e código LaTeX para uso em documentos técnicos.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Any

VALORES_REFERENCIA = {
    "pequena": 1875,   # Ótimo global conhecido (busca exata)
    "media": 16332,    # Melhor valor conhecido pelos professores
    "grande": 60417,   # Melhor valor conhecido pelos professores
}


def calcular_quartis(valores: list[float]) -> tuple[float, float, float]:
    """Calcula Q1 (25%), Mediana (50%) e Q3 (75%)."""
    if not valores:
        return 0.0, 0.0, 0.0
    vals = sorted(valores)
    n = len(vals)
    mediana = statistics.median(vals)

    if n == 1:
        return vals[0], vals[0], vals[0]

    metade_inf = vals[: n // 2]
    metade_sup = vals[(n + 1) // 2 :] if n % 2 != 0 else vals[n // 2 :]
    q1 = statistics.median(metade_inf) if metade_inf else vals[0]
    q3 = statistics.median(metade_sup) if metade_sup else vals[-1]
    return q1, mediana, q3


def calcular_mann_whitney_u(amostra1: list[float], amostra2: list[float]) -> dict[str, Any]:
    """Calcula o teste U de Mann-Whitney (rank-sum) em Python puro.
    
    Retorna dicionário completo com U, z-score, p-valor e tamanho de efeito.
    Usa aproximação normal com correção de empates (tied ranks) e continuidade.
    """
    n1 = len(amostra1)
    n2 = len(amostra2)
    if n1 == 0 or n2 == 0:
        return {"u": 0.0, "z": 0.0, "p_valor": 1.0, "tamanho_efeito_r": 0.0, "vargha_delaney_A": 0.5}

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

    mu = (n1 * n2) / 2.0
    n_total = n1 + n2
    termo_empate = sum(t**3 - t for t in t_empates) / (n_total * (n_total - 1)) if n_total > 1 else 0.0
    var_u = (n1 * n2 / 12.0) * ((n_total + 1) - termo_empate)
    sigma = math.sqrt(var_u) if var_u > 0 else 1.0

    z = (abs(u - mu) - 0.5) / sigma if sigma > 0 else 0.0
    p_valor = math.erfc(z / math.sqrt(2.0))

    # Tamanho do efeito r de Rosenthal: r = |z| / sqrt(N)
    r_efeito = abs(z) / math.sqrt(n_total) if n_total > 0 else 0.0

    # Vargha-Delaney A (probabilidade de superioridade de amostra1 sobre amostra2)
    # A = (R1/n1 - (n1+1)/2) / n2 = U1 / (n1 * n2)
    a_superioridade = u1 / (n1 * n2) if (n1 * n2) > 0 else 0.5

    return {
        "u": u,
        "u1": u1,
        "u2": u2,
        "z": z,
        "p_valor": p_valor,
        "tamanho_efeito_r": r_efeito,
        "vargha_delaney_A": a_superioridade,
    }


def classificar_efeito(r: float) -> str:
    """Classificação clássica do tamanho de efeito r (Cohen)."""
    if r < 0.1:
        return "Desprezível"
    elif r < 0.3:
        return "Pequeno"
    elif r < 0.5:
        return "Médio"
    else:
        return "Grande"


def analisar_resultados(
    caminho_csv: pathlib.Path,
    diretorio_saida: pathlib.Path | None = None,
):
    if not caminho_csv.exists():
        print(f"Erro: Arquivo não encontrado em {caminho_csv}")
        return

    dados_por_chave = defaultdict(list)
    tempos_por_chave = defaultdict(list)
    evals_por_chave = defaultdict(list)
    solucoes_por_chave = defaultdict(list)

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
            if "melhor_solucao" in row:
                solucoes_por_chave[chave].append(row["melhor_solucao"])

    ordem_instancias = ["pequena", "media", "grande"]
    ordem_algos = ["greedy", "sa", "ag"]
    nomes_algos = {"greedy": "Greedy (Random Restart)", "sa": "Simulated Annealing", "ag": "Algoritmo Genético"}

    # BUFFER DE TEXTO PARA RELATÓRIO
    linhas_relatorio = []
    def log(texto: str = ""):
        print(texto)
        linhas_relatorio.append(texto)

    log("=" * 105)
    log("RELATÓRIO ESTATÍSTICO COMPARATIVO DE ALGORITMOS DE OTIMIZAÇÃO")
    log(f"Arquivo analisado: {caminho_csv.resolve()}")
    log("=" * 105)

    # 1. TABELA PRINCIPAL DE DESEMPENHO
    log("\n1. TABELA PRINCIPAL DE DESEMPENHO")
    log("-" * 105)
    header = (
        f"{'Instância':<9} | {'Algoritmo':<7} | {'N':<3} | {'Média ± DesvPad':<22} | "
        f"{'Mediana [Q1; Q3]':<22} | {'Melhor h':<10} | {'Pior h':<10} | {'Gap Ref (Melhor)':<16}"
    )
    log(header)
    log("-" * 105)

    tabela_linhas_latex = []

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
            q1, mediana, q3 = calcular_quartis(valores)
            melhor_h = min(valores)
            pior_h = max(valores)
            tempo_med = statistics.mean(tempos) if tempos else 0.0

            gap_melhor_str = f"{(melhor_h - h_ref) / h_ref * 100:+.2f}%" if h_ref > 0 else "-"
            gap_medio_str = f"{(media - h_ref) / h_ref * 100:+.2f}%" if h_ref > 0 else "-"

            linha_txt = (
                f"{inst:<9} | {algo.upper():<7} | {n_amostras:<3} | "
                f"{media:10.1f} ± {dp:<8.1f} | {mediana:8.1f} [{q1:6.1f}; {q3:6.1f}] | "
                f"{melhor_h:10.1f} | {pior_h:10.1f} | {gap_melhor_str:<16}"
            )
            log(linha_txt)

            tabela_linhas_latex.append({
                "instancia": inst.capitalize(),
                "algoritmo": algo.upper(),
                "n": n_amostras,
                "media_dp": f"{media:.1f} $\\pm$ {dp:.1f}",
                "mediana": f"{mediana:.1f}",
                "q1_q3": f"[{q1:.1f}, {q3:.1f}]",
                "melhor": f"{melhor_h:.1f}",
                "pior": f"{pior_h:.1f}",
                "tempo": f"{tempo_med:.3f}s",
                "gap_melhor": gap_melhor_str,
                "gap_medio": gap_medio_str,
            })
        log("-" * 105)

    # 2. EFICIÊNCIA COMPUTACIONAL E TAXA DE SUCESSO (ÓTIMOS LOCAIS NA PEQUENA)
    log("\n2. EFICIÊNCIA COMPUTACIONAL, TEMPO DE PAREDE E TAXA DE SUCESSO NO ÓTIMO GLOBAL")
    log("-" * 105)
    log(f"{'Instância':<9} | {'Algoritmo':<7} | {'Tempo Médio (s)':<15} | {'Avaliações':<12} | {'Throughput (eval/s)':<20} | {'Sucesso h*=1875':<15}")
    log("-" * 105)

    for inst in ordem_instancias:
        for algo in ordem_algos:
            valores = dados_por_chave.get((inst, algo), [])
            tempos = tempos_por_chave.get((inst, algo), [])
            evals = evals_por_chave.get((inst, algo), [])
            if not valores:
                continue

            tempo_med = statistics.mean(tempos) if tempos else 0.001
            evals_med = statistics.mean(evals) if evals else 0
            throughput = evals_med / tempo_med if tempo_med > 0 else 0.0

            if inst == "pequena":
                sucessos = sum(1 for v in valores if abs(v - 1875.0) < 1e-4)
                taxa_sucesso_str = f"{sucessos}/{len(valores)} ({sucessos/len(valores)*100:.1f}%)"
            else:
                taxa_sucesso_str = "N/A"

            log(f"{inst:<9} | {algo.upper():<7} | {tempo_med:15.3f} | {int(evals_med):<12} | {throughput:20.1f} | {taxa_sucesso_str:<15}")
        log("-" * 105)

    # 3. TESTES DE HIPÓTESE NÃO-PARAMÉTRICOS (WILCOXON / MANN-WHITNEY)
    log("\n3. TESTES NÃO-PARAMÉTRICOS DE MANN-WHITNEY (COMPARAÇÕES PAR A PAR)")
    log("Objetivo: Testar se as diferenças de desempenho entre algoritmos possuem significância estatística.")
    log("-" * 105)
    log(f"{'Instância':<9} | {'Comparação':<20} | {'U':<8} | {'z-score':<8} | {'p-valor':<12} | {'Tamanho Efeito (r)':<20} | {'Conclusão (alpha=0.05)':<22}")
    log("-" * 105)

    pares = [("greedy", "sa"), ("greedy", "ag"), ("sa", "ag")]
    for inst in ordem_instancias:
        for a1, a2 in pares:
            vals1 = dados_por_chave.get((inst, a1), [])
            vals2 = dados_por_chave.get((inst, a2), [])
            if not vals1 or not vals2:
                continue

            resultado = calcular_mann_whitney_u(vals1, vals2)
            u = resultado["u"]
            z = resultado["z"]
            p = resultado["p_valor"]
            r_efeito = resultado["tamanho_efeito_r"]
            classif_efeito = classificar_efeito(r_efeito)

            p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"
            conclusao = "Diferença Significativa" if p < 0.05 else "Sem Dif. Significativa"

            log(f"{inst:<9} | {a1.upper()} vs {a2.upper():<13} | {u:8.1f} | {z:8.3f} | {p_str:<12} | {r_efeito:.3f} ({classif_efeito:<11}) | {conclusao:<22}")
        log("-" * 105)

    # 4. PROCESSAMENTO DO EXPERIMENTO ADICIONAL DO AG (CROSSOVER OX)
    arq_adicional = caminho_csv.parent / "adicional_ag_crossover.csv"
    if arq_adicional.exists():
        log("\n4. EXPERIMENTO ADICIONAL: PAPEL DO OPERADOR DE RECOMBINAÇÃO")
        log("Pergunta: 'A recombinação (OX) realmente consegue combinar bons trechos de dois pais em um filho melhor?'")
        log("-" * 105)

        dados_crossover = defaultdict(list)
        tempos_crossover = defaultdict(list)
        with arq_adicional.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                cfg = row["configuracao"]
                dados_crossover[cfg].append(float(row["melhor_fitness"]))
                if "tempo_segundos" in row:
                    tempos_crossover[cfg].append(float(row["tempo_segundos"]))

        configs = list(dados_crossover.keys())
        if len(configs) >= 2:
            c1, c2 = configs[0], configs[1]
            v1, v2 = dados_crossover[c1], dados_crossover[c2]
            res_teste = calcular_mann_whitney_u(v1, v2)

            for c in (c1, c2):
                vals = dados_crossover[c]
                media = statistics.mean(vals)
                dp = statistics.stdev(vals) if len(vals) > 1 else 0.0
                q1, med, q3 = calcular_quartis(vals)
                log(f"  Configuração: {c:<35} | Média = {media:10.1f} ± {dp:<6.1f} | Mediana = {med:8.1f} | Melhor = {min(vals):8.1f}")

            log("\n  >> Teste Estatístico Mann-Whitney entre AG com OX vs AG sem OX:")
            p_val = res_teste["p_valor"]
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            log(f"     Estatística U = {res_teste['u']:.1f} | z-score = {res_teste['z']:.3f} | p-valor = {p_str} | Efeito r = {res_teste['tamanho_efeito_r']:.3f}")

            if p_val < 0.05 and statistics.median(v1) < statistics.median(v2):
                log("     [CONCLUSÃO CIENTÍFICA]: A recombinação Order Crossover (OX) é ESTATISTICAMENTE SUPERIOR (p < 0.05) à busca puramente mutacional.")
                log("     Isso comprova a hipótese de que blocos de adjacência e sequenciamento transmitidos entre pais produzem filhos melhores.")
            else:
                log("     [CONCLUSÃO CIENTÍFICA]: Não houve evidência de superioridade estatística isolada do crossover sob as condições testadas.")
            log("-" * 105)

    # 5. GERADOR DE CÓDIGO LATEX
    log("\n5. CÓDIGO LATEX FORMATADO PARA O RELATÓRIO FINAL:")
    latex_code = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Desempenho comparativo dos algoritmos nas instâncias do problema de sequenciamento de manutenção.}",
        r"\label{tab:resultados_tp2}",
        r"\small",
        r"\begin{tabular}{llccccccc}",
        r"\hline",
        r"\textbf{Instância} & \textbf{Algoritmo} & $N$ & \textbf{Média $\pm$ Desv. Padrão} & \textbf{Mediana [Q1; Q3]} & \textbf{Melhor $h$} & \textbf{Pior $h$} & \textbf{Tempo (s)} & \textbf{Gap Ref.} \\",
        r"\hline",
    ]
    inst_anterior = ""
    for r in tabela_linhas_latex:
        if inst_anterior and r["instancia"] != inst_anterior:
            latex_code.append(r"\hline")
        inst_anterior = r["instancia"]
        latex_code.append(
            f"{r['instancia']} & {r['algoritmo']} & {r['n']} & {r['media_dp']} & "
            f"{r['mediana']} & {r['melhor']} & {r['pior']} & {r['tempo']} & {r['gap_melhor']} \\\\"
        )
    latex_code.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    codigo_latex_str = "\n".join(latex_code)
    log(codigo_latex_str)

    # SALVAMENTO EM DISCO (SE DIRETÓRIO FORNECIDO)
    saida_dir = diretorio_saida or caminho_csv.parent
    saida_dir.mkdir(parents=True, exist_ok=True)
    arquivo_txt = saida_dir / "relatorio_estatistico.txt"
    arquivo_tex = saida_dir / "tabela_latex.tex"

    arquivo_txt.write_text("\n".join(linhas_relatorio), encoding="utf-8")
    arquivo_tex.write_text(codigo_latex_str, encoding="utf-8")
    print(f"\n[OK] Relatório estatístico salvo em: {arquivo_txt.resolve()}")
    print(f"[OK] Código LaTeX salvo em: {arquivo_tex.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise estatística avançada do TP2")
    parser.add_argument("csv", nargs="?", default="tp2/resultados_testes_03/resultados_gerais.csv", help="Caminho do CSV de resultados gerais")
    parser.add_argument("--saida", type=pathlib.Path, default=None, help="Diretório para salvar os relatórios exportados")
    args = parser.parse_args()

    caminho_alvo = pathlib.Path(args.csv)
    analisar_resultados(caminho_alvo, args.saida)
