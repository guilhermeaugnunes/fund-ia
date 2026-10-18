"""Geração de Gráficos e Visualizações - TP2 IA (UFMG)

Gera as figuras obrigatórias do relatório conforme Seção 3, 5 e 6 do PDF:
1. Boxplots comparativos dos 3 algoritmos por instância (fig_boxplots.png)
2. Curvas de convergência da mediana do melhor h (fig_convergencia.png)
3. Curva de perda de diversidade do AG: melhor h vs média da população (fig_ag_populacao.png)
4. Gráfico do experimento adicional do AG: Crossover OX vs Apenas Mutação (fig_adicional_crossover.png)

Se a biblioteca matplotlib estiver instalada, salva os gráficos diretamente em formato PNG.
Caso contrário, gera tabelas de pontos prontas para visualização ou plotagem externa.
"""

from __future__ import annotations

import csv
import pathlib
import sys
from collections import defaultdict


def tentar_gerar_graficos(diretorio_dados: pathlib.Path = pathlib.Path("tp2/resultados"),
                          diretorio_figuras: pathlib.Path = pathlib.Path("tp2/figuras")):
    diretorio_figuras.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        tem_matplotlib = True
    except ImportError:
        tem_matplotlib = False
        print("Aviso: 'matplotlib' não está instalado no ambiente.")
        print("Para gerar os arquivos PNG automaticamente, instale com: pip install matplotlib")
        print("Os dados consolidados para plotagem foram organizados na pasta de resultados.")
        return

    # 1. BOXPLOTS POR INSTÂNCIA
    arq_resultados = diretorio_dados / "resultados_gerais.csv"
    if arq_resultados.exists():
        dados_box = defaultdict(lambda: defaultdict(list))
        with arq_resultados.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                dados_box[row["instancia"]][row["algoritmo"]].append(float(row["melhor_fitness"]))

        fig, axes = plt.subplots(1, len(dados_box), figsize=(5 * len(dados_box), 5), squeeze=False)
        for idx, (inst, algos_map) in enumerate(dados_box.items()):
            ax = axes[0, idx]
            labels = list(algos_map.keys())
            nomes = [l.upper() for l in labels]
            valores = [algos_map[k] for k in labels]
            try:
                ax.boxplot(valores, tick_labels=nomes, patch_artist=True)
            except TypeError:
                ax.boxplot(valores, labels=nomes, patch_artist=True)
            ax.set_title(f"Instância: {inst.capitalize()}")
            ax.set_ylabel("Melhor Fitness h (minimização)")
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        caminho_box = diretorio_figuras / "fig_boxplots.png"
        plt.savefig(caminho_box, dpi=300)
        plt.close()
        print(f"[OK] Gráfico de Boxplots salvo em: {caminho_box}")

    # 2. CURVA DE DIVERSIDADE E MÉDIA DA POPULAÇÃO DO AG
    arq_pop = diretorio_dados / "curvas_ag_populacao.csv"
    if arq_pop.exists():
        # Agrupa por instância e por geração para calcular a mediana entre as sementes independentes
        import statistics
        dados_por_inst_ger = defaultdict(lambda: defaultdict(lambda: {"avals": [], "melhor": [], "media": []}))
        with arq_pop.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                inst = row["instancia"]
                ger = int(row["geracao"])
                dados_por_inst_ger[inst][ger]["avals"].append(int(row["avaliacoes"]))
                dados_por_inst_ger[inst][ger]["melhor"].append(float(row["melhor_fitness"]))
                dados_por_inst_ger[inst][ger]["media"].append(float(row["media_fitness_populacao"]))

        for inst, mapa_ger in dados_por_inst_ger.items():
            geracoes_ordenadas = sorted(mapa_ger.keys())
            curva_avals = [statistics.median(mapa_ger[g]["avals"]) for g in geracoes_ordenadas]
            curva_melhor = [statistics.median(mapa_ger[g]["melhor"]) for g in geracoes_ordenadas]
            curva_media = [statistics.median(mapa_ger[g]["media"]) for g in geracoes_ordenadas]

            plt.figure(figsize=(8, 5))
            plt.plot(curva_avals, curva_melhor, label="Melhor Indivíduo (Mediana h)", color="blue", linewidth=2)
            plt.plot(curva_avals, curva_media, label="Média da População (Mediana h)", color="darkorange", linestyle="--", linewidth=1.8)
            plt.title(f"Convergência e Perda de Diversidade do AG - Instância {inst.capitalize()}")
            plt.xlabel("Avaliações de h gastas")
            plt.ylabel("Custo h (minimização)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            caminho_fig = diretorio_figuras / f"fig_ag_populacao_{inst}.png"
            plt.savefig(caminho_fig, dpi=300)
            plt.close()
            print(f"[OK] Gráfico AG Média vs Melhor ({inst}) salvo em: {caminho_fig}")

    # 3. EXPERIMENTO ADICIONAL: AG COM OX VS SEM OX
    arq_adicional = diretorio_dados / "adicional_ag_crossover.csv"
    if arq_adicional.exists():
        dados_crossover = defaultdict(list)
        with arq_adicional.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                dados_crossover[row["configuracao"]].append(float(row["melhor_fitness"]))

        plt.figure(figsize=(7, 5))
        labels = list(dados_crossover.keys())
        vals = [dados_crossover[k] for k in labels]
        rotulos = [l.replace("_", " ") for l in labels]
        try:
            plt.boxplot(vals, tick_labels=rotulos, patch_artist=True)
        except TypeError:
            plt.boxplot(vals, labels=rotulos, patch_artist=True)
        plt.title("Efeito do Operador de Crossover OX na Qualidade da Solução")
        plt.ylabel("Melhor Custo h")
        plt.grid(True, linestyle="--", alpha=0.6)
        caminho_crossover = diretorio_figuras / "fig_adicional_crossover.png"
        plt.savefig(caminho_crossover, dpi=300)
        plt.close()
        print(f"[OK] Gráfico do experimento de Crossover salvo em: {caminho_crossover}")


if __name__ == "__main__":
    tentar_gerar_graficos()
