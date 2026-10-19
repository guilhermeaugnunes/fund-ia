# Execução e Análise dos Experimentos

## Pré-requisitos

Certifique-se de ter o **Python 3.8 ou superior** instalado.

O projeto utiliza prioritariamente bibliotecas da biblioteca padrão do Python:

- `pathlib`
- `csv`
- `statistics`
- `time`
- `argparse`

Para a geração opcional de gráficos, é necessária a biblioteca `matplotlib`.

### Instalação das dependências de visualização

```bash
pip install matplotlib
```

## Instruções de Execução

Todos os comandos a seguir devem ser executados a partir da raiz do repositório.

### 1. Validação da Avaliação Incremental (Benchmark)

Para comparar o tempo de execução e verificar a correção matemática da avaliação incremental de *swap* em relação à avaliação completa:

```bash
python tp2/src/benchmark_incremental.py
```

### 2. Execução dos Experimentos Completos (Protocolo Oficial)

Para executar a bateria experimental unificada, contemplando os algoritmos **Greedy**, **Simulated Annealing** e **Algoritmo Genético**, respeitando os orçamentos oficiais de avaliações e as repetições regulamentares:

- **Instâncias pequenas e médias:** 30 repetições.
- **Instâncias grandes:** 10 repetições.

```bash
python tp2/src/experimentos_completos.py --protocolo
```

#### Execução de testes personalizados

Para executar testes rápidos com parâmetros definidos sob demanda:

```bash
python tp2/src/experimentos_completos.py --instancia pequena --algoritmo todos --orcamento 50000 --repeticoes 5
```

Nesse exemplo, são utilizados:

- **Instância:** pequena.
- **Algoritmos:** todos.
- **Orçamento:** 50.000 avaliações.
- **Repetições:** 5.

### 3. Análise Estatística e Testes de Hipótese

Para processar os resultados consolidados em CSV, calcular métricas descritivas e executar testes não paramétricos de Mann–Whitney par a par:

```bash
python tp2/src/analise_estatistica.py tp2/resultados_testes_03/resultados_gerais.csv
```

A análise contempla as seguintes métricas:

- Média.
- Desvio-padrão.
- Mediana.
- Quartis.
- *Gap* em relação à referência.
- Testes de hipótese de Mann–Whitney par a par.

### 4. Geração de Gráficos

Para gerar e exportar automaticamente as figuras dos experimentos em formato PNG:

```bash
python tp2/src/gerar_graficos.py
```

A geração dos gráficos inclui:

- Boxplots para comparação dos resultados.
- Curvas de convergência populacional do Algoritmo Genético (AG).
- Gráficos dos testes do operador de *crossover* OX.

**Observação:** a geração dos gráficos depende da instalação da biblioteca `matplotlib`.

## Resumo dos Comandos

| Etapa | Comando |
|---|---|
| Benchmark incremental | `python tp2/src/benchmark_incremental.py` |
| Experimentos oficiais | `python tp2/src/experimentos_completos.py --protocolo` |
| Testes personalizados | `python tp2/src/experimentos_completos.py --instancia pequena --algoritmo todos --orcamento 50000 --repeticoes 5` |
| Análise estatística | `python tp2/src/analise_estatistica.py tp2/resultados_testes_03/resultados_gerais.csv` |
| Geração de gráficos | `python tp2/src/gerar_graficos.py` |

## Fluxo de Execução Recomendado

1. Execute o benchmark para validar a avaliação incremental.
2. Execute os experimentos completos utilizando o protocolo oficial.
3. Processe os resultados consolidados com a análise estatística.
4. Gere os gráficos para visualizar e comparar o desempenho dos algoritmos.