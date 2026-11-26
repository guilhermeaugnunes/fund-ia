# Documentação Técnica — Ambiente e Simulador (Tarefa 2a)

## 1. Representação da Grade e Parâmetros
O ambiente bidimensional foi implementado em Python utilizando a biblioteca NumPy como uma matriz de dimensões $50 \times 50$, seguindo a convenção de indexação `grade[y, x]`, onde a origem $(0, 0)$ situa-se no canto superior esquerdo, $x \in [0, 49]$ representa a coluna e $y \in [0, 49]$ representa a linha (crescendo para baixo).

Cada célula da matriz armazena um dos três códigos inteiros definidos no enunciado:
* `0`: Espaço vazio (célula ocupável sem lixo);
* `+1`: Lixo (célula ocupável com recompensa disponível);
* `-1`: Parede interna (obstáculo não ocupável).

Na geração estocástica do mapa (executada a cada chamada de `reset()` via gerador pseudoaleatório `np.random.default_rng`), foram adotadas as seguintes densidades:
* **Densidade de lixo:** $20\%$ (`dens_lixo = 0.20`), correspondendo a aproximadamente $500$ unidades de lixo distribuídas pela grade a cada episódio;
* **Densidade de paredes internas:** $10\%$ (`dens_parede = 0.10`), correspondendo a aproximadamente $250$ obstáculos internos.

A posição inicial de Marvin é fixada em $s_{start} = (0, 0)$ no início de cada episódio, garantindo-se explicitamente na inicialização que essa célula esteja sempre livre de paredes.

## 2. Percepção e Codificação de Estados
Marvin possui percepção estritamente local, observando apenas a célula atual e suas quatro adjacentes nas direções cardeais: $s = (c_{atual}, c_N, c_S, c_L, c_O)$. Células vizinhas situadas fora dos limites da grade ($x < 0$, $x \ge 50$, $y < 0$ ou $y \ge 50$) são interpretadas pelos sensores de forma idêntica às paredes internas.

Para indexar diretamente as linhas da Q-table ($|\mathcal{S}| = 2 \cdot 3^4 = 162$ estados), a quíntupla perceptual é mapeada em um inteiro único $\text{idx} \in \{0, 1, \dots, 161\}$ através de representação posicional em base mista:
$$\text{idx} = 81 \cdot c_{atual} + 27 \cdot c_N + 9 \cdot c_S + 3 \cdot c_L + c_O$$
onde $c_{atual} \in \{0:\text{vazio}, 1:\text{lixo}\}$ e $c_N, c_S, c_L, c_O \in \{0:\text{vazio}, 1:\text{lixo}, 2:\text{parede}\}$.

## 3. Dinâmica de Transição, Ações e Recompensas
O conjunto de ações $\mathcal{A}$ foi codificado com índices inteiros fixos de `0` a `6` (`0: norte`, `1: sul`, `2: leste`, `3: oeste`, `4: aleatória`, `5: parado`, `6: coletar`).
* **Ação estocástica (`aleatória`):** Sorteia uniformemente ($p = 0,25$ cada) uma das quatro direções cardeais antes de avaliar o deslocamento.
* **Colisões e bordas:** Qualquer tentativa de deslocamento para fora dos limites da grade ou contra uma célula de parede (`-1`) mantém o agente na mesma posição $(x, y)$ e aplica penalidade de $-5$.
* **Coleta:** Ao executar `coletar` sobre uma célula com lixo (`+1`), o agente recebe $+10$ e o conteúdo da célula é alterado para `0` (vazio), permanecendo assim até o final do episódio de $H = 1000$ passos. Caso não haja lixo na célula atual, aplica-se penalidade de $-1$.
* **Demais transições:** Deslocamentos válidos e a ação `parado` resultam em recompensa $0$.

O método `step(acao)` retorna a tupla `(proximo_estado, recompensa, info)`, em que o dicionário `info` contabiliza eventos por passo (`coletou`, `coleta_invalida` e `mov_invalido`) para rastreamento das métricas de treinamento.