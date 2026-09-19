from dataclasses import dataclass
from typing import Optional

# tipagem do estado em bool [p1, p2, p3, p4, tocha]
# False = margem origem, true = margem destino
Estado = tuple[bool, bool, bool, bool, bool]

# dicionario do problema: tempo que cada personagem leva para atravessar a ponte
# indice 0 = 1 min, indice 3 = 10 min
Tempos = [1, 2, 5, 10]


@dataclass
class No:
    """
    Representa um nó na árvore de busca.
    Colegas: Usem 'no.estado' para verificar se chegaram ao objetivo e
    'no.pai' para reconstruir o caminho no final.
    """

    estado: Estado  # armazena a tupla de bool
    pai: Optional["No"] = None
    acao: str | None = None
    # não é necessário no problema de BFS e DFS, mas ajudará a modelar o próximo
    custo_acumulado: int = 0
