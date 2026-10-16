import random

from utils import avaliar_fitness, carregar_instancia


class ProblemaManutencaoAtivos:
    def __init__(self, nome_arquivo_instancia, pasta="instancias"):
        self.n, self.base, self.dict_ativos = carregar_instancia(
            nome_arquivo_instancia, pasta
        )
        self.ids_ativos = list(self.dict_ativos.keys())

    def solucao_aleatoria(self):
        """Gera uma permutação aleatória (ponto de partida)"""
        solucao = self.ids_ativos.copy()
        random.shuffle(solucao)
        return solucao

    def avaliar(self, solucao):
        """Função de avaliação (fitness) h(s) - sempre de minimização"""
        return avaliar_fitness(solucao, self.base, self.dict_ativos)

    # GERADOR DE VIZINHANÇAS COMPLETAS (Busca Local)

    def sucessores_swap(self, solucao):
        """Gera todos os sucessores via operador Swap (troca)"""
        sucessores = []
        n = len(solucao)
        for i in range(n):
            for j in range(i + 1, n):
                nova_solucao = solucao.copy()
                nova_solucao[i], nova_solucao[j] = nova_solucao[j], nova_solucao[i]
                sucessores.append(nova_solucao)
        return sucessores

    def sucessores_insertion(self, solucao):
        """Gera todos os sucessores via operador Insertion (realocação)"""
        sucessores = []
        n = len(solucao)
        for i in range(n):
            elemento = solucao[i]
            temp = solucao[:i] + solucao[i + 1 :]
            for j in range(len(temp) + 1):
                nova_solucao = temp[:j] + [elemento] + temp[j:]
                if nova_solucao != solucao:
                    sucessores.append(nova_solucao)
        return sucessores

    def sucessores_2opt(self, solucao):
        """Gera todos os sucessores via operador 2-opt (inversão de subsegmento)"""
        sucessores = []
        n = len(solucao)
        for i in range(n):
            for j in range(i + 2, n + 1):
                nova_solucao = solucao.copy()
                nova_solucao[i:j] = reversed(nova_solucao[i:j])
                sucessores.append(nova_solucao)
        return sucessores

    # GERADORES DE VIZINHOS ÚNICOS / ALEATÓRIOS (SA / Random Walk)

    def vizinho_aleatorio_swap(self, solucao):
        """Retorna um único vizinho aleatório via Swap"""
        nova = solucao.copy()
        n = len(nova)
        i, j = random.sample(range(n), 2)
        nova[i], nova[j] = nova[j], nova[i]
        return nova

    def vizinho_aleatorio_2opt(self, solucao):
        """Retorna um único vizinho aleatório via 2-opt"""
        nova = solucao.copy()
        n = len(nova)
        i, j = sorted(random.sample(range(n), 2))
        nova[i : j + 1] = reversed(nova[i : j + 1])
        return nova

    def vizinho_aleatorio_insertion(self, solucao):
        """Retorna um único vizinho aleatório via Insertion (realocação)"""
        nova = solucao.copy()
        n = len(nova)
        i = random.randint(0, n - 1)
        elemento = nova.pop(i)
        j = random.randint(0, n - 1)
        nova.insert(j, elemento)
        return nova

    def sucessor_aleatorio(self, solucao, operador="swap"):
        """Sorteia um único sucessor aleatório aplicando o operador escolhido"""
        if operador == "swap":
            return self.vizinho_aleatorio_swap(solucao)
        elif operador == "2opt":
            return self.vizinho_aleatorio_2opt(solucao)
        elif operador == "insertion":
            return self.vizinho_aleatorio_insertion(solucao)
        else:
            raise ValueError(f"Operador desconhecido: {operador}")

    # OPERADORES GENÉTICOS (Cruzamento e Mutação)

    def cruzamento_ox(self, pai1, pai2):
        """Order Crossover (OX) para permutações (Algoritmo Genético)"""
        n = len(pai1)
        filho = [None] * n

        pt1, pt2 = sorted(random.sample(range(n), 2))
        filho[pt1 : pt2 + 1] = pai1[pt1 : pt2 + 1]

        idx_pai2 = (pt2 + 1) % n
        idx_filho = (pt2 + 1) % n

        for _ in range(n - (pt2 - pt1 + 1)):
            while pai2[idx_pai2] in filho:
                idx_pai2 = (idx_pai2 + 1) % n
            filho[idx_filho] = pai2[idx_pai2]
            idx_filho = (idx_filho + 1) % n

        return filho

    def mutacao_swap(self, solucao, pm=0.1):
        """Aplica mutação por troca com probabilidade pm"""
        nova = solucao.copy()
        if random.random() < pm:
            n = len(nova)
            i, j = random.sample(range(n), 2)
            nova[i], nova[j] = nova[j], nova[i]
        return nova
