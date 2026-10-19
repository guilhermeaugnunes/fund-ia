import math
import pathlib
import random
import sys
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from genetic import (
    cruzamento_ox,
    mutacao_swap,
    selecao_torneio,
    selecao_roleta_minimizacao,
    calcular_diversidade,
    genetic_algorithm,
)


class ProblemaDummy:
    """Problema sintético baseado na instância pequena para testes rápidos e isolados."""

    def __init__(self, nome="manutencao_pequena.txt"):
        caminho = pathlib.Path(__file__).resolve().parents[1] / "instancias" / nome
        linhas = caminho.read_text().splitlines()
        self.n = int(linhas[0])
        self.base = tuple(map(float, linhas[1].split()))
        self.dict_ativos = {}
        for idx, linha in enumerate(linhas[2:], 1):
            x, y, p, d, w = map(float, linha.split())
            self.dict_ativos[idx] = {"x": x, "y": y, "p": p, "d": d, "w": w}

    def solucao_aleatoria(self):
        return list(self.dict_ativos.keys())

    def avaliar(self, solucao):
        pos = self.base
        tempo = 0.0
        custo = 0.0
        for a in solucao:
            d = self.dict_ativos[a]
            prox = (d["x"], d["y"])
            tempo += round(math.dist(pos, prox)) + d["p"]
            custo += d["w"] * max(0.0, tempo - d["d"])
            pos = prox
        return custo


class TesteAlgoritmoGenetico(unittest.TestCase):
    def setUp(self):
        self.problema = ProblemaDummy()
        self.n = self.problema.n

    def test_cruzamento_ox_validade(self):
        rng = random.Random(42)
        pai1 = list(range(1, 13))
        pai2 = list(reversed(pai1))

        for _ in range(50):
            filho1, filho2 = cruzamento_ox(pai1, pai2, rng)
            self.assertEqual(len(filho1), 12)
            self.assertEqual(len(filho2), 12)
            self.assertEqual(sorted(filho1), pai1)
            self.assertEqual(sorted(filho2), pai1)
            self.assertEqual(len(set(filho1)), 12)
            self.assertEqual(len(set(filho2)), 12)

    def test_mutacao_swap_validade(self):
        rng = random.Random(42)
        base = list(range(1, 13))

        for _ in range(50):
            mutado = mutacao_swap(base, taxa_mutacao=1.0, rng=rng)
            self.assertEqual(len(mutado), 12)
            self.assertEqual(sorted(mutado), base)
            self.assertEqual(len(set(mutado)), 12)

    def test_selecao_torneio_e_roleta(self):
        rng = random.Random(42)
        populacao = [
            ([1, 2, 3], 100.0),
            ([2, 1, 3], 50.0),
            ([3, 2, 1], 200.0),
        ]
        # Com torneio cobrindo toda a população, o menor custo (50.0) deve vencer
        escolhido = selecao_torneio(populacao, tamanho_torneio=3, rng=rng)
        self.assertEqual(escolhido, [2, 1, 3])

        # Roleta deve retornar um indivíduo válido
        escolhido_roleta = selecao_roleta_minimizacao(populacao, rng=rng)
        self.assertIn(escolhido_roleta, [[1, 2, 3], [2, 1, 3], [3, 2, 1]])

    def test_orcamento_estrito(self):
        for orcamento in (1, 5, 20, 50, 150):
            r = genetic_algorithm(
                self.problema,
                orcamento=orcamento,
                tamanho_populacao=20,
                seed=123,
            )
            self.assertEqual(r["avaliacoes"], orcamento)
            self.assertEqual(r["melhor_fitness"], self.problema.avaliar(r["melhor_solucao"]))
            self.assertEqual(sorted(r["melhor_solucao"]), sorted(self.problema.solucao_aleatoria()))

    def test_reprodutibilidade_seed(self):
        res1 = genetic_algorithm(self.problema, orcamento=100, seed=777)
        res2 = genetic_algorithm(self.problema, orcamento=100, seed=777)

        self.assertEqual(res1["melhor_solucao"], res2["melhor_solucao"])
        self.assertEqual(res1["melhor_fitness"], res2["melhor_fitness"])
        self.assertEqual(res1["avaliacoes"], res2["avaliacoes"])
        self.assertEqual(res1["historico"], res2["historico"])

    def test_elitismo_monotonicidade(self):
        res = genetic_algorithm(
            self.problema,
            orcamento=300,
            tamanho_populacao=20,
            elitismo=2,
            seed=999,
        )
        historico = res["historico"]
        fitness_seq = [h for _, h in historico]
        for i in range(len(fitness_seq) - 1):
            self.assertGreaterEqual(
                fitness_seq[i],
                fitness_seq[i + 1],
                "O melhor fitness não pode piorar ao longo do tempo (elitismo)",
            )

    def test_curva_populacao_e_diversidade(self):
        res = genetic_algorithm(
            self.problema,
            orcamento=200,
            tamanho_populacao=20,
            seed=42,
        )
        curva = res["metricas"]["curva_populacao"]
        self.assertTrue(len(curva) >= 1)
        primeira_ger = curva[0]
        self.assertIn("media_fitness_populacao", primeira_ger)
        self.assertIn("cromossomos_distintos", primeira_ger)
        self.assertIn("taxa_diversidade", primeira_ger)
        self.assertTrue(0.0 <= primeira_ger["taxa_diversidade"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
