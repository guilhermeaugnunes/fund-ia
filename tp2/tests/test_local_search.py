import math
import pathlib
import sys
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from local_search import (AvaliacaoIncrementalSwap, greedy_random_restart,
                          simulated_annealing, tamanho_vizinhanca, _movimentos,
                          _aplicar_movimento)


class ProblemaDeTeste:
    def __init__(self, nome='manutencao_pequena.txt'):
        caminho = pathlib.Path(__file__).resolve().parents[1] / 'instancias' / nome
        linhas = caminho.read_text().splitlines()
        self.n = int(linhas[0])
        self.base = tuple(map(float, linhas[1].split()))
        self.dict_ativos = {}
        for idx, linha in enumerate(linhas[2:], 1):
            x, y, p, d, w = map(float, linha.split())
            self.dict_ativos[idx] = {'x': x, 'y': y, 'p': p, 'd': d, 'w': w}

    def solucao_aleatoria(self):
        return list(self.dict_ativos)

    def avaliar(self, solucao):
        pos = self.base
        tempo = 0.0
        custo = 0.0
        for a in solucao:
            d = self.dict_ativos[a]
            prox = (d['x'], d['y'])
            tempo += round(math.dist(pos, prox)) + d['p']
            custo += d['w'] * max(0, tempo-d['d'])
            pos = prox
        return custo

# Teste da busca local, incluindo greedy random restart e simulated annealing
class TesteBuscaLocal(unittest.TestCase):
    def setUp(self):
        self.p = ProblemaDeTeste()

    def test_referencia_4366(self):
        self.assertEqual(self.p.avaliar(list(range(1, 13))), 4366)

    def test_vizinhancas_sem_duplicatas(self):
        for op in ('swap', 'insertion', '2opt'):
            origem = list(range(1, 13))
            candidatos = [tuple(_aplicar_movimento(origem, op, i, j)) for i, j in _movimentos(12, op)]
            self.assertEqual(len(candidatos), len(set(candidatos)), op)
            self.assertEqual(len(candidatos), tamanho_vizinhanca(12, op), op)
            for c in candidatos:
                self.assertEqual(sorted(c), origem)

    def test_incremental_equivale_completa(self):
        for arquivo in ('manutencao_pequena.txt', 'manutencao_media.txt', 'manutencao_grande.txt'):
            p = ProblemaDeTeste(arquivo)
            seq = p.solucao_aleatoria()
            cache = AvaliacaoIncrementalSwap(p, seq)
            for i, j in list(_movimentos(len(seq), 'swap'))[::max(1, len(seq))]:
                self.assertEqual(cache.avaliar_troca(i, j), p.avaliar(_aplicar_movimento(seq, 'swap', i, j)))

    def test_greedy_estrategias_orcamento_e_seed(self):
        for modo in ('best', 'first'):
            a = greedy_random_restart(self.p, 240, estrategia=modo, seed=123, intervalo_historico=25)
            b = greedy_random_restart(self.p, 240, estrategia=modo, seed=123, intervalo_historico=25)
            self.assertEqual(a, b)
            self.assertEqual(a['avaliacoes'], 240)
            self.assertEqual(a['melhor_fitness'], self.p.avaliar(a['melhor_solucao']))
            self.assertEqual(a['historico'][-1][0], 240)
            self.assertTrue(all(y<=x for (_, x), (_, y) in zip(a['historico'], a['historico'][1:])))

    def test_greedy_operadores(self):
        for op in ('swap', 'insertion', '2opt'):
            a = greedy_random_restart(self.p, 240, operador=op, seed=1)
            self.assertEqual(a['avaliacoes'], 240)
            self.assertEqual(a['melhor_fitness'], self.p.avaliar(a['melhor_solucao']))

    def test_sa_calibracao_e_orcamento(self):
        a = simulated_annealing(self.p, 240, seed=34, amostras_calibracao=15, blocos=8)
        b = simulated_annealing(self.p, 240, seed=34, amostras_calibracao=15, blocos=8)
        self.assertEqual(a, b)
        self.assertEqual(a['avaliacoes'], 240)
        self.assertEqual(a['melhor_fitness'], self.p.avaliar(a['melhor_solucao']))
        self.assertEqual(a['historico'][-1][0], 240)
        self.assertEqual(len(a['metricas']['taxas_piores_por_bloco']), 8)
        self.assertLess(a['metricas']['Tf'], a['metricas']['T0'])

    def test_orcamentos_minimos(self):
        self.assertEqual(greedy_random_restart(self.p, 1, seed=1)['avaliacoes'], 1)
        self.assertEqual(simulated_annealing(self.p, 1, seed=1)['avaliacoes'], 1)
        self.assertEqual(simulated_annealing(self.p, 2, seed=1)['avaliacoes'], 2)


if __name__ == '__main__':
    unittest.main()