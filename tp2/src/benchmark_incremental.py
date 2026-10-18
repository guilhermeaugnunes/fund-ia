"""
Compara tempo da avaliação completa x incremental (swap) com os mesmos movimentos
Comando para execução: python tp2/src/benchmark_incremental.py
"""

import pathlib
import time
from local_search import AvaliacaoIncrementalSwap, _aplicar_movimento, _movimentos
from problem import ProblemaManutencaoAtivos

pasta = pathlib.Path(__file__).resolve().parents[1] / 'instancias'
for arquivo in ('manutencao_pequena.txt', 'manutencao_media.txt', 'manutencao_grande.txt'):
    p = ProblemaManutencaoAtivos(arquivo, pasta=str(pasta))
    solucao = sorted(p.ids_ativos)
    pares = list(_movimentos(len(solucao), 'swap'))
    
    # Até 500 movimentos por cenário para testes rápidos
    pares = pares[:500]
    
    t0 = time.perf_counter()
    resultado_completo = [p.avaliar(_aplicar_movimento(solucao,'swap',i,j)) for i,j in pares]
    tempo_completo = time.perf_counter() - t0
    t0 = time.perf_counter()
    cache = AvaliacaoIncrementalSwap(p, solucao)
    
    resultado_incremental = [cache.avaliar_troca(i,j) for i,j in pares]
    
    tempo_incremental = time.perf_counter() - t0
    assert resultado_completo == resultado_incremental, 'Erro na avaliação incremental!'
    print(f'{arquivo}: {len(pares)} movimentos; '
          f'completa={tempo_completo:.6f}s incremental={tempo_incremental:.6f}s '
          f'ganho={tempo_completo/tempo_incremental:.2f}x')