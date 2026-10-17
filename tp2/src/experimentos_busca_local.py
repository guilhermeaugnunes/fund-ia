"""
Comandos para execução:
 python tp2/src/experimentos_busca_local.py --instancia pequena --orcamento 50000 --repeticoes 1
 python tp2/src/experimentos_busca_local.py --instancia todas --protocolo --repeticoes 1
"""
import argparse
import csv
import pathlib
import statistics
import time

from local_search import greedy_random_restart, simulated_annealing
from problem import ProblemaManutencaoAtivos

INSTANCIAS = {'pequena': ('manutencao_pequena.txt', 50000, 30),
              'media': ('manutencao_media.txt', 150000, 30),
              'grande': ('manutencao_grande.txt', 300000, 10)}


def executar(instancias, repeticoes, orcamento, saida, com_sa=True):
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    curvas = []
    taxas_sa = []
    otimos = []
    for instancia in instancias:
        arquivo, budget_padrao, rep_padrao = INSTANCIAS[instancia]
        budget = budget_padrao if orcamento is None else orcamento
        qtd = rep_padrao if repeticoes is None else repeticoes
        pasta = pathlib.Path(__file__).resolve().parents[1] / 'instancias'
        problema = ProblemaManutencaoAtivos(arquivo, pasta=str(pasta))
        
        # T2: comparar as 3 vizinhanças sob mesmo orçamento
        configuracoes = [('greedy', op, modo) for op in ('swap', 'insertion', '2opt')
                         for modo in ('best', 'first')]
        if com_sa:
            # Para integração, usar a vizinhança selecionada no T2
            # Aqui swap é apenas padrão inicial, não uma escolha definitiva
            configuracoes.append(('sa', 'swap', '-'))
        for algoritmo, operador, estrategia in configuracoes:
            resultados = []
            for i in range(qtd):
                seed = 1000 + i
                t0 = time.perf_counter()
                if algoritmo == 'greedy':
                    r = greedy_random_restart(problema, budget, operador=operador,
                                             estrategia=estrategia, seed=seed)
                else:
                    r = simulated_annealing(problema, budget, operador=operador, seed=seed)
                duracao = time.perf_counter() - t0
                resultados.append(r)
                linha = dict(instancia=instancia, algoritmo=algoritmo, operador=operador,
                             estrategia=estrategia, seed=seed, avaliacoes=r['avaliacoes'],
                             melhor_fitness=r['melhor_fitness'], segundos=round(duracao, 5),
                             passos_melhoria=r['metricas'].get('passos_melhoria'),
                             otimos_locais_distintos=r['metricas'].get('otimos_locais_distintos'),
                             random_restarts=r['metricas'].get('random_restarts'))
                linhas.append(linha)
                chaves = dict(instancia=instancia, algoritmo=algoritmo,
                              operador=operador, estrategia=estrategia, seed=seed)
                linha['melhor_solucao'] = ' '.join(map(str, r['melhor_solucao']))
                linha['T0'] = r['metricas'].get('T0')
                linha['Tf'] = r['metricas'].get('Tf')
                linha['alpha'] = r['metricas'].get('alpha')
                linha['taxa_aceitacao_piores_global'] = r['metricas'].get('taxa_aceitacao_piores_global')
                linha['taxa_aceitacao_total'] = r['metricas'].get('taxa_aceitacao_total')
                linha['piores_propostas'] = r['metricas'].get('piores_propostas')
                linha['piores_aceitas'] = r['metricas'].get('piores_aceitas')
                linha['inicializacoes'] = r['metricas'].get('inicializacoes')
                linha['incremental_swap'] = r['metricas'].get('incremental_swap')
                calibracao = r['metricas'].get('calibracao') or {}
                linha['escala_delta'] = calibracao.get('escala_delta')
                linha['amostras_piores'] = calibracao.get('amostras_piores')
                for avaliacoes, fitness in r['historico']:
                    curvas.append({**chaves, 'avaliacoes': avaliacoes,
                                   'melhor_fitness': fitness})
                for bloco in r['metricas'].get('taxas_piores_por_bloco', []):
                    taxas_sa.append({**chaves, **bloco})
                for indice, fitness in enumerate(r['metricas'].get('otimos_locais_fitness', []), 1):
                    otimos.append({**chaves, 'indice_otimo_local': indice, 'fitness_otimo_local': fitness})
                print(f'{instancia:7} {algoritmo:7} {operador:9} {estrategia:5} '
                      f'seed={seed} h={r["melhor_fitness"]:.0f} eval={r["avaliacoes"]} {duracao:.2f}s', flush=True)
            custos = [r['melhor_fitness'] for r in resultados]
            print('  RESUMO:', 'media=', round(statistics.mean(custos), 2),
                  'min=', min(custos), 'mediana=', statistics.median(custos), flush=True)
    def salvar_csv(nome, dados, colunas):
        with (saida / nome).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=colunas, delimiter=',')
            writer.writeheader()
            writer.writerows(dados)

    if linhas:
        salvar_csv('resultados_busca_local.csv', linhas, list(linhas[0]))
    salvar_csv('convergencia_busca_local.csv', curvas,
              ['instancia', 'algoritmo', 'operador', 'estrategia', 'seed', 'avaliacoes', 'melhor_fitness'])
    salvar_csv('aceitacao_sa.csv', taxas_sa,
              ['instancia', 'algoritmo', 'operador', 'estrategia', 'seed', 'bloco',
               'avaliacoes_inicio', 'avaliacoes_fim', 'piores_propostas',
               'piores_aceitas', 'taxa_aceitacao_piores'])
    salvar_csv('otimos_locais_greedy.csv', otimos,
              ['instancia', 'algoritmo', 'operador', 'estrategia', 'seed',
               'indice_otimo_local', 'fitness_otimo_local'])
    return linhas


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Comparação T2 e execução de busca local TP2')
    p.add_argument('--instancia', choices=(*INSTANCIAS, 'todas'), default='pequena')
    p.add_argument('--orcamento', type=int, default=None, help='Sobrescreve o orçamento (para testes rápidos)')
    p.add_argument('--repeticoes', type=int, default=1, help='Número de sementes por configuração')
    p.add_argument('--protocolo', action='store_true', help='Usa 30/30/10 execuções conforme o enunciado')
    p.add_argument('--sem-sa', action='store_true')
    p.add_argument('--saida', type=pathlib.Path, default=pathlib.Path('tp2/resultados_busca_local'))
    args = p.parse_args()
    executar(list(INSTANCIAS) if args.instancia == 'todas' else [args.instancia],
             None if args.protocolo else args.repeticoes, args.orcamento,
             args.saida, com_sa=not args.sem_sa)
