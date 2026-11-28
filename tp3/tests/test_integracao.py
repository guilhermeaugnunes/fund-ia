import time

import numpy as np

from src.agente import AgenteQLearning, criar_agenda_epsilon
from src.ambiente import Ambiente
from src.treinamento import extrair_tabela_politica, treinar_marvin


def test_episodio_completo_1000_passos():
    """Simula o loop exato que o agente da Pessoa 2 vai executar em 1 episódio."""
    env = Ambiente(tamanho=50, dens_lixo=0.20, dens_parede=0.10, seed=123)
    rng = np.random.default_rng(123)

    estado = env.reset()
    assert isinstance(estado, int)
    assert 0 <= estado <= 161

    recompensa_total = 0
    lixos_coletados = 0
    coletas_invalidas = 0
    movs_invalidos = 0

    for _ in range(1000):
        # Simula um agente escolhendo ações aleatórias (0 a 6)
        acao = int(rng.integers(0, 7))
        prox_estado, recompensa, info = env.step(acao)

        assert 0 <= prox_estado <= 161
        assert recompensa in (-5, -1, 0, 10)

        recompensa_total += recompensa
        if info["coletou"]:
            lixos_coletados += 1
        if info["coleta_invalida"]:
            coletas_invalidas += 1
        if info["mov_invalido"]:
            movs_invalidos += 1

        estado = prox_estado

    # Garante que o episódio rodou os 1000 passso sem quebrar
    assert (lixos_coletados + coletas_invalidas + movs_invalidos) <= 1000


def test_desempenho_500_episodios():
    """Verifica se 500 episodios de 1000 passos rodam rápido (abaixo de 5 segundos)."""
    env = Ambiente(tamanho=50, dens_lixo=0.20, dens_parede=0.10, seed=42)
    rng = np.random.default_rng(42)

    inicio = time.perf_counter()
    for _ in range(500):
        env.reset()
        for _ in range(1000):
            acao = int(rng.integers(0, 7))
            env.step(acao)
    duracao = time.perf_counter() - inicio

    print(f"\nTempo para 500 episódios (500.000 passos): {duracao:.2f}s")
    assert duracao < 5.0


def test_treinamento_marvin_convergencia_basica():
    """Verifica a integração completa Ambiente + AgenteQLearning em 60 episódios."""
    env = Ambiente(tamanho=50, dens_lixo=0.20, dens_parede=0.10, seed=42)
    agente = AgenteQLearning(alpha=0.1, gamma=0.9, q0=0.0, seed=42)
    agenda = criar_agenda_epsilon("exponencial", epsilon_0=1.0, epsilon_min=0.01, decaimento=0.95)

    hist = treinar_marvin(env, agente, agenda, n_episodios=60, horizonte=1000)

    assert hist["recompensas"].shape == (60,)
    assert hist["lixos_coletados"].shape == (60,)
    assert 0.0 < hist["fracao_visitada"] <= 1.0
    # Nos últimos episódios a recompensa média deve ser positiva e superior aos iniciais
    assert np.mean(hist["recompensas"][-10:]) > np.mean(hist["recompensas"][:10])

    tabela = extrair_tabela_politica(agente)
    assert len(tabela) == 162

