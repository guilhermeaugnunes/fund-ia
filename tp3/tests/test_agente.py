import numpy as np
import pytest

from src.agente import (
    N_ACOES,
    N_ESTADOS,
    AgenteQLearning,
    codificar_estado,
    criar_agenda_epsilon,
    decodificar_estado,
    formatar_estado,
)
from src.treinamento import media_movel


def test_codificar_e_decodificar_todos_os_162_estados():
    vistos = set()
    for c_atual in (0, 1):
        for c_n in (0, 1, 2):
            for c_s in (0, 1, 2):
                for c_l in (0, 1, 2):
                    for c_o in (0, 1, 2):
                        idx = codificar_estado(c_atual, c_n, c_s, c_l, c_o)
                        assert 0 <= idx < N_ESTADOS
                        assert decodificar_estado(idx) == (c_atual, c_n, c_s, c_l, c_o)
                        vistos.add(idx)
    assert len(vistos) == N_ESTADOS


def test_codificar_e_decodificar_invalidos():
    with pytest.raises(ValueError):
        codificar_estado(2, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        codificar_estado(0, 3, 0, 0, 0)
    with pytest.raises(ValueError):
        decodificar_estado(-1)
    with pytest.raises(ValueError):
        decodificar_estado(162)


def test_formatar_estado():
    idx = codificar_estado(0, 2, 0, 1, 0)
    assert formatar_estado(idx) == "(vazio, parede, vazio, lixo, vazio)"


def test_inicializacao_neutra_q_table():
    agente = AgenteQLearning(alpha=0.1, gamma=0.9, q0=0.0, seed=7)
    assert agente.q_table.shape == (N_ESTADOS, N_ACOES)
    assert agente.visitas.shape == (N_ESTADOS, N_ACOES)
    assert np.all(agente.q_table == 0.0)
    assert np.all(agente.visitas == 0)
    assert agente.fracao_visitada() == 0.0


def test_desempate_aleatorio_no_argmax():
    agente = AgenteQLearning(seed=42)
    acoes = {agente.selecionar_acao(estado=0, epsilon=0.0) for _ in range(100)}
    # Com todos Q-valores empatados em 0.0, deve sortear múltiplas ações, não só índice 0
    assert len(acoes) > 1


def test_explotacao_escolhe_melhor_acao():
    agente = AgenteQLearning(seed=42)
    agente.q_table[5, 6] = 10.0
    for _ in range(20):
        assert agente.selecionar_acao(estado=5, epsilon=0.0) == 6


def test_exploracao_pura_visita_todas_as_acoes():
    agente = AgenteQLearning(seed=42)
    agente.q_table[0, 6] = 100.0
    acoes = {agente.selecionar_acao(estado=0, epsilon=1.0) for _ in range(200)}
    assert acoes == set(range(N_ACOES))


def test_atualizacao_td_q_learning():
    agente = AgenteQLearning(alpha=0.2, gamma=0.9, q0=0.0, seed=1)
    agente.q_table[10, 2] = 1.0
    agente.q_table[20, 6] = 5.0  # max_a' Q(20, a') = 5.0

    erro_td = agente.atualizar(estado=10, acao=2, recompensa=10.0, proximo_estado=20)
    # alvo = 10.0 + 0.9 * 5.0 = 14.5; erro = 14.5 - 1.0 = 13.5
    # novo Q = 1.0 + 0.2 * 13.5 = 3.7
    assert np.isclose(erro_td, 13.5)
    assert np.isclose(agente.q_table[10, 2], 3.7)
    assert agente.visitas[10, 2] == 1
    assert np.isclose(agente.fracao_visitada(), 1.0 / (N_ESTADOS * N_ACOES))


def test_agendas_epsilon():
    const = criar_agenda_epsilon("constante", epsilon_const=0.1)
    assert const(1) == 0.1 and const(500) == 0.1

    exp = criar_agenda_epsilon("exponencial", epsilon_0=1.0, epsilon_min=0.01, decaimento=0.99)
    assert np.isclose(exp(1), 0.99)
    assert exp(1000) == 0.01

    harm = criar_agenda_epsilon("harmonica", epsilon_min=0.01)
    assert harm(1) == 1.0
    assert np.isclose(harm(20), 0.05)
    assert harm(500) == 0.01


def test_media_movel():
    res = media_movel([1.0, 2.0, 3.0, 4.0], janela=2)
    assert np.allclose(res, [1.0, 1.5, 2.5, 3.5])
