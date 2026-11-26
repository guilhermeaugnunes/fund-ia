import numpy as np

from src.ambiente import Ambiente


def test_inicializacao_e_reset():
    env = Ambiente(tamanho=50, dens_lixo=0.20, dens_parede=0.10, seed=42)

    # Verifica dimensões da grade
    assert env.grade.shape == (50, 50)

    # Marvin sempre começa em (0, 0) e essa célula não pode ser parede
    assert (env.x, env.y) == (0, 0)
    assert env.grade[0, 0] != Ambiente.PAREDE

    # Valores na grade devem ser apenas -1, 0 ou 1
    valores_unicos = set(np.unique(env.grade))
    assert valores_unicos.issubset({Ambiente.PAREDE, Ambiente.VAZIO, Ambiente.LIXO})


def test_limites_da_grade():
    env = Ambiente(tamanho=50, dens_lixo=0.0, dens_parede=0.0, seed=42)

    # Em (0, 0), tentar ir para NORTE ou OESTE deve falhar (-5) e manter posição
    for acao_invalida in [Ambiente.NORTE, Ambiente.OESTE]:
        _, rec, info = env.step(acao_invalida)
        assert rec == -5
        assert info["mov_invalido"] is True
        assert (env.x, env.y) == (0, 0)

    # Coloca Marvin no canto inferior direito (49, 49) e testa SUL e LESTE
    env.x, env.y = 49, 49
    for acao_invalida in [Ambiente.SUL, Ambiente.LESTE]:
        _, rec, info = env.step(acao_invalida)
        assert rec == -5
        assert info["mov_invalido"] is True
        assert (env.x, env.y) == (49, 49)


def test_movimento_valido_e_colisao_parede():
    env = Ambiente(tamanho=50, dens_lixo=0.0, dens_parede=0.0, seed=42)

    # Coloca uma parede à direita de (0, 0), ou seja, em (x=1, y=0)
    env.grade[0, 1] = Ambiente.PAREDE

    # Tentar ir para LESTE deve bater na parede
    _, rec, info = env.step(Ambiente.LESTE)
    assert rec == -5
    assert info["mov_invalido"] is True
    assert (env.x, env.y) == (0, 0)

    # Ir para SUL (x=0, y=1) deve funcionar e dar recompensa 0
    _, rec, info = env.step(Ambiente.SUL)
    assert rec == 0
    assert info["mov_invalido"] is False
    assert (env.x, env.y) == (0, 1)


def test_coleta_lixo_e_remocao():
    env = Ambiente(tamanho=50, dens_lixo=0.0, dens_parede=0.0, seed=42)

    # Coloca lixo onde Marvin está (0, 0)
    env.grade[0, 0] = Ambiente.LIXO

    # Primeira coleta: sucesso (+10) e lixo desaparece
    _, rec, info = env.step(Ambiente.COLETAR)
    assert rec == 10
    assert info["coletou"] is True
    assert info["coleta_invalida"] is False
    assert env.grade[0, 0] == Ambiente.VAZIO

    # Segunda coleta no mesmo local agora vazio: falha (-1)
    _, rec, info = env.step(Ambiente.COLETAR)
    assert rec == -1
    assert info["coletou"] is False
    assert info["coleta_invalida"] is True


def test_acao_parado():
    env = Ambiente(tamanho=50, dens_lixo=0.0, dens_parede=0.0, seed=42)
    env.x, env.y = 10, 10

    _, rec, info = env.step(Ambiente.PARADO)
    assert rec == 0
    assert (env.x, env.y) == (10, 10)
    assert not any(info.values())


def test_acao_aleatoria():
    env = Ambiente(tamanho=50, dens_lixo=0.0, dens_parede=0.0, seed=42)
    env.x, env.y = 25, 25  # Centro do mapa sem paredes

    _, rec, info = env.step(Ambiente.ALEATORIA)
    assert rec == 0
    assert info["mov_invalido"] is False
    # Deve ter andado exatamente 1 célula em alguma das 4 direções
    dist_manhattan = abs(env.x - 25) + abs(env.y - 25)
    assert dist_manhattan == 1


def test_codificacao_estados():
    env = Ambiente(tamanho=50, dens_lixo=0.0, dens_parede=0.0, seed=42)

    # Estado 0: tudo vazio (c_atual=0, N=0, S=0, L=0, O=0) -> no meio da grade sem lixo/parede
    env.x, env.y = 10, 10
    assert env.observar() == 0

    # Estado 161: lixo na célula atual (1) e paredes nos 4 vizinhos (2, 2, 2, 2)
    env.grade[10, 10] = Ambiente.LIXO
    env.grade[9, 10] = Ambiente.PAREDE  # Norte
    env.grade[11, 10] = Ambiente.PAREDE  # Sul
    env.grade[10, 11] = Ambiente.PAREDE  # Leste
    env.grade[10, 9] = Ambiente.PAREDE  # Oeste
    assert env.observar() == 161
