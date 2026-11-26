import numpy as np


class Ambiente:
    # Constantes para facilitar a leitura do código
    VAZIO = 0
    LIXO = 1
    PAREDE = -1

    # Ordem fixa das 7 ações (combinar com a Pessoa 2!)
    NORTE = 0
    SUL = 1
    LESTE = 2
    OESTE = 3
    ALEATORIA = 4
    PARADO = 5
    COLETAR = 6

    def __init__(self, tamanho=50, dens_lixo=0.20, dens_parede=0.10, seed=None):
        self.tamanho = tamanho
        self.dens_lixo = dens_lixo
        self.dens_parede = dens_parede
        self.rng = np.random.default_rng(seed)

        self.grade = None
        self.x = 0
        self.y = 0
        self.reset()

    def reset(self):
        """Reinicia o mapa para um novo episódio e coloca Marvin em (0, 0)."""
        # 1. Cria uma grade toda vazia (zeros)
        self.grade = np.zeros((self.tamanho, self.tamanho), dtype=int)

        # 2. Sorteia um número aleatório entre 0 e 1 para cada célula
        sorteio = self.rng.random((self.tamanho, self.tamanho))

        # 3. Coloca paredes e lixo com base nas densidades
        self.grade[sorteio < self.dens_parede] = self.PAREDE
        mascara_lixo = (sorteio >= self.dens_parede) & (
            sorteio < self.dens_parede + self.dens_lixo
        )
        self.grade[mascara_lixo] = self.LIXO

        # 4. Garante que a posição inicial (0, 0) nunca tenha parede
        if self.grade[0, 0] == self.PAREDE:
            self.grade[0, 0] = self.VAZIO

        # 5. Posiciona Marvin no início
        self.x = 0
        self.y = 0

        return self.observar()

    def _ler_celula(self, x, y):
        """Retorna o conteúdo de (x, y) já convertido para a codificação do sensor."""
        # Se estiver fora da grade, o sensor enxerga como parede (código 2)
        if x < 0 or x >= self.tamanho or y < 0 or y >= self.tamanho:
            return 2

        # Lembre-se: grade[linha, coluna] -> grade[y, x]
        valor = self.grade[y, x]
        if valor == self.VAZIO:
            return 0
        elif valor == self.LIXO:
            return 1
        else:  # PAREDE (-1)
            return 2

    def observar(self):
        """Retorna um número único de 0 a 161 representando o que Marvin vê."""
        # A célula atual nunca é parede, então retorna 0 (vazio) ou 1 (lixo)
        c_atual = self._ler_celula(self.x, self.y)

        # Vizinhos: Norte (y-1), Sul (y+1), Leste (x+1), Oeste (x-1)
        c_N = self._ler_celula(self.x, self.y - 1)
        c_S = self._ler_celula(self.x, self.y + 1)
        c_L = self._ler_celula(self.x + 1, self.y)
        c_O = self._ler_celula(self.x - 1, self.y)

        # Converte a quíntupla (c_atual, c_N, c_S, c_L, c_O) em um índice único (0 a 161)
        estado_idx = c_atual * 81 + c_N * 27 + c_S * 9 + c_L * 3 + c_O * 1
        return estado_idx

    def step(self, acao):
        """Executa uma ação e retorna (novo_estado, recompensa, info)."""
        info = {"coletou": False, "coleta_invalida": False, "mov_invalido": False}

        # Se a ação for ALEATORIA (4), sorteia uma das 4 direções (0, 1, 2 ou 3)
        if acao == self.ALEATORIA:
            acao = int(self.rng.integers(0, 4))

        # Caso 1: Ações de movimento (NORTE, SUL, LESTE, OESTE)
        if acao in (self.NORTE, self.SUL, self.LESTE, self.OESTE):
            nx, ny = self.x, self.y
            if acao == self.NORTE:
                ny -= 1
            elif acao == self.SUL:
                ny += 1
            elif acao == self.LESTE:
                nx += 1
            elif acao == self.OESTE:
                nx -= 1

            # Verifica se bateu na borda ou em uma parede interna
            fora_da_grade = nx < 0 or nx >= self.tamanho or ny < 0 or ny >= self.tamanho
            if fora_da_grade or self.grade[ny, nx] == self.PAREDE:
                recompensa = -5
                info["mov_invalido"] = True
            else:
                self.x, self.y = nx, ny
                recompensa = 0

        # Caso 2: Ficar PARADO
        elif acao == self.PARADO:
            recompensa = 0

        # Caso 3: Tentar COLETAR lixo
        elif acao == self.COLETAR:
            if self.grade[self.y, self.x] == self.LIXO:
                self.grade[self.y, self.x] = self.VAZIO  # Lixo desaparece
                recompensa = 10
                info["coletou"] = True
            else:
                recompensa = -1
                info["coleta_invalida"] = True
        else:
            raise ValueError(f"Ação inválida: {acao}")

        return self.observar(), recompensa, info
