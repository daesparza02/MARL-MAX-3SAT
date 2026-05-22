import functools
import random
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv


class Entorno3SAT(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "voto_3sat_v9_secuencial_egoista"}

    LOCAL_OBS_DIM = 33    # clausula(10) + resultados(10) + presion_neta(10) + ya_satisfecho(1) + mi_deseo_actual(1) + var_actual(1)
    GLOBAL_STATE_DIM = 43 # demanda_favor(10) + demanda_contra(10) + resultados(10) + clausula(10) + ya_satisfecho(1) + var_actual(1) + frac_satisfechos(1)
    TOTAL_OBS_DIM = 76

    def __init__(self, num_agentes=40, num_variables=10):
        self.render_mode = None
        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.max_pasos = num_variables  # 1 paso por variable
        self.possible_agents = []
        for i in range (0,self.num_agentes):
            aux = "agente_" + str(i)
            self.possible_agents.append(aux)

        self.action_spaces = {
            agent: Discrete(3)
            for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(self.TOTAL_OBS_DIM,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_pasos = 0
        self.clausulas_privadas = {}

        if options is not None and "problema_inyectado" in options:
            self.clausulas_privadas = options["problema_inyectado"]
        else:
            for agent in self.agents:
                variables_disponibles = list(range(self.num_variables))
                vars_interes = random.sample(variables_disponibles, 3)

                clausula = []
                for var in vars_interes:
                    signo = random.choice([0, 1])
                    clausula.append((var, signo))

                self.clausulas_privadas[agent] = clausula

        self.demanda_favor = np.zeros(self.num_variables, dtype=np.float32)
        self.demanda_contra = np.zeros(self.num_variables, dtype=np.float32)
        for agent in self.agents:
            for var_idx, deseo in self.clausulas_privadas[agent]:
                if deseo == 1:
                    self.demanda_favor[var_idx] += 1.0
                else:
                    self.demanda_contra[var_idx] += 1.0
        self.demanda_favor_norm = self.demanda_favor / self.num_agentes
        self.demanda_contra_norm = self.demanda_contra / self.num_agentes
        self.presion_neta = (self.demanda_favor - self.demanda_contra) / self.num_agentes

        self.clause_maps = {}
        for agent in self.agents:
            mapa = np.zeros(self.num_variables, dtype=np.float32)
            for var_idx, deseo in self.clausulas_privadas[agent]:
                mapa[var_idx] = 1.0 if deseo == 1 else -1.0
            self.clause_maps[agent] = mapa

        # Orden de votacion: mas consenso primero
        # Calcular el consenso (diferencia absoluta entre favor y contra) para cada variable
        consenso = []
        for i in range(self.num_variables):
            diferencia = self.demanda_favor[i] - self.demanda_contra[i]
            if diferencia < 0:
                diferencia = -diferencia  # valor absoluto manual
            consenso.append(diferencia)

        # Ordenar las variables de mayor a menor consenso
        orden_votacion = []
        consenso_restante = consenso.copy()
        for _ in range(self.num_variables):
            indice_mayor = 0
            for i in range(self.num_variables):
                if consenso_restante[i] > consenso_restante[indice_mayor]:
                    indice_mayor = i
            orden_votacion.append(indice_mayor)
            consenso_restante[indice_mayor] = -1  # marcar como ya usado

        self.orden_votacion = orden_votacion

        # Estado: resultados de variables decididas (-1 = pendiente)
        self.resultados_decididos = np.full(self.num_variables, -1.0, dtype=np.float32)

        # Presion dinamica (inicialmente = estatica, nadie satisfecho aun)
        self._presion_neta_din = self.presion_neta.copy()
        self._favor_norm_din = self.demanda_favor_norm.copy()
        self._contra_norm_din = self.demanda_contra_norm.copy()

        # Para compatibilidad con evaluar: guardar votos por variable
        self.votos_favor = np.zeros(self.num_variables, dtype=np.int32)
        self.votos_contra = np.zeros(self.num_variables, dtype=np.int32)
        self.estado_votacion = np.zeros(self.num_variables, dtype=np.int32)

        # Historial de votos por agente (para heatmap)
        self.historial_votos = {agent: np.full(self.num_variables, -1, dtype=np.int32) for agent in self.agents}

        observations = {}
        for agent in self.agents:
            observations[agent] = self._crear_observacion(agent)
        return observations, {}

    def step(self, actions):
        # Variable que se vota en este paso
        var_idx = self.orden_votacion[self.num_pasos]
        self.num_pasos += 1

        favor = 0
        contra = 0
        for agent_id, accion in actions.items():
            self.historial_votos[agent_id][var_idx] = int(accion)
            if accion == 1:
                favor += 1
            elif accion == 0:
                contra += 1
            # accion == 2: abstencion, no cuenta

        self.votos_favor[var_idx] = favor
        self.votos_contra[var_idx] = contra
        self.estado_votacion[var_idx] = favor

        resultado = 1 if favor > contra else 0
        self.resultados_decididos[var_idx] = float(resultado)

        self._actualizar_presion_dinamica()

        terminado = (self.num_pasos >= self.max_pasos)
        rewards = {agent: 0.0 for agent in self.agents}

        if terminado:
            resultado_leyes = (self.resultados_decididos > 0.5).astype(int)
            for agent in self.agents:
                clausula = self.clausulas_privadas[agent]
                satisfecho = any(resultado_leyes[var_idx_c] == deseo for var_idx_c, deseo in clausula)
                rewards[agent] = 100.0 if satisfecho else 0.0

        terminations = {agent: terminado for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        observations = {}
        for agent in self.agents:
            observations[agent] = self._crear_observacion(agent)

        if terminado:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _actualizar_presion_dinamica(self):
        favor = np.zeros(self.num_variables, dtype=np.float32)
        contra = np.zeros(self.num_variables, dtype=np.float32)
        n_insatisfechos = 0
        for agent in self.possible_agents:
            if self._agente_satisfecho(agent):
                continue
            n_insatisfechos += 1
            for var_idx, deseo in self.clausulas_privadas[agent]:
                if deseo == 1:
                    favor[var_idx] += 1.0
                else:
                    contra[var_idx] += 1.0
        divisor = max(n_insatisfechos, 1)
        self._presion_neta_din = (favor - contra) / divisor
        self._favor_norm_din = favor / divisor
        self._contra_norm_din = contra / divisor

    def _agente_satisfecho(self, agent):
        for var_idx, deseo in self.clausulas_privadas[agent]:
            if self.resultados_decididos[var_idx] < 0:
                continue  # variable no decidida aun
            resultado = 1 if self.resultados_decididos[var_idx] > 0.5 else 0
            if resultado == deseo:
                return True
        return False

    def _crear_observacion(self, agent):
        clausula = self.clause_maps[agent]
        ya_satisfecho = 1.0 if self._agente_satisfecho(agent) else 0.0

        # Cual es la siguiente variable a votar (si no hemos terminado)
        if self.num_pasos < self.max_pasos:
            var_actual = self.orden_votacion[self.num_pasos]
        else:
            var_actual = self.orden_votacion[-1]

        var_actual_norm = float(var_actual) / self.num_variables
        mi_deseo_actual = clausula[var_actual]  # -1, 0, o +1

        # --- LOCAL (33 dims) ---
        local_obs = np.concatenate([
            clausula,                                    # 10
            self.resultados_decididos,                   # 10
            self._presion_neta_din,                      # 10 (dinamica: excluye satisfechos)
            np.array([ya_satisfecho], dtype=np.float32), # 1
            np.array([mi_deseo_actual], dtype=np.float32), # 1
            np.array([var_actual_norm], dtype=np.float32),  # 1
        ])

        # Fraccion de agentes ya satisfechos
        n_satisfechos = 0
        for agente in self.possible_agents:
            if self._agente_satisfecho(agente):
                n_satisfechos += 1
        frac_satisfechos = float(n_satisfechos) / self.num_agentes

        global_state = np.concatenate([
            self._favor_norm_din,                        # 10 (dinamica: excluye satisfechos)
            self._contra_norm_din,                       # 10 (dinamica: excluye satisfechos)
            self.resultados_decididos,                   # 10
            clausula,                                    # 10
            np.array([ya_satisfecho], dtype=np.float32), # 1
            np.array([var_actual_norm], dtype=np.float32),  # 1
            np.array([frac_satisfechos], dtype=np.float32), # 1
        ])

        return np.concatenate([local_obs, global_state]).astype(np.float32)
