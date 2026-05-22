import functools
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
from pettingzoo import ParallelEnv


class Entorno3SAT(ParallelEnv):
   

    metadata = {"render_modes": ["human"], "name": "voto_3sat_v13d_esperanzas_simple"}

    
    # Indice 0: stay  (indiferencia)
    # Indice 1: drop  (prefiere x_v = 1)
    # Indice 2: jump  (prefiere x_v = 0)
    ACTION_TO_E0 = np.array([0.875, 0.75, 1.0], dtype=np.float32)
    ACTION_TO_E1 = np.array([0.875, 1.0, 0.75], dtype=np.float32)

    def __init__(self, num_agentes=40, num_variables=10):
        self.render_mode = None

        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.possible_agents = [f"agente_{i}" for i in range(num_agentes)]

        # UNICO cambio respecto a V1: accion ternaria (3 valores por variable)
        self.action_spaces = {
            agent: MultiDiscrete([3] * self.num_variables)
            for agent in self.possible_agents
        }

        # Observacion identica a V1: DNI + clause map
        self.observation_spaces = {
            agent: MultiDiscrete([self.num_agentes] + [3] * self.num_variables)
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
                vars_interes = random.sample(range(self.num_variables), 3)
                signos = [random.choice([0, 1]) for _ in range(3)]
                self.clausulas_privadas[agent] = list(zip(vars_interes, signos))

        # Para evaluacion: rastrear las acciones (no afecta al entrenamiento)
        self.last_actions = {
            agent: np.zeros(self.num_variables, dtype=np.int32)
            for agent in self.agents
        }
        self.resultado_final = np.zeros(self.num_variables, dtype=np.int32)

        observations = {agent: self._crear_observacion(agent) for agent in self.agents}
        return observations, {}

    def step(self, actions):
        self.num_pasos += 1

        # Agregar contribuciones a E_0 y E_1 por variable
        sum_E0 = np.zeros(self.num_variables, dtype=np.float32)
        sum_E1 = np.zeros(self.num_variables, dtype=np.float32)
        for agent_id, accion in actions.items():
            accion_arr = np.asarray(accion, dtype=np.int32)
            self.last_actions[agent_id] = accion_arr.copy()
            sum_E0 += self.ACTION_TO_E0[accion_arr]
            sum_E1 += self.ACTION_TO_E1[accion_arr]

        # Arbitro: argmax por variable (empate gana 1)
        self.resultado_final = (sum_E1 >= sum_E0).astype(np.int32)

        LIMITE_PASOS = 1
        terminado = (self.num_pasos >= LIMITE_PASOS)
        truncado = False

        rewards = {}
        for agent in self.agents:
            clausula = self.clausulas_privadas[agent]
            satisfecho = any(
                self.resultado_final[var_idx] == deseo
                for var_idx, deseo in clausula
            )
            rewards[agent] = 100.0 if satisfecho else 0.0

        terminations = {agent: terminado for agent in self.agents}
        truncations = {agent: truncado for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        observations = {agent: self._crear_observacion(agent) for agent in self.agents}

        if terminado:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _crear_observacion(self, agent):
        agente_idx = int(agent.split("_")[1])
        mapa_leyes = [1] * self.num_variables
        for var_idx, signo in self.clausulas_privadas[agent]:
            mapa_leyes[var_idx] = 2 if signo == 1 else 0
        return np.array([agente_idx] + mapa_leyes, dtype=np.int64)

    def get_truthful_action(self, agent):
        
        truthful = np.zeros(self.num_variables, dtype=np.int32)
        for var_idx, deseo in self.clausulas_privadas[agent]:
            truthful[var_idx] = 1 if deseo == 1 else 2
        return truthful
