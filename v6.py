import functools
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
from pettingzoo import ParallelEnv


class Entorno3SAT(ParallelEnv):
    

    metadata = {"render_modes": ["human"], "name": "voto_3sat_v6_egoista_informado"}

    def __init__(self, num_agentes=40, num_variables=10):
        self.render_mode = None

        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.possible_agents = [f"agente_{i}" for i in range(num_agentes)]

        
        self.action_spaces = {
            agent: MultiDiscrete([2] * self.num_variables)
            for agent in self.possible_agents
        }

        
        max_presion = self.num_agentes + 1  # 0 a 40
        estructura_obs = (
            [self.num_agentes]
            + [3] * self.num_variables
            + [max_presion] * self.num_variables
            + [max_presion] * self.num_variables
        )

        self.observation_spaces = {
            agent: MultiDiscrete(estructura_obs)
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
        self.estado_votacion = np.zeros(self.num_variables, dtype=np.int32)
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

       
        # Para cada variable: cuantos agentes la quieren a favor / en contra
        self.presion_favor = np.zeros(self.num_variables, dtype=np.int32)
        self.presion_contra = np.zeros(self.num_variables, dtype=np.int32)

        for agent in self.agents:
            for var_idx, deseo in self.clausulas_privadas[agent]:
                if deseo == 1:
                    self.presion_favor[var_idx] += 1
                else:
                    self.presion_contra[var_idx] += 1

        
        self.votos_favor = np.zeros(self.num_variables, dtype=np.int32)
        self.votos_contra = np.zeros(self.num_variables, dtype=np.int32)

        observations = {}
        for agent in self.agents:
            observations[agent] = self._crear_observacion(agent)

        return observations, {}

    def step(self, actions):
        self.num_pasos += 1

       
        votos_ronda = np.zeros(self.num_variables, dtype=np.int32)
        for agent_id, accion_agente in actions.items():
            votos_ronda += accion_agente

        self.estado_votacion = votos_ronda
        self.votos_favor = votos_ronda.copy()
        self.votos_contra = self.num_agentes - votos_ronda

        
        LIMITE_PASOS = 1
        terminado = (self.num_pasos >= LIMITE_PASOS)

        rewards = {agent: 0.0 for agent in self.agents}

        if terminado:
            mayoria_absoluta = self.num_agentes / 2
            resultado_final = []
            for res in self.estado_votacion:
                if res > mayoria_absoluta:
                    resultado_final.append(1)
                else:
                    resultado_final.append(0)

            for agent in self.agents:
                clausula = self.clausulas_privadas[agent]
                satisfecho = False

                for variable_idx, deseo_agente in clausula:
                    if resultado_final[variable_idx] == deseo_agente:
                        satisfecho = True
                        break

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

    def _crear_observacion(self, agent):
        partes_del_texto = agent.split("_")  
        numero_en_texto = partes_del_texto[1]  
        agente_idx = int(numero_en_texto)

        # Mapa de clausula (identico a V1)
        mapa_leyes = [1] * self.num_variables
        clausula = self.clausulas_privadas[agent]
        for var_idx, signo in clausula:
            if signo == 1:
                mapa_leyes[var_idx] = 2  
            else:
                mapa_leyes[var_idx] = 0 

        # Presion de demanda 
        presion_f = self.presion_favor.tolist()
        presion_c = self.presion_contra.tolist()

        return np.array([agente_idx] + mapa_leyes + presion_f + presion_c,dtype=np.int64)
