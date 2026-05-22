import functools
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
from pettingzoo import ParallelEnv


class Entorno3SAT(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "voto_3sat_v5_mappo"}

    # Dimensiones fijas para que el script de entrenamiento las lea
    LOCAL_OBS_DIM = 22    # Lo que ve el actor: clausula(10) + margen(10) + paso(1) + satisfaccion(1)
    GLOBAL_STATE_DIM = 42 # Lo que ve el critico: demanda_favor(10) + demanda_contra(10) + margen(10) + clausula(10) + paso(1) + satisfaccion(1)
    TOTAL_OBS_DIM = 64    # actor+critico concatenados

    def __init__(self, num_agentes=40, num_variables=10, max_pasos=10):
        self.render_mode = None

        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.max_pasos = max_pasos
        self.possible_agents = []
        for i in range (0,self.num_agentes):
            aux = "agente_" + str(i)
            self.possible_agents.append(aux)

       
        self.action_spaces = {
            agent: MultiDiscrete([3] * self.num_variables)
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

        #Papeletas es un diccionario que para cada agente guarda su votación
        self.papeletas = {
            agent: np.full(self.num_variables, 2, dtype=np.int32)
            for agent in self.agents
        }
        self.votos_favor = np.zeros(self.num_variables, dtype=np.int32)
        self.votos_contra = np.zeros(self.num_variables, dtype=np.int32)

        
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

        #Esto es mi mapa leyes que tenia antes en crear observacion. Lo hago aqui porque al ser multipaso es ineficiente inicializar todo el rato en crear observacion
        self.clause_maps = {}
        for agent in self.agents:
            mapa = np.zeros(self.num_variables, dtype=np.float32)
            for var_idx, deseo in self.clausulas_privadas[agent]:
                if deseo == 1:
                    mapa[var_idx] = 1.0  
                else:
                    mapa[var_idx] = -1.0
            self.clause_maps[agent] = mapa

        self.estado_votacion = np.zeros(self.num_variables, dtype=np.int32)

        observations = {}
        for agent in self.agents:
            observations[agent] = self._crear_observacion(agent)
        return observations, {}

    def step(self, actions):
        self.num_pasos += 1

        # Se actualiza el diccionario papeletas solo con las clausulas que le interesa al agente
        
        for agent_id, accion in actions.items():
            papeleta_limpia = np.full(self.num_variables, 2, dtype=np.int32)
            for var_idx, _ in self.clausulas_privadas[agent_id]:
                papeleta_limpia[var_idx] = accion[var_idx]
            self.papeletas[agent_id] = papeleta_limpia

        #Recuento 
        votos_favor = np.zeros(self.num_variables, dtype=np.int32)
        votos_contra = np.zeros(self.num_variables, dtype=np.int32)
        for agent_id in self.agents:
            papeleta = self.papeletas[agent_id]
            for i in range(self.num_variables):
                if papeleta[i] == 1:
                    votos_favor[i] += 1
                if papeleta[i] == 0:
                    votos_contra[i] += 1

        self.votos_favor = votos_favor
        self.votos_contra = votos_contra
        self.estado_votacion = votos_favor  

        resultado_leyes=[]
        for i in range(self.num_variables):
            if self.votos_favor[i] > self.votos_contra[i]:
                resultado_leyes.append(1)
            else:
                resultado_leyes.append(0)

        terminado = (self.num_pasos >= self.max_pasos)
        multiplicador = 5.0 if terminado else 1.0

        rewards = {}
        for agent in self.agents:
            clausula = self.clausulas_privadas[agent]
            satisfecho = False

            for var_idx, deseo in clausula:
                if resultado_leyes[var_idx] == deseo:
                    satisfecho = True
                    break

            if satisfecho:
                rewards[agent] = 100.0 * multiplicador
            else:
                 rewards[agent] = -100.0 * multiplicador

        terminations = {agent: terminado for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        observations = {}
        for agent in self.agents:
            observations[agent] = self._crear_observacion(agent)

        if terminado:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

   
    def _clausula_satisfecha(self, agent):
        resultado_leyes=[]
        for i in range(self.num_variables):
            if self.votos_favor[i] > self.votos_contra[i]:
                resultado_leyes.append(1)
            else:
                resultado_leyes.append(0)
        for var_idx, deseo in self.clausulas_privadas[agent]:
            if resultado_leyes[var_idx] == deseo:
                return True
        return False

    def _crear_observacion(self, agent):
        
        clausula = self.clause_maps[agent]  
        margen = [] #Margen normalizado del resultado de votacion de cada variable
        for i in range(self.num_variables):
            diferencia = self.votos_favor[i] - self.votos_contra[i]
            margen_variable = diferencia / max(self.num_agentes, 1)
            margen.append(margen_variable)
        margen = np.array(margen, dtype=np.float32)
        
        fraccion_pasos = self.num_pasos / self.max_pasos
        paso_norm = np.array([fraccion_pasos], dtype=np.float32)


        if self._clausula_satisfecha(agent):
            valor_satisfecho = 1.0
        else:
            valor_satisfecho = 0.0
        satisfecho  = np.array([valor_satisfecho], dtype=np.float32)

        
        local_obs = np.concatenate([
            clausula,    # 10: que variables me importan y en que direccion
            margen,      # 10: como va la votacion ahora mismo
            paso_norm,   # 1:  en que turno estamos
            satisfecho,  # 1:  estoy satisfecho ya?
        ])

        global_state = np.concatenate([
            self.demanda_favor_norm,   # 10: cuantos agentes quieren cada variable a favor
            self.demanda_contra_norm,  # 10: cuantos agentes quieren cada variable en contra
            margen,                    # 10: estado actual de la votacion
            clausula,                  # 10: clausula de este agente (para saber de quien es el valor)
            paso_norm,                 # 1:  turno actual
            satisfecho,                # 1:  este agente esta satisfecho?
        ])

        return np.concatenate([local_obs, global_state]).astype(np.float32)
