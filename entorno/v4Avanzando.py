import functools
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
from pettingzoo import ParallelEnv

class Entorno3SAT(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "voto_3sat_v4_egoista_persistente"} 

    def __init__(self, num_agentes=40, num_variables=10):
        self.render_mode = None
        
        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.possible_agents = []
        for i in range (0,self.num_agentes):
            aux = "agente_" + str(i)
            self.possible_agents.append(aux)


        
        self.action_spaces = {
            agent: MultiDiscrete([3] * self.num_variables)
            for agent in self.possible_agents 
        }
        
        
        max_votos = self.num_agentes + 1
        estructura_obs = [self.num_agentes] + [3] * self.num_variables + [max_votos] * self.num_variables + [max_votos] * self.num_variables

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

        
        self.papeletas = {agent: np.full(self.num_variables, 2, dtype=np.int32) for agent in self.agents}
        self.votos_favor = np.zeros(self.num_variables, dtype=np.int32)
        self.votos_contra = np.zeros(self.num_variables, dtype=np.int32)
        
        # Variable legado para que el script evaluar2.py pueda seguir dibujando la gráfica
        self.estado_votacion = np.zeros(self.num_variables, dtype=np.int32)

        observations = {}
        for agent in self.agents:
            observations[agent] = self._crear_observacion(agent)
        return observations, {}

    def step(self, actions):
        self.num_pasos += 1
        
        
        for agent_id, accion_agente in actions.items():
            self.papeletas[agent_id] = accion_agente

        
        votos_favor = np.zeros(self.num_variables, dtype=np.int32)
        votos_contra = np.zeros(self.num_variables, dtype=np.int32)
        
        for agent_id in self.agents:
            papeleta = self.papeletas[agent_id]
            votos_favor += (papeleta == 1).astype(np.int32)
            votos_contra += (papeleta == 0).astype(np.int32)

        self.votos_favor = votos_favor
        self.votos_contra = votos_contra
        self.estado_votacion = votos_favor 

        
        resultado_final_leyes = []
        for i in range(self.num_variables):
            if self.votos_favor[i] > self.votos_contra[i]:
                resultado_final_leyes.append(1)
            else:
                resultado_final_leyes.append(0)
        
        LIMITE_PASOS = 10 
        terminado = (self.num_pasos >= LIMITE_PASOS)
        truncado = False
        
        rewards = {agent: 0.0 for agent in self.agents}
        multiplicador = 5.0 if terminado else 1.0

        for agent in self.agents:
            clausula = self.clausulas_privadas[agent]
            satisfecho = False
            
            for variable_idx, deseo_agente in clausula:
                if resultado_final_leyes[variable_idx] == deseo_agente:
                    satisfecho = True
                    break
            
            if satisfecho: 
                rewards[agent] = (100.0 * multiplicador) 
            else:
                rewards[agent] = 0.0
                
        terminations = {agent: terminado for agent in self.agents}
        truncations = {agent: truncado for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        observations = {}
        for agent in self.agents:
            observations[agent] = self._crear_observacion(agent)
            
        if terminado:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _crear_observacion(self, agent):
        agente_idx = int(agent.split("_")[1]) 

        mapa_leyes = [1] * self.num_variables
        clausula = self.clausulas_privadas[agent]
        for var_idx, signo in clausula:
            if signo == 1:
                mapa_leyes[var_idx] = 2  
            else:
                mapa_leyes[var_idx] = 0  
            
        obs = [agente_idx] + mapa_leyes + self.votos_favor.tolist() + self.votos_contra.tolist()
        return np.array(obs, dtype=np.int64)