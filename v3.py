import functools
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
from pettingzoo import ParallelEnv

class Entorno3SAT(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "voto_3sat_v3_mixto_opt"} 

    def __init__(self, num_agentes=40, num_variables=10):
        self.render_mode = None
        
        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.possible_agents = []
        for i in range (0,num_agentes):
            aux=aux = "agente_" + str(i)
            self.possible_agents.append(aux)

        self.action_spaces = {
            agent: MultiDiscrete([2] * self.num_variables)
            for agent in self.possible_agents 
        }
        
        # DNI (1) + Presión Favor (10) + Presión Contra (10) + Cláusula Privada (10)
        max_valor = self.num_agentes + 1
        estructura_obs = [self.num_agentes] + [max_valor] * (self.num_variables * 2) + [3] * self.num_variables

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
            for agent in self.agents:#Creación de las clausulas aleatorias
                variables_disponibles = list(range(self.num_variables))
                vars_interes = random.sample(variables_disponibles, 3)

                clausula = []
                for var in vars_interes:
                    signo = random.choice([0, 1])
                    clausula.append((var, signo))

                self.clausulas_privadas[agent] = clausula

        
        self.presion_favor = np.zeros(self.num_variables, dtype=np.int32)
        self.presion_contra = np.zeros(self.num_variables, dtype=np.int32)

        for agent in self.possible_agents:
            for var_idx, signo in self.clausulas_privadas[agent]:
                if signo == 1:
                    self.presion_favor[var_idx] += 1
                else:
                    self.presion_contra[var_idx] += 1

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
        
        LIMITE_PASOS = 1 
        terminado = (self.num_pasos >= LIMITE_PASOS)
        truncado = False
        
        rewards = {agent: 0.0 for agent in self.agents}

        if terminado:
            mayoria_absoluta = self.num_agentes / 2
            resultado_final_leyes = []
            for res in self.estado_votacion:
                if res > mayoria_absoluta:
                    resultado_final_leyes.append(1)
                else:
                    resultado_final_leyes.append(0)
            clausulas_satisfechas_totales = 0
            satisfecho_individual = {a: False for a in self.possible_agents}
            
            # 1. Evaluamos nivel global e individual
            for agent_eval in self.possible_agents:
                clausula = self.clausulas_privadas[agent_eval]
                for variable_idx, deseo_agente in clausula:
                    if resultado_final_leyes[variable_idx] == deseo_agente:
                        clausulas_satisfechas_totales += 1
                        satisfecho_individual[agent_eval] = True
                        break 
            
            # 2. RECOMPENSA MIXTA
            premio_comunitario = float(clausulas_satisfechas_totales) * 0.1
            
            for agent in self.agents:
                premio_egoista = 10.0 if satisfecho_individual[agent] else 0.0
                rewards[agent] = premio_egoista + premio_comunitario
                
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
        partes_del_texto = agent.split("_")  
        numero_en_texto = partes_del_texto[1]  
        agente_idx = int(numero_en_texto)
        
        
        mapa_privado = [1] * self.num_variables
        for var_idx, signo in self.clausulas_privadas[agent]:
            if signo == 1:
                mapa_privado[var_idx] = 2  
            else:
                mapa_privado[var_idx] = 0  
            
        
        obs = [agente_idx] + self.presion_favor.tolist() + self.presion_contra.tolist() + mapa_privado
        return np.array(obs, dtype=np.int64)