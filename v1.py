import functools
import random
import numpy as np
from gymnasium.spaces import Discrete, Box, MultiDiscrete
from pettingzoo import ParallelEnv

class Entorno3SAT(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "voto_3sat_v4_egoista"} #obligatorio por libreria

    def __init__(self, num_agentes=40, num_variables=10): #Constructor del entorno
        self.render_mode = None #(No tiene visualización grafica)
        
        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.possible_agents = []
        for i in range (0,num_agentes):
            aux=aux = "agente_" + str(i)
            self.possible_agents.append(aux)

        self.action_spaces = {
            agent: MultiDiscrete([2] * self.num_variables)
            for agent in self.possible_agents #Para iterar en las claves
        }
        
        

        self.observation_spaces = {
            agent: MultiDiscrete([self.num_agentes] + [3] * self.num_variables)
            for agent in self.possible_agents
        }
    @functools.lru_cache(maxsize=None)#Guarda los resultados de esta funcion en la cache, recomendado pero no obligatorio por libreria
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def reset(self, seed=None, options=None): #Funcion a la que se llama para empezar una nueva partida
        self.agents = self.possible_agents[:]#Hace una copia de self.possible_agentes, [:] sirve para no desaparecer con la eliminacion de possible_agents
        self.num_pasos = 0
        self.estado_votacion = np.zeros(self.num_variables, dtype=np.int32) #Inicializacion con 0
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
        
        LIMITE_PASOS = 1 #Partida de un solo turno
        terminado = (self.num_pasos >= LIMITE_PASOS)
        truncado = False
        
        rewards = {agent: 0.0 for agent in self.agents}#Inicializamos el diccionario de rewards

        if terminado:
            mayoria_absoluta = self.num_agentes / 2
            resultado_final_leyes = []
            for res in self.estado_votacion:
                if res > mayoria_absoluta:
                    resultado_final_leyes.append(1)
                else:
                    resultado_final_leyes.append(0)
            
            for agent in self.agents:
                clausula = self.clausulas_privadas[agent]
                satisfecho = False
                
                for variable_idx, deseo_agente in clausula:
                    if resultado_final_leyes[variable_idx] == deseo_agente:
                        satisfecho = True
                        break
                
                rewards[agent] = 100.0 if satisfecho else 0.0
                
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
        agente_idx = int(numero_en_texto)      # Ya tengo aqui el numero del agente
        # Mapa inicializado en 1 (Neutro)
        mapa_leyes = [1] * self.num_variables
        
        clausula = self.clausulas_privadas[agent]
        for var_idx, signo in clausula:
            # 2 (A favor) si el signo es 1, 0 (En contra) si el signo es 0
            if signo == 1:
                mapa_leyes[var_idx] = 2  
            else:
                mapa_leyes[var_idx] = 0  
        
        # Devolvemos una lista simple de 11 números enteros
        return np.array([agente_idx] + mapa_leyes, dtype=np.int64)