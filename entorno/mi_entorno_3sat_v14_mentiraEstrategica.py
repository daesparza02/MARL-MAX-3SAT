import functools
import random
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from generador_mentira_v14 import generar_pool


class Entorno3SAT(ParallelEnv):
    """
    V14: mecanica del arbitro 7/8 de V12, intacta, mas:
      - presion dinamica (demanda excluyendo satisfechos),
      - matriz de acoplamiento (cuantos opositores vivos de mis variables
        neutralizaria apoyando cada variable),
      - inyeccion de instancias donde mentir es optimo (p_inyeccion).

    Objetivo: estudiar si emerge la mentira estrategica (apoyar a un grupo
    para neutralizarlo y quitarse oposicion futura).
    """

    metadata = {"render_modes": ["human"], "name": "voto_3sat_v14_mentira"}

    LOCAL_OBS_DIM = 52    # clausula(10)+var_status(10)+exp(1)+paso(1)+presion_neta_din(10)+acoplamiento(20)
    GLOBAL_STATE_DIM = 42  # favor_din(10)+contra_din(10)+var_status(10)+clausula(10)+paso(1)+frac_sat(1)
    TOTAL_OBS_DIM = 94

    EXPECTATION_LEVELS = np.array([0.0, 0.5, 0.75, 0.875, 1.0], dtype=np.float32)
    NUM_ACTIONS = 3
    TERMINAL_REWARD = 100.0

    def __init__(self, num_agentes=40, num_variables=10,
                 p_inyeccion=0.0, tam_pool=300):
        self.render_mode = None
        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.p_inyeccion = float(p_inyeccion)
        self.possible_agents = [f"agente_{i}" for i in range(num_agentes)]

        self.action_spaces = {
            agent: Discrete(self.NUM_ACTIONS)
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(self.TOTAL_OBS_DIM,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Pool de instancias mentira-optimas (solo si se va a inyectar)
        self._pool = []
        if self.p_inyeccion > 0.0:
            print(f"[V14] Generando pool de {tam_pool} instancias mentira-optimas...")
            self._pool = generar_pool(tam_pool, num_agentes, num_variables)
            print("[V14] Pool listo.")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_pasos = 0
        self.current_var = 0
        self.info_mentira = None

        if options is not None and "problema_inyectado" in options:
            self.clausulas_privadas = {a: list(c) for a, c in options["problema_inyectado"].items()}
            if "info_mentira" in options:
                self.info_mentira = options["info_mentira"]
        elif self.p_inyeccion > 0.0 and self._pool and random.random() < self.p_inyeccion:
            clausulas, meta = random.choice(self._pool)
            self.clausulas_privadas = {a: list(c) for a, c in clausulas.items()}
            self.info_mentira = meta
        else:
            self.clausulas_privadas = {}
            for agent in self.agents:
                vars_interes = random.sample(range(self.num_variables), 3)
                signos = [random.choice([0, 1]) for _ in range(3)]
                self.clausulas_privadas[agent] = list(zip(vars_interes, signos))

        self.var_status = np.full(self.num_variables, -1, dtype=np.int32)

        # Mapa de clausula por agente (-1/0/+1)
        self.clause_maps = {}
        for agent in self.possible_agents:
            mapa = np.zeros(self.num_variables, dtype=np.float32)
            for var_idx, deseo in self.clausulas_privadas[agent]:
                mapa[var_idx] = 1.0 if deseo == 1 else -1.0
            self.clause_maps[agent] = mapa

        # Indice de literales: quien quiere cada (variable, deseo)
        self._quiere = {}
        for agent in self.possible_agents:
            for var_idx, deseo in self.clausulas_privadas[agent]:
                self._quiere.setdefault((var_idx, deseo), set()).add(agent)

        self.expectation_idx = {
            agent: self._compute_true_expectation_idx(agent)
            for agent in self.agents
        }

        self.last_actions_history = {agent: [] for agent in self.agents}

        self._actualizar_presion_dinamica()
        observations = {agent: self._crear_observacion(agent) for agent in self.agents}
        return observations, {}

    def step(self, actions):
        i = self.current_var
        raw_actions = {a: int(actions[a]) for a in actions}

        sum_E0 = 0.0
        sum_E1 = 0.0
        agent_reports = {}
        for agent_id, accion in actions.items():
            accion_int = int(accion)
            current_idx = self.expectation_idx[agent_id]
            E0, E1 = self._action_to_reports(accion_int, current_idx)
            sum_E0 += E0
            sum_E1 += E1
            agent_reports[agent_id] = (E0, E1)

        chosen_value = 1 if sum_E1 >= sum_E0 else 0
        self.var_status[i] = chosen_value

        for agent_id in self.agents:
            E0, E1 = agent_reports[agent_id]
            chosen_E = E1 if chosen_value == 1 else E0
            new_idx = int(np.argmin(np.abs(self.EXPECTATION_LEVELS - chosen_E)))
            self.expectation_idx[agent_id] = new_idx
            self.last_actions_history[agent_id].append(raw_actions[agent_id])

        self.current_var += 1
        self.num_pasos += 1

        terminado = (self.current_var >= self.num_variables)
        truncado = False

        rewards = {}
        for agent in self.agents:
            terminal = 0.0
            if terminado and self._clausula_satisfecha_final(agent):
                terminal = self.TERMINAL_REWARD
            rewards[agent] = terminal

        terminations = {agent: terminado for agent in self.agents}
        truncations = {agent: truncado for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self._actualizar_presion_dinamica()
        observations = {agent: self._crear_observacion(agent) for agent in self.agents}

        if terminado:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # ----------------------------------------------------------------
    # Mecanica del arbitro (identica a V12)
    # ----------------------------------------------------------------
    def _action_to_reports(self, accion, current_idx):
        current_val = float(self.EXPECTATION_LEVELS[current_idx])
        if current_idx == 0 or current_idx == 4:
            return current_val, current_val
        if accion == 0:
            return current_val, current_val
        elif accion == 1:
            E0 = float(self.EXPECTATION_LEVELS[current_idx - 1])
            E1 = 1.0
            return E0, E1
        elif accion == 2:
            E0 = 1.0
            E1 = float(self.EXPECTATION_LEVELS[current_idx - 1])
            return E0, E1
        else:
            return current_val, current_val

    def _compute_true_expectation_idx(self, agent):
        literales_libres = 0
        ya_satisfecha = False
        for var_idx, deseo in self.clausulas_privadas[agent]:
            estado = self.var_status[var_idx]
            if estado == -1:
                literales_libres += 1
            elif estado == deseo:
                ya_satisfecha = True
        if ya_satisfecha:
            return 4
        if literales_libres == 0:
            return 0
        if literales_libres == 1:
            return 1
        if literales_libres == 2:
            return 2
        return 3

    def _clausula_satisfecha_final(self, agent):
        for var_idx, deseo in self.clausulas_privadas[agent]:
            if self.var_status[var_idx] == deseo:
                return True
        return False

    def _fraccion_satisfechos(self):
        contador = 0
        for agent in self.possible_agents:
            for var_idx, deseo in self.clausulas_privadas[agent]:
                if self.var_status[var_idx] == deseo:
                    contador += 1
                    break
        return contador / self.num_agentes

    # ----------------------------------------------------------------
    # Senales nuevas de V14
    # ----------------------------------------------------------------
    def _actualizar_presion_dinamica(self):
        """Demanda favor/contra por variable excluyendo agentes satisfechos."""
        favor = np.zeros(self.num_variables, dtype=np.float32)
        contra = np.zeros(self.num_variables, dtype=np.float32)
        for agent in self.possible_agents:
            if self.is_locked(agent):
                continue
            for var_idx, deseo in self.clausulas_privadas[agent]:
                if deseo == 1:
                    favor[var_idx] += 1.0
                else:
                    contra[var_idx] += 1.0
        self._favor_din_norm = favor / self.num_agentes
        self._contra_din_norm = contra / self.num_agentes
        self._presion_neta_din = (favor - contra) / self.num_agentes

    def _matriz_acoplamiento(self, agent):
        """
        Para cada (variable B, valor v): que fraccion de mis opositores vivos
        quedaria satisfecha (y por tanto neutralizada) si B=v.
        Opositor = agente vivo que quiere alguna de mis variables aun sin
        decidir en el valor contrario al mio.
        """
        opositores = set()
        for A, dA in self.clausulas_privadas[agent]:
            if self.var_status[A] == -1:
                opositores |= self._quiere.get((A, 1 - dA), set())
        opositores = {g for g in opositores if not self.is_locked(g)}

        vec = np.zeros(self.num_variables * 2, dtype=np.float32)
        if opositores:
            for B in range(self.num_variables):
                if self.var_status[B] != -1:
                    continue
                for v in (0, 1):
                    sat = self._quiere.get((B, v), set())
                    vec[B * 2 + v] = len(opositores & sat) / self.num_agentes
        return vec

    def _crear_observacion(self, agent):
        clausula = self.clause_maps[agent]
        var_status_f = self.var_status.astype(np.float32)
        exp_val = float(self.EXPECTATION_LEVELS[self.expectation_idx[agent]])
        paso_norm = self.num_pasos / max(self.num_variables, 1)
        acoplamiento = self._matriz_acoplamiento(agent)

        local_obs = np.concatenate([
            clausula,                                   # 10
            var_status_f,                               # 10
            np.array([exp_val], dtype=np.float32),      # 1
            np.array([paso_norm], dtype=np.float32),    # 1
            self._presion_neta_din,                     # 10
            acoplamiento,                               # 20
        ])

        fraccion_sat = self._fraccion_satisfechos()
        global_state = np.concatenate([
            self._favor_din_norm,                       # 10
            self._contra_din_norm,                      # 10
            var_status_f,                               # 10
            clausula,                                   # 10
            np.array([paso_norm], dtype=np.float32),    # 1
            np.array([fraccion_sat], dtype=np.float32), # 1
        ])

        return np.concatenate([local_obs, global_state]).astype(np.float32)

    def get_truthful_action(self, agent):
        current_idx = self.expectation_idx[agent]
        if current_idx == 0 or current_idx == 4:
            return 0
        for var_idx, deseo in self.clausulas_privadas[agent]:
            if var_idx == self.current_var:
                return 1 if deseo == 1 else 2
        return 0

    def is_locked(self, agent):
        current_idx = self.expectation_idx[agent]
        return current_idx == 0 or current_idx == 4
