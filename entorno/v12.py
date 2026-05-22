import functools
import random
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


class Entorno3SAT(ParallelEnv):
   

    metadata = {"render_modes": ["human"], "name": "voto_3sat_v13g_mappo"}

    LOCAL_OBS_DIM = 22
    GLOBAL_STATE_DIM = 42
    TOTAL_OBS_DIM = 64

    EXPECTATION_LEVELS = np.array([0.0, 0.5, 0.75, 0.875, 1.0], dtype=np.float32)
    NUM_ACTIONS = 3
    TERMINAL_REWARD = 100.0

    def __init__(self, num_agentes=40, num_variables=10, bonus_participacion=0.0):
        self.render_mode = None
        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.bonus_participacion = float(bonus_participacion)
        self.possible_agents = [f"agente_{i}" for i in range(num_agentes)]

        self.action_spaces = {
            agent: Discrete(self.NUM_ACTIONS)
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
        self.current_var = 0
        self.clausulas_privadas = {}

        if options is not None and "problema_inyectado" in options:
            self.clausulas_privadas = options["problema_inyectado"]
        else:
            for agent in self.agents:
                vars_interes = random.sample(range(self.num_variables), 3)
                signos = [random.choice([0, 1]) for _ in range(3)]
                self.clausulas_privadas[agent] = list(zip(vars_interes, signos))

        self.var_status = np.full(self.num_variables, -1, dtype=np.int32)

        # Demanda agregada estatica (solo para el critico)
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

        self.clause_maps = {}
        for agent in self.agents:
            mapa = np.zeros(self.num_variables, dtype=np.float32)
            for var_idx, deseo in self.clausulas_privadas[agent]:
                mapa[var_idx] = 1.0 if deseo == 1 else -1.0
            self.clause_maps[agent] = mapa

        self.expectation_idx = {
            agent: self._compute_true_expectation_idx(agent)
            for agent in self.agents
        }

        self.last_actions_history = {agent: [] for agent in self.agents}

        observations = {agent: self._crear_observacion(agent) for agent in self.agents}
        return observations, {}

    def step(self, actions):
        i = self.current_var

        pre_locked = {a: (self.expectation_idx[a] in (0, 4)) for a in self.agents}
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
            bonus = 0.0
            if self.bonus_participacion > 0.0 and not pre_locked[agent]:
                for var_idx, deseo in self.clausulas_privadas[agent]:
                    if var_idx == i and chosen_value == deseo:
                        a_int = raw_actions[agent]
                        push_truthful = (deseo == 1 and a_int == 1) or (deseo == 0 and a_int == 2)
                        if push_truthful:
                            bonus = self.bonus_participacion
                        break
            terminal = 0.0
            if terminado and self._clausula_satisfecha_final(agent):
                terminal = self.TERMINAL_REWARD
            rewards[agent] = bonus + terminal

        terminations = {agent: terminado for agent in self.agents}
        truncations = {agent: truncado for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        observations = {agent: self._crear_observacion(agent) for agent in self.agents}

        if terminado:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _action_to_reports(self, accion, current_idx):
        current_val = float(self.EXPECTATION_LEVELS[current_idx])
        if current_idx == 0 or current_idx == 4:
            return current_val, current_val
        if accion == 0:  # stay
            return current_val, current_val
        elif accion == 1:  # drop ("prefiero x=1")
            E0 = float(self.EXPECTATION_LEVELS[current_idx - 1])
            E1 = 1.0
            return E0, E1
        elif accion == 2:  # jump ("prefiero x=0")
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

    def _crear_observacion(self, agent):
        clausula = self.clause_maps[agent]
        var_status_f = self.var_status.astype(np.float32)
        exp_val = float(self.EXPECTATION_LEVELS[self.expectation_idx[agent]])
        paso_norm = self.num_pasos / max(self.num_variables, 1)

        local_obs = np.concatenate([
            clausula,                                            # 10
            var_status_f,                                        # 10
            np.array([exp_val], dtype=np.float32),               # 1
            np.array([paso_norm], dtype=np.float32),             # 1
        ])

        fraccion_sat = self._fraccion_satisfechos()
        global_state = np.concatenate([
            self.demanda_favor_norm,                             # 10
            self.demanda_contra_norm,                            # 10
            var_status_f,                                        # 10
            clausula,                                            # 10
            np.array([paso_norm], dtype=np.float32),             # 1
            np.array([fraccion_sat], dtype=np.float32),          # 1
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
