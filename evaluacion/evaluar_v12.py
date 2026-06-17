import os
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# El entorno de V12 vive en ../entorno/ y el generador en ../entrenamiento/
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, "..", "entorno"))
sys.path.insert(0, os.path.join(_BASE, "..", "entrenamiento"))
from v12 import Entorno3SAT
from generador_mentira_v14 import generar_pool


# =====================================================================
# Arquitectura MAPPO anclada a V12 (actor 22 / critico 42)
# =====================================================================
class PassthroughExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space):
        super().__init__(observation_space, features_dim=observation_space.shape[0])

    def forward(self, obs):
        return obs


class SplitMlpExtractor(nn.Module):
    def __init__(self, feature_dim, local_dim=22, global_dim=42, hidden_size=128):
        super().__init__()
        self.local_dim = local_dim
        self.latent_dim_pi = hidden_size
        self.latent_dim_vf = hidden_size
        self.policy_net = nn.Sequential(
            nn.Linear(local_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(global_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features[:, :self.local_dim])

    def forward_critic(self, features):
        return self.value_net(features[:, self.local_dim:])


class MAPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = PassthroughExtractor
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = SplitMlpExtractor(
            feature_dim=self.features_dim,
            local_dim=Entorno3SAT.LOCAL_OBS_DIM,
            global_dim=Entorno3SAT.GLOBAL_STATE_DIM,
        )


# =====================================================================
# FASE A: sanidad sobre instancias aleatorias (satisfaccion + veracidad)
# =====================================================================
def ejecutar_partida_aleatoria(env, model):
    obs_dict, _ = env.reset()
    tcp, tcn, tfc = [], [], []
    terminado = False
    while not terminado:
        var_idx = env.current_var
        truthful = {a: env.get_truthful_action(a) for a in env.possible_agents}
        locked = {a: env.is_locked(a) for a in env.possible_agents}
        acciones = {}
        for agent in env.agents:
            action, _ = model.predict(obs_dict[agent], deterministic=True)
            acciones[agent] = action
        obs_dict, rewards, terms, truncs, infos = env.step(acciones)
        for agent in env.possible_agents:
            if locked[agent]:
                continue
            real = int(env.last_actions_history[agent][-1])
            ok = int(real == truthful[agent])
            vc = {v: d for v, d in env.clausulas_privadas[agent]}
            if var_idx in vc:
                (tcp if vc[var_idx] == 1 else tcn).append(ok)
            else:
                tfc.append(ok)
        terminado = all(terms.values())
    sat = sum(1 for a in env.possible_agents if env._clausula_satisfecha_final(a))
    sat_pct = sat / env.num_agentes * 100
    total = len(tcp) + len(tcn) + len(tfc)
    tot_ok = sum(tcp) + sum(tcn) + sum(tfc)
    truth_pct = (100 * tot_ok / total) if total > 0 else 0.0
    return sat_pct, truth_pct, tcp, tcn, tfc


# =====================================================================
# FASE B: test estrategico sobre instancias mentira-optimas held-out
# (las MISMAS que usa V14). V12, al ser veraz, deberia FALLAR aqui.
# =====================================================================
def ejecutar_test_mentira(env, model, instancia, meta):
    B = meta["var_apoyo"]
    v = meta["valor_apoyo"]
    P = meta["protagonista"]
    accion_apoyo = 1 if v == 1 else 2  # drop si v=1, jump si v=0
    obs_dict, _ = env.reset(options={"problema_inyectado": instancia})
    mintio = False
    terminado = False
    while not terminado:
        cur = env.current_var
        acciones = {}
        for agent in env.agents:
            action, _ = model.predict(obs_dict[agent], deterministic=True)
            acciones[agent] = action
        if cur == B:
            mintio = (int(acciones[P]) == accion_apoyo)
        obs_dict, rewards, terms, truncs, infos = env.step(acciones)
        terminado = all(terms.values())
    return mintio, env._clausula_satisfecha_final(P)


def evaluar(nombre_modelo):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(BASE_DIR, "..", "modelos definitivos", nombre_modelo)
    if not ruta.endswith(".zip"):
        ruta += ".zip"
    print(f"Buscando modelo en: {ruta}")
    if not os.path.exists(ruta):
        print("ERROR: No encuentro el modelo.")
        return
    ruta = ruta[:-4]

    env = Entorno3SAT(num_agentes=40, num_variables=10)
    model = PPO.load(ruta, custom_objects={"policy_class": MAPPOPolicy})
    print(f"Modelo V12 cargado: {nombre_modelo}\n")

    # ---------------- FASE A ----------------
    print("--- FASE A: SANIDAD (500 partidas aleatorias) ---")
    N = 500
    s_sat = s_truth = 0.0
    TCP, TCN, TFC = [], [], []
    for i in range(N):
        sat, truth, tcp, tcn, tfc = ejecutar_partida_aleatoria(env, model)
        s_sat += sat
        s_truth += truth
        TCP.extend(tcp); TCN.extend(tcn); TFC.extend(tfc)
        if i % 50 == 0:
            print(".", end="", flush=True)
    print("\n" + "=" * 50)
    print(f"  Satisfaccion media:  {s_sat / N:.2f}%")
    print(f"  Veracidad media:     {s_truth / N:.2f}%")
    if TCP:
        print(f"    deseo=1 (drop): {100*sum(TCP)/len(TCP):.2f}%")
    if TCN:
        print(f"    deseo=0 (jump): {100*sum(TCN)/len(TCN):.2f}%")
    if TFC:
        print(f"    fuera   (stay): {100*sum(TFC)/len(TFC):.2f}%")
    print("=" * 50)

    # ---------------- FASE B ----------------
    print("\n--- FASE B: TEST ESTRATEGICO (instancias mentira-optimas held-out) ---")
    M = 300
    print(f"Generando {M} instancias held-out nuevas...")
    pool = generar_pool(M, num_agentes=40, num_variables=10, seed=12345)
    n_mintio = n_exito = n_sat = 0
    for instancia, meta in pool:
        mintio, p_sat = ejecutar_test_mentira(env, model, instancia, meta)
        n_mintio += int(mintio)
        n_sat += int(p_sat)
        n_exito += int(mintio and p_sat)
    print("\n" + "=" * 50)
    print("RESULTADOS V12 - TEST ESTRATEGICO")
    print("=" * 50)
    print(f"  Instancias evaluadas:                 {M}")
    print(f"  El protagonista MIENTE (apoya B):     {100*n_mintio/M:.1f}%")
    print(f"  El protagonista acaba SATISFECHO:     {100*n_sat/M:.1f}%")
    print(f"  Exito estrategico (miente Y gana):    {100*n_exito/M:.1f}%")
    print("=" * 50)
    print("  Metrica limpia del avance = 'MIENTE': V12 es veraz (~5%) vs V14 (~99%).")
    print("  En satisfaccion, V12 puede salvar al protagonista por dinamica de")
    print("  poblacion (no por mentir); el avance real lo marca V14 mintiendo.")
    print("  Baseline veraz idealizado (todos perfectamente honestos): 0%.")


if __name__ == "__main__":
    # Por defecto carga modelos definitivos/v12.zip
    modelo = sys.argv[1] if len(sys.argv) > 1 else "v12.zip"
    evaluar(modelo)
