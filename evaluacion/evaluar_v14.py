import os
import numpy as np
from stable_baselines3 import PPO

from mi_entorno_3sat_v14_mentiraEstrategica import Entorno3SAT
from entrenar_mappo_v14 import MAPPOPolicy, SplitMlpExtractor
from generador_mentira_v14 import generar_pool


class MAPPOPolicyV14(MAPPOPolicy):
    def _build_mlp_extractor(self) -> None:
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
# =====================================================================
def ejecutar_test_mentira(env, model, instancia, meta):
    B = meta["var_apoyo"]
    v = meta["valor_apoyo"]
    P = meta["protagonista"]
    accion_apoyo = 1 if v == 1 else 2  # drop si v=1, jump si v=0

    obs_dict, _ = env.reset(options={"problema_inyectado": instancia, "info_mentira": meta})

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

    p_satisfecho = env._clausula_satisfecha_final(P)
    return mintio, p_satisfecho


def evaluar():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(BASE_DIR, "modelos", "mappo_3sat_v14_mentira_final.zip")
    print(f"Buscando modelo en: {ruta}")
    if not os.path.exists(ruta):
        if not os.path.exists(ruta[:-4] + ".zip"):
            print("ERROR: No encuentro el modelo.")
            return
        ruta = ruta[:-4]

    # Entorno SIN inyeccion automatica: controlamos la distribucion a mano
    env = Entorno3SAT(num_agentes=40, num_variables=10, p_inyeccion=0.0)
    model = PPO.load(ruta, custom_objects={"policy_class": MAPPOPolicyV14})
    print("Modelo cargado. Evaluando...\n")

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

    n_mintio = 0
    n_exito = 0  # mintio Y protagonista satisfecho
    n_sat = 0    # protagonista satisfecho (independientemente de si mintio)
    for instancia, meta in pool:
        mintio, p_sat = ejecutar_test_mentira(env, model, instancia, meta)
        if mintio:
            n_mintio += 1
        if p_sat:
            n_sat += 1
        if mintio and p_sat:
            n_exito += 1

    print("\n" + "=" * 50)
    print("RESULTADOS V14 - TEST ESTRATEGICO")
    print("=" * 50)
    print(f"  Instancias evaluadas:                 {M}")
    print(f"  El protagonista MIENTE (apoya B):     {100*n_mintio/M:.1f}%")
    print(f"  El protagonista acaba SATISFECHO:     {100*n_sat/M:.1f}%")
    print(f"  Exito estrategico (miente Y gana):    {100*n_exito/M:.1f}%")
    print("=" * 50)
    print("  Baseline veraz (por construccion): 0% satisfecho.")
    print("  -> Cuanto mas alto el exito estrategico, mas ha aprendido")
    print("     el agente a mentir cuando le conviene.")


if __name__ == "__main__":
    evaluar()
