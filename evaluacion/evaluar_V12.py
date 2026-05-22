import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from mi_entorno_3sat_v13g_mappo import Entorno3SAT
from entrenar_mappo_v13g import MAPPOPolicy, SplitMlpExtractor


class MAPPOPolicyV13g(MAPPOPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitMlpExtractor(
            feature_dim=self.features_dim,
            local_dim=Entorno3SAT.LOCAL_OBS_DIM,
            global_dim=Entorno3SAT.GLOBAL_STATE_DIM,
        )


def generar_casos_laboratorio(num_variables, num_agentes=40):
    casos = {}
    casos["Caso A - Utopia"] = {f"agente_{i}": [(0, 1), (1, 0), (2, 0)] for i in range(num_agentes)}

    conflicto = {}
    for i in range(num_agentes):
        if i < num_agentes // 2:
            conflicto[f"agente_{i}"] = [(0, 1), (1, 0), (2, 0)]
        else:
            conflicto[f"agente_{i}"] = [(0, 0), (1, 0), (2, 0)]
    casos["Caso B - Conflicto"] = conflicto

    escalera = {}
    for i in range(num_agentes):
        if i < 10:
            escalera[f"agente_{i}"] = [(0, 1), (1, 1), (2, 1)]
        elif i < 20:
            escalera[f"agente_{i}"] = [(3, 0), (4, 0), (5, 0)]
        elif i < 30:
            escalera[f"agente_{i}"] = [(6, 1), (7, 0), (8, 1)]
        else:
            escalera[f"agente_{i}"] = [(9, 0), (0, 0), (1, 0)]
    casos["Caso C - Bloques Escalera"] = escalera

    pares_impares = {}
    for i in range(num_agentes):
        if i % 2 == 0:
            pares_impares[f"agente_{i}"] = [(0, 1), (2, 1), (4, 1)]
        else:
            pares_impares[f"agente_{i}"] = [(1, 0), (3, 0), (5, 0)]
    casos["Caso D - Pares e Impares"] = pares_impares

    cadena_rescate = {}
    for i in range(0, 10):
        cadena_rescate[f"agente_{i}"] = [(0, 1), (1, 1), (2, 1)]
    for i in range(10, 20):
        cadena_rescate[f"agente_{i}"] = [(0, 0), (3, 1), (4, 1)]
    for i in range(20, 30):
        cadena_rescate[f"agente_{i}"] = [(1, 0), (5, 1), (6, 1)]
    for i in range(30, 40):
        cadena_rescate[f"agente_{i}"] = [(2, 0), (7, 1), (8, 1)]
    casos["Caso E - Cadena de Rescate"] = cadena_rescate

    instancia_1 = [
        (1, 2, -3), (-4, 5, 6), (7, -8, 9), (-10, 1, -2), (3, 4, -5), (-6, -7, 8), (9, 10, -1), (-2, 3, 4),
        (5, -6, 7), (8, 9, 10), (-1, -3, 5), (2, 4, -6), (-7, 8, -9), (10, -1, 3), (-2, 5, -8), (4, 6, 7),
        (-9, -10, 2), (1, -4, 8), (3, -5, 9), (-6, 7, -10), (2, -3, 4), (-5, 6, -8), (1, 7, 9), (-2, -4, 10),
        (3, 5, -6), (-8, 9, -1), (4, -7, 10), (-2, 5, 8), (1, -6, 9), (3, -4, 7), (-5, 8, -10), (2, 6, -9),
        (-1, 4, -7), (3, 8, 10), (-2, -5, 9), (1, 6, -8), (-4, 7, -10), (2, -9, 3), (5, -1, 4), (-6, 8, 10)
    ]
    instancia_2 = [
        (-3, 1, 8), (2, -9, 5), (-4, 7, -10), (6, -1, 3), (-8, 2, 9), (5, -7, 4), (-10, 1, 6), (-2, 3, -5),
        (9, -8, 7), (-4, 1, -6), (10, -2, 3), (5, -9, 8), (-7, 4, -1), (6, -2, 10), (-3, 8, -5), (1, -9, 4),
        (-7, 2, -6), (10, -3, 5), (-8, 1, -4), (9, -2, 7), (-6, 3, -10), (5, -1, 8), (-4, 2, -9), (7, -3, 6),
        (-10, 1, -5), (8, -2, 4), (-9, 3, -7), (6, -1, 10), (-5, 2, -8), (4, -3, 9), (-7, 1, -6), (10, -2, 5),
        (-8, 3, -4), (9, -1, 7), (-6, 2, -10), (5, -3, 8), (-4, 1, -9), (7, -2, 6), (-10, 3, -5), (8, -1, 4)
    ]
    instancia_3 = [
        (2, -5, 8), (1, 9, -3), (-7, 4, -10), (6, -2, 5), (-8, 1, 9), (-3, 7, -4), (10, -6, 2), (5, -8, 1),
        (-9, 3, -7), (4, -10, 6), (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, -1, -2), (-3, -4, -5), (-6, -7, -8),
        (-9, -10, 1), (2, -3, 4), (-5, 6, -7), (8, -9, 10), (1, -4, 7), (-2, 5, -8), (3, -6, 9), (-10, 2, -4),
        (5, -7, 1), (-8, 3, -6), (9, -1, 4), (-2, 6, -10), (7, -3, 5), (-9, 8, -1), (-1, -3, 6), (2, 4, -7),
        (-5, 8, -9), (10, -2, 4), (-1, 5, -7), (3, -8, 10), (-4, 6, -9), (2, -1, 5), (-3, 7, -10), (8, -2, 4)
    ]
    instancia_4 = [
        (-1, -2, -3), (-4, -5, -6), (-7, -8, -9), (-10, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, -1), (-2, -4, -6),
        (-8, -10, 1), (3, 5, 7), (9, -2, 4), (-6, 8, -10), (1, -3, 5), (-7, 9, -1), (2, -4, 6), (-8, 10, -3),
        (5, -7, 9), (-1, 4, -6), (8, -10, 2), (-3, 7, -9), (1, 5, 10), (-2, -6, -9), (3, 4, 8), (-7, 1, -5),
        (2, -8, 10), (-3, 6, -1), (9, -4, 5), (-10, 7, -2), (1, -8, 3), (-6, 9, -4), (5, -2, 7), (-10, 1, -9),
        (8, -3, 6), (-4, 2, -7), (10, -5, 1), (-9, 6, -3), (7, -1, 4), (-8, 2, -5), (10, -6, 3), (-4, 9, -1)
    ]

    def convertir(inst):
        d = {}
        for i, cl in enumerate(inst):
            d[f"agente_{i}"] = [(abs(l) - 1, 1 if l > 0 else 0) for l in cl]
        return d

    casos["Caso AdHoc 1"] = convertir(instancia_1)
    casos["Caso AdHoc 2"] = convertir(instancia_2)
    casos["Caso AdHoc 3"] = convertir(instancia_3)
    casos["Caso AdHoc 4"] = convertir(instancia_4)
    return casos


def ejecutar_partida(env_raw, model, caso_datos=None):
    options = {"problema_inyectado": caso_datos} if caso_datos else None
    obs_dict, _ = env_raw.reset(options=options)

    tcp, tcn, tfc = [], [], []

    terminado = False
    while not terminado:
        var_idx = env_raw.current_var
        truthful = {a: env_raw.get_truthful_action(a) for a in env_raw.possible_agents}
        locked = {a: env_raw.is_locked(a) for a in env_raw.possible_agents}

        acciones = {}
        for agent in env_raw.agents:
            action, _ = model.predict(obs_dict[agent], deterministic=True)
            acciones[agent] = action

        obs_dict, rewards, terms, truncs, infos = env_raw.step(acciones)

        for agent in env_raw.possible_agents:
            if locked[agent]:
                continue
            real = int(env_raw.last_actions_history[agent][-1])
            ok = int(real == truthful[agent])
            vc = {v: d for v, d in env_raw.clausulas_privadas[agent]}
            if var_idx in vc:
                (tcp if vc[var_idx] == 1 else tcn).append(ok)
            else:
                tfc.append(ok)

        terminado = all(terms.values())

    resultado = env_raw.var_status.copy()
    sat = sum(1 for a in env_raw.possible_agents if env_raw._clausula_satisfecha_final(a))
    sat_pct = sat / env_raw.num_agentes * 100

    total = len(tcp) + len(tcn) + len(tfc)
    tot_ok = sum(tcp) + sum(tcn) + sum(tfc)
    truth_pct = (100 * tot_ok / total) if total > 0 else 0.0
    return resultado, sat_pct, truth_pct, tcp, tcn, tfc


def evaluar():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(BASE_DIR, "modelos", "mappo_3sat_v13g_mappo_final.zip")
    print(f"Buscando modelo en: {ruta}")
    if not os.path.exists(ruta):
        if not os.path.exists(ruta[:-4] + ".zip"):
            print("ERROR: No encuentro el modelo.")
            return
        ruta = ruta[:-4]

    env_raw = Entorno3SAT(num_agentes=40, num_variables=10, bonus_participacion=0.0)
    model = PPO.load(ruta, custom_objects={"policy_class": MAPPOPolicyV13g})
    print("Modelo cargado. Evaluando...")

    print("\n--- FASE 1: CASOS PERSONALIZADOS ---")
    for nombre, datos in generar_casos_laboratorio(10, 40).items():
        resultado, sat, truth, *_ = ejecutar_partida(env_raw, model, datos)
        print(f"\n{nombre}")
        print(f"  Asignacion final:  x = {[int(v) for v in resultado]}")
        print(f"  Satisfaccion:      {sat:.1f}%")
        print(f"  Truthfulness:      {truth:.1f}%")

    print("\n--- FASE 2: ESTADISTICAS GLOBALES (500 PARTIDAS) ---")
    N = 500
    s_sat = s_truth = 0.0
    TCP, TCN, TFC = [], [], []
    for i in range(N):
        resultado, sat, truth, tcp, tcn, tfc = ejecutar_partida(env_raw, model)
        s_sat += sat
        s_truth += truth
        TCP.extend(tcp); TCN.extend(tcn); TFC.extend(tfc)
        if i % 10 == 0:
            print(".", end="", flush=True)

    print("\n\n" + "=" * 50)
    print("RESULTADOS V13g (MAPPO solo, 500 partidas)")
    print("=" * 50)
    print(f"  Tasa de satisfaccion media:  {s_sat / N:.2f}%")
    print(f"  Tasa de truthfulness media:  {s_truth / N:.2f}%")
    print("=" * 50)
    print("  (excluyendo pasos con agente bloqueado)")
    print("\nTruthfulness por tipo de variable:")
    if TCP:
        print(f"  En clausula deseo=1 (drop):  {100*sum(TCP)/len(TCP):.2f}%   (n={len(TCP)})")
    if TCN:
        print(f"  En clausula deseo=0 (jump):  {100*sum(TCN)/len(TCN):.2f}%   (n={len(TCN)})")
    if TFC:
        print(f"  Fuera de clausula (stay):    {100*sum(TFC)/len(TFC):.2f}%   (n={len(TFC)})")

    try:
        plt.figure(figsize=(9, 5))
        vals = [100*sum(TCP)/max(len(TCP),1), 100*sum(TCN)/max(len(TCN),1), 100*sum(TFC)/max(len(TFC),1)]
        plt.bar(["deseo=1\n(drop)", "deseo=0\n(jump)", "fuera\n(stay)"], vals,
                color=['steelblue', 'salmon', 'lightgray'])
        plt.title(f"V13g (MAPPO solo) | Truthfulness por tipo\nGlobal: {s_truth/N:.1f}%  Sat: {s_sat/N:.1f}%")
        plt.ylabel("% acciones truthful")
        plt.ylim(0, 105)
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()
        plt.close('all')
    except Exception as e:
        print(f"Error grafica: {e}")


if __name__ == "__main__":
    evaluar()
