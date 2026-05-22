import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from mi_entorno_3sat_v13d_esperanzas_simple import Entorno3SAT


def generar_casos_laboratorio(num_variables, num_agentes=40):
    casos = {}

    utopia = {f"agente_{i}": [(0, 1), (1, 0), (2, 0)] for i in range(num_agentes)}
    casos["Caso A - Utopia"] = utopia

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

    def convertir_instancia(instancia_bruta):
        caso_formateado = {}
        for i, clausula in enumerate(instancia_bruta):
            clausula_formateada = []
            for literal in clausula:
                var_idx = abs(literal) - 1
                deseo = 1 if literal > 0 else 0
                clausula_formateada.append((var_idx, deseo))
            caso_formateado[f"agente_{i}"] = clausula_formateada
        return caso_formateado

    casos["Caso AdHoc 1"] = convertir_instancia(instancia_1)
    casos["Caso AdHoc 2"] = convertir_instancia(instancia_2)
    casos["Caso AdHoc 3"] = convertir_instancia(instancia_3)
    casos["Caso AdHoc 4"] = convertir_instancia(instancia_4)

    return casos


def ejecutar_partida_v13d(env_raw, model, caso_datos=None):
    options = {"problema_inyectado": caso_datos} if caso_datos else None
    obs_dict, _ = env_raw.reset(options=options)

    truthful_actions = {agent: env_raw.get_truthful_action(agent) for agent in env_raw.possible_agents}

    acciones_dict = {}
    for agent in env_raw.agents:
        obs_agente = obs_dict[agent]
        action, _ = model.predict(obs_agente, deterministic=True)
        acciones_dict[agent] = action

    obs_dict, rewards, terms, truncs, infos = env_raw.step(acciones_dict)

    total_decisiones = env_raw.num_agentes * env_raw.num_variables
    truthful_count = 0
    for agent in env_raw.possible_agents:
        accion_real = env_raw.last_actions[agent]
        accion_truth = truthful_actions[agent]
        truthful_count += int(np.sum(accion_real == accion_truth))

    truthfulness_pct = (truthful_count / total_decisiones) * 100

    resultado = env_raw.resultado_final
    clausulas_satisfechas = 0
    for agent in env_raw.possible_agents:
        for var_idx, deseo in env_raw.clausulas_privadas[agent]:
            if resultado[var_idx] == deseo:
                clausulas_satisfechas += 1
                break
    satisfaccion_pct = (clausulas_satisfechas / env_raw.num_agentes) * 100

    return resultado.copy(), satisfaccion_pct, truthfulness_pct


def evaluar():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    nombre_archivo = "ppo_3sat_v13d_esperanzas_simple_final.zip"
    ruta_modelo = os.path.join(BASE_DIR, "modelos", nombre_archivo)

    print(f"Buscando modelo en: {ruta_modelo}")

    if not os.path.exists(ruta_modelo):
        ruta_sin_ext = ruta_modelo[:-4]
        if not os.path.exists(ruta_sin_ext + ".zip"):
            print("ERROR: No encuentro el archivo del modelo.")
            return
        ruta_modelo = ruta_sin_ext

    env_raw = Entorno3SAT(num_agentes=40, num_variables=10)
    model = PPO.load(ruta_modelo)
    print("Modelo cargado. Evaluando...")

    print("\n--- FASE 1: CASOS PERSONALIZADOS ---")
    casos = generar_casos_laboratorio(10, 40)

    for nombre_caso, datos_caso in casos.items():
        resultado, satisfaccion, truthfulness = ejecutar_partida_v13d(env_raw, model, datos_caso)
        print(f"\n{nombre_caso}")
        print(f"  Asignacion final:  x = {[int(v) for v in resultado]}")
        print(f"  Satisfaccion:      {satisfaccion:.1f}%")
        print(f"  Veracidad:      {truthfulness:.1f}%")

    print("\n--- FASE 2: ESTADISTICAS GLOBALES (500 PARTIDAS ALEATORIAS) ---")
    total_partidas = 500
    suma_satis = 0.0
    suma_truth = 0.0

    truth_en_clausula_pos = []
    truth_en_clausula_neg = []
    truth_fuera_clausula = []

    for i in range(total_partidas):
        resultado, satisfaccion, truthfulness = ejecutar_partida_v13d(env_raw, model)
        suma_satis += satisfaccion
        suma_truth += truthfulness

        for agent in env_raw.possible_agents:
            accion = env_raw.last_actions[agent]
            truth_accion = env_raw.get_truthful_action(agent)
            vars_clausula = {v: d for v, d in env_raw.clausulas_privadas[agent]}
            for v in range(env_raw.num_variables):
                correcto = int(accion[v] == truth_accion[v])
                if v in vars_clausula:
                    if vars_clausula[v] == 1:
                        truth_en_clausula_pos.append(correcto)
                    else:
                        truth_en_clausula_neg.append(correcto)
                else:
                    truth_fuera_clausula.append(correcto)

        if i % 10 == 0:
            print(".", end="", flush=True)

    satis_media = suma_satis / total_partidas
    truth_media = suma_truth / total_partidas

    print("\n\n" + "=" * 50)
    print("RESULTADOS V13d (500 partidas aleatorias)")
    print("=" * 50)
    print(f"  Tasa de satisfaccion media:  {satis_media:.2f}%")
    print(f"  Tasa de veracidad media:  {truth_media:.2f}%")
    print("=" * 50)

    print("\nVeracidad desglosada por tipo de variable:")
    if truth_en_clausula_pos:
        pct = 100 * sum(truth_en_clausula_pos) / len(truth_en_clausula_pos)
        print(f"  Variables en clausula con deseo=1 (veracidad=drop):  {pct:.2f}%")
    if truth_en_clausula_neg:
        pct = 100 * sum(truth_en_clausula_neg) / len(truth_en_clausula_neg)
        print(f"  Variables en clausula con deseo=0 (veracidad=jump):  {pct:.2f}%")
    if truth_fuera_clausula:
        pct = 100 * sum(truth_fuera_clausula) / len(truth_fuera_clausula)
        print(f"  Variables fuera de clausula (veracidad=stay):        {pct:.2f}%")

    try:
        plt.figure(figsize=(9, 5))
        etiquetas = ["En clausula\n(deseo=1)\nveracidad=drop",
                     "En clausula\n(deseo=0)\nveracidad=jump",
                     "Fuera de\nclausula\nveracidad=stay"]
        valores = [
            100 * sum(truth_en_clausula_pos) / max(len(truth_en_clausula_pos), 1),
            100 * sum(truth_en_clausula_neg) / max(len(truth_en_clausula_neg), 1),
            100 * sum(truth_fuera_clausula) / max(len(truth_fuera_clausula), 1),
        ]
        plt.bar(etiquetas, valores, color=['steelblue', 'salmon', 'lightgray'])
        plt.title(f"V13d | Veracidad por tipo de variable (500 partidas)\nGlobal: {truth_media:.1f}%   Satisfaccion: {satis_media:.1f}%")
        plt.ylabel("% acciones veraces")
        plt.ylim(0, 105)
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()
        plt.close('all')
    except Exception as e:
        print(f"Error al generar grafica: {e}")


if __name__ == "__main__":
    evaluar()
