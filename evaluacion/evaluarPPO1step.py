import os
from random import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from v1 import Entorno3SAT


def generar_casos_laboratorio(num_variables, num_agentes=40):
    casos = {}

    utopia = {f"agente_{i}": [(0, 1), (1, 0), (2, 0)] for i in range(num_agentes)}
    casos["Caso A - Utopia (Facil)"] = utopia

    conflicto = {}
    for i in range(num_agentes):
        if i < num_agentes // 2:
            conflicto[f"agente_{i}"] = [(0, 1), (1, 0), (2, 0)]
        else:
            conflicto[f"agente_{i}"] = [(0, 0), (1, 0), (2, 0)]
    casos["Caso B - Conflicto (Medio)"] = conflicto

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


def graficar_evolucion(matriz_votos, titulo, tasa_satisfaccion):
    try:
        plt.figure(figsize=(12, 10))

        ax = sns.heatmap(matriz_votos, cmap="RdYlGn", center=0.5, vmin=0, vmax=1,
                         annot=False, linewidths=0.5, linecolor='lightgray',
                         cbar_kws={'label': 'Voto (0 Contra, 1 A favor)'})

        plt.title(f"{titulo} | FOTO FINAL | Satisfacción: {tasa_satisfaccion:.1f}%")

        sumas_por_ley = matriz_votos.sum(axis=0)

        etiquetas_x = []
        for i, suma in enumerate(sumas_por_ley):
            etiquetas_x.append(f"Ley {i}\n({suma} v)")

        ax.set_xticks(np.arange(len(sumas_por_ley)) + 0.5)
        ax.set_xticklabels(etiquetas_x, rotation=0)

        plt.xlabel("Variables (Leyes) y Resultado Final")
        plt.ylabel("Agentes (0 al 39)")

        plt.tight_layout()
        print(f"   > Abriendo gráfica FINAL para: {titulo}...")
        plt.show()
    except Exception as e:
        print(f"Error al generar gráfica: {e}")

def ejecutar_partida(env_raw, model, caso_datos=None):
    options = {"problema_inyectado": caso_datos} if caso_datos else None
    obs_dict, _ = env_raw.reset(options=options)

    acciones_dict = {}
    votos_reales = []

    for agent in env_raw.agents:
        obs_agente = obs_dict[agent]
        action, _ = model.predict(obs_agente, deterministic=True)
        acciones_dict[agent] = action

        votos_reales.append(action)

    obs_dict, rewards, terms, truncs, infos = env_raw.step(acciones_dict)

    mayoria_absoluta = env_raw.num_agentes / 2
    resultado_final_leyes = (env_raw.estado_votacion > mayoria_absoluta).astype(int)

    clausulas_satisfechas = 0
    for agent in env_raw.possible_agents:
        clausula = env_raw.clausulas_privadas[agent]
        for var_idx, deseo_agente in clausula:
            if resultado_final_leyes[var_idx] == deseo_agente:
                clausulas_satisfechas += 1
                break

    tasa_satisfaccion = (clausulas_satisfechas / env_raw.num_agentes) * 100

    matriz_votos = np.array(votos_reales)

    return matriz_votos, tasa_satisfaccion


def evaluar():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    nombre_archivo = "ppo_3sat_final_egoistaFuncionaBien.zip"
    ruta_modelo = os.path.join(BASE_DIR, "modelos", nombre_archivo)

    print(f"Buscando cerebro en: {ruta_modelo}")

    if not os.path.exists(ruta_modelo + ".zip") and not os.path.exists(ruta_modelo):
        print(f"ERROR: No encuentro el archivo.")
        return

    env_raw = Entorno3SAT(num_agentes=40, num_variables=10)
    model = PPO.load(ruta_modelo)
    print("Modelo cargado. Vamos a examinarlo de verdad.")

    print("\n--- FASE 1: VISUALIZACIÓN ---")
    casos = generar_casos_laboratorio(10, 40)

    for nombre_caso, datos_caso in casos.items():
        matriz_votos, tasa = ejecutar_partida(env_raw, model, datos_caso)
        graficar_evolucion(matriz_votos, nombre_caso, tasa)

    print("\n--- FASE 2: ESTADÍSTICAS GLOBALES (500 PARTIDAS ALEATORIAS) ---")
    total_partidas = 500
    suma_tasas = 0.0

    for i in range(total_partidas):
        _, tasa = ejecutar_partida(env_raw, model)
        suma_tasas += tasa
        if i % 10 == 0: print(".", end="", flush=True)

    tasa_media = suma_tasas / total_partidas

    print(f"\n\nRESULTADOS DEL MODELO ACTUAL:")
    print(f"---------------------------------------")
    print(f"Satisfacción Media:  {tasa_media:.2f}% de cláusulas")
    print(f"---------------------------------------")

if __name__ == "__main__":
    evaluar()
