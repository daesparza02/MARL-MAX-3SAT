import os
import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from mi_entorno_3sat_v6_egoistaInformado import Entorno3SAT

def entrenar():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    MODEL_DIR = os.path.join(BASE_DIR, "modelos")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    NUM_AGENTES = 40
    NUM_VARIABLES = 10

    TOTAL_TIMESTEPS = 10_000_000

    print(f"--- ENTRENAMIENTO BLINDADO (Sin paradas) ---")
    print(f"   > Objetivo: {TOTAL_TIMESTEPS} pasos.")
    print(f"   > Guardando en: {MODEL_DIR}")

    env = Entorno3SAT(num_agentes=NUM_AGENTES, num_variables=NUM_VARIABLES)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        batch_size=2048,
        n_steps=2048,
        gamma=0.99,
        ent_coef=0.05
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_3sat_v6_egoistaInformado"
    )

    print("Entrenando... (Volveré dentro de unas horas)")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    nombre_final = "ppo_3sat_v6_egoistaInformado_final"
    ruta_final = os.path.join(MODEL_DIR, nombre_final)
    model.save(ruta_final)

    print("---------------------------------------------------------")
    print(f"TERMINADO. Modelo guardado en: {ruta_final}.zip")
    print("Ahora ejecuta evaluar.py.")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    entrenar()
