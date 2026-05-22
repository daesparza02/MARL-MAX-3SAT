import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from mi_entorno_3sat_v13d_esperanzas_simple import Entorno3SAT


def entrenar():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    MODEL_DIR = os.path.join(BASE_DIR, "modelos")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    NUM_AGENTES = 40
    NUM_VARIABLES = 10
    TOTAL_TIMESTEPS = 10_000_000

    print("=" * 60)
    print("  ENTRENAMIENTO PPO - V13d (Esperanzas, V1 con accion ternaria)")
    print("=" * 60)
    print(f"  Agentes: {NUM_AGENTES}")
    print(f"  Variables: {NUM_VARIABLES}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Algoritmo: IPPO (PPO con parameter sharing)")
    print(f"  Observacion: DNI + mapa de clausula (identica a V1)")
    print(f"  Accion: MultiDiscrete([3]*10) - stay / drop / jump por variable")
    print(f"  Recompensa: 100 si clausula propia satisfecha, 0 si no (identica a V1)")
    print(f"  Pasos por episodio: 1 (identica a V1)")
    print("=" * 60)

    env = Entorno3SAT(num_agentes=NUM_AGENTES, num_variables=NUM_VARIABLES)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        batch_size=2048,
        n_steps=2048,
        gamma=0.99,
        ent_coef=0.05,
    )

    checkpoint = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_3sat_v13d_esperanzas_simple"
    )

    print("\nEntrenando...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint)

    nombre_final = "ppo_3sat_v13d_esperanzas_simple_final"
    ruta_final = os.path.join(MODEL_DIR, nombre_final)
    model.save(ruta_final)

    print("\n" + "=" * 60)
    print(f"  TERMINADO. Modelo guardado en: {ruta_final}.zip")
    print("=" * 60)


if __name__ == "__main__":
    entrenar()
