import os
import torch
import torch.nn as nn
import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from mi_entorno_3sat_v9_secuencial_egoista import Entorno3SAT


class PassthroughExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space):
        super().__init__(observation_space, features_dim=observation_space.shape[0])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class SplitMlpExtractor(nn.Module):
    def __init__(self, feature_dim, local_dim=34, global_dim=44, hidden_size=128):
        super().__init__()
        self.local_dim = local_dim
        self.latent_dim_pi = hidden_size
        self.latent_dim_vf = hidden_size

        self.policy_net = nn.Sequential(
            nn.Linear(local_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.value_net = nn.Sequential(
            nn.Linear(global_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features[:, :self.local_dim])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features[:, self.local_dim:])


class MAPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = PassthroughExtractor
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitMlpExtractor(
            feature_dim=self.features_dim,
            local_dim=Entorno3SAT.LOCAL_OBS_DIM,
            global_dim=Entorno3SAT.GLOBAL_STATE_DIM,
        )


def entrenar():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    MODEL_DIR = os.path.join(BASE_DIR, "modelos")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    NUM_AGENTES = 40
    NUM_VARIABLES = 10
    TOTAL_TIMESTEPS = 20_000_000

    print("=" * 60)
    print("  ENTRENAMIENTO MAPPO - V9 (Secuencial Egoista)")
    print("=" * 60)
    print(f"  Agentes: {NUM_AGENTES}")
    print(f"  Variables: {NUM_VARIABLES}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Actor ve: {Entorno3SAT.LOCAL_OBS_DIM} dims (info privada)")
    print(f"  Critico ve: {Entorno3SAT.GLOBAL_STATE_DIM} dims (estado global)")
    print("=" * 60)

    env = Entorno3SAT(num_agentes=NUM_AGENTES, num_variables=NUM_VARIABLES)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    model = PPO(
        MAPPOPolicy,
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        batch_size=2048,
        n_steps=4096,
        gamma=0.95,
        ent_coef=0.01,
    )

    checkpoint = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="mappo_3sat_v9_egoista"
    )

    print("\nEntrenando...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint)

    nombre_final = "mappo_3sat_v9_secuencial_egoista_final"
    ruta_final = os.path.join(MODEL_DIR, nombre_final)
    model.save(ruta_final)

    print("\n" + "=" * 60)
    print(f"  TERMINADO. Modelo guardado en: {ruta_final}.zip")
    print("=" * 60)


if __name__ == "__main__":
    entrenar()
