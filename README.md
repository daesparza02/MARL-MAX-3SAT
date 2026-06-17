# MAX-3SAT como juego de votación multiagente

Este repositorio contiene el código fuente del Trabajo de Fin de Grado *"MAX-3SAT como juego de votación multiagente"*. El proyecto modela el problema clásico de MAX-3SAT como un sistema de decisión colectiva donde 40 agentes, cada uno con una cláusula privada de tres literales, asignan los valores de 10 variables booleanas mediante diferentes mecanismos de votación, todo ello aprendido por refuerzo (PPO / MAPPO).

A lo largo del trabajo se han desarrollado **trece versiones del entorno** que exploran distintos mecanismos de votación, funciones de recompensa y arquitecturas de aprendizaje. Los resultados, análisis y conclusiones de cada versión están detallados en la memoria del TFG. La última (V14) cierra el arco del trabajo con la pregunta inversa a V12: si mentir fuera la estrategia óptima, ¿emergería también la deshonestidad?

---

## Estructura del repositorio

```
.
├── entorno/                  ←  LÓGICA DEL TRABAJO (núcleo)
├── entrenamiento/            ←  scripts auxiliares de entrenamiento
├── evaluacion/               ←  scripts auxiliares de evaluación
└── modelos definitivos/      ←  modelos ya entrenados (.zip)
```

> **Importante para el corrector:** la lógica conceptual del trabajo — el modelado del problema MAX-3SAT como juego multiagente, los espacios de observación y acción, las dinámicas de votación, las funciones de recompensa y los distintos mecanismos explorados — vive **íntegramente en la carpeta `entorno/`**. Las otras tres carpetas son herramientas auxiliares para entrenar, evaluar o cargar modelos, pero no contienen aportación conceptual del trabajo.

---

## `entorno/` — Núcleo del trabajo

Cada archivo de esta carpeta define una clase `Entorno3SAT` que hereda de `pettingzoo.ParallelEnv` y modela una versión concreta del juego de votación. Aquí está concentrado todo el diseño experimental: la dinámica temporal, los espacios de acción/observación, las reglas de mayoría, la función de recompensa y, en V11, V12 y V14, el mecanismo del árbitro inspirado en el algoritmo de 7/8-aproximación.

| Archivo | Versión memoria | Descripción breve |
|---|---|---|
| `v1.py` | V1 | IPPO egoísta, observación mínima (DNI + cláusula). Punto de partida. |
| `v2.py` | V2 | IPPO comunitario con presión de demanda. Primer salto significativo. |
| `v3.py` | V3 | IPPO con recompensa mixta (10·egoísta + 0,1·comunitario). |
| `v4Ini.py` | V4 simple | IPPO multi-paso (10 pasos de negociación, voto binario). |
| `v4Avanzando.py` | V4 avanzando | V4 + abstención voluntaria + tablero persistente. |
| `v5.py` | V5 | Primera versión con MAPPO. Crítico centralizado, abstención forzada. |
| `v6.py` | V6 | IPPO egoísta con observación enriquecida (presión de demanda). |
| `v7.py` | V7 | MAPPO comunitario, un solo paso, abstención forzada. |
| `v8.py` | V8 | MAPPO comunitario con voto libre y presión informada en el actor. |
| `v9Egoista.py` | V9 egoísta | MAPPO secuencial (una variable por paso), recompensa terminal egoísta. |
| `v9Comunitario.py` | V9 comunitario | MAPPO secuencial con recompensa intermedia + bono final. |
| `v10.py` | V10 | MAPPO + voto libre + recompensa egoísta. Cooperación emergente. |
| `v11.py` | V11 | Mecanismo del árbitro 7/8 (simultáneo, IPPO, acción ternaria). |
| `v12.py` | V12 | Mecanismo del árbitro 7/8 secuencial con MAPPO. Mejor resultado del trabajo. |
| `mi_entorno_3sat_v14_mentiraEstrategica.py` | V14 | Mentira estratégica: árbitro 7/8 secuencial (como V12) ampliado con presión dinámica y matriz de acoplamiento en la observación del actor. |

Para entender realmente qué hace cada modelo, **estos son los archivos que hay que leer**.

---

## `entrenamiento/` — Scripts auxiliares de entrenamiento

Scripts que enlazan los entornos con Stable-Baselines3 (PPO/MAPPO) y lanzan el entrenamiento. La arquitectura MAPPO (`PassthroughExtractor` + `SplitMlpExtractor` + `MAPPOPolicy`) está definida aquí, ya que es la pieza que SB3 necesita para separar las observaciones del actor y del crítico.

| Archivo | Para qué versiones se usa |
|---|---|
| `entrenar.py` | Plantilla PPO genérica (V1, V2, V3, V4, V6). Cambiar el `import` del entorno según la versión deseada. |
| `entrenar_mappo.py` | Plantilla MAPPO genérica (V5, V7, V8, V10, V9 comunitario). Cambiar el `import` según versión. |
| `entrenar_mappo_V9.py` | MAPPO para V9 egoísta. |
| `entrenar_ppo_V11.py` | PPO para V11. |
| `entrenar_mappo_V12.py` | MAPPO para V12 (entrenamiento prolongado, 80M+ pasos). |
| `entrenar_mappo_v14.py` | MAPPO para V14 (mentira estratégica, ~110M pasos, con inyección de instancias mentira-óptimas). |
| `generador_mentira_v14.py` | **Dependencia de V14** (no es un script ejecutable). Genera y verifica instancias donde mentir es demostrablemente óptimo; lo usan el entorno de V14 (inyección) y su evaluador (Fase B). |

Todos los scripts guardan checkpoints periódicos en `modelos/` y registran métricas para TensorBoard en `logs/`.

---

##  `evaluacion/` — Scripts auxiliares de evaluación

Scripts que cargan un modelo entrenado y lo evalúan sobre 500 partidas aleatorias y sobre un conjunto fijo de casos *ad hoc* (utopía, conflicto, escalera, pares/impares, cadena de rescate y cuatro instancias densas). Generan tasas de satisfacción y, en V11/V12, métricas adicionales de **veracidad**.

| Archivo | Para qué versiones se usa |
|---|---|
| `evaluarPPO1step.py` | Versiones IPPO de un solo paso (V1, V2, V3, V6). |
| `evaluarPPOvariosStep.py` | V6 multipaso (con sondeos). |
| `evaluarPPOAbstentencion.py` | V4 avanzando (con abstención). |
| `evaluarMAPPO.py` | Versiones MAPPO simultáneas y secuenciales (V5, V7, V8, V9, V10). |
| `evaluar_V11.py` | V11 (árbitro 7/8 simultáneo). |
| `evaluar_V12.py` | V12 (árbitro 7/8 secuencial). |
| `evaluar_v14.py` | V14: Fase A (instancias aleatorias → satisfacción + veracidad) y Fase B (instancias mentira-óptimas *held-out* → tasa de mentira estratégica). |

---

##  `modelos definitivos/` — Modelos entrenados

Contiene los `.zip` finales de cada versión, listos para ser cargados con `PPO.load()`. Son los modelos cuyos resultados aparecen en las tablas comparativas de la memoria.

```
v1.zip, v2.zip, v3.zip, v5.zip, v6.zip, v7.zip, v8.zip
v9Comunitario.zip, v9Egoista.zip
v10.zip, v11.zip, v12.zip
mappo_3sat_v14_mentira_final.zip      (V14)
```

> Nota: V4 (en sus dos variantes) no tiene modelo definitivo guardado porque produjo resultados degenerados — los agentes se comportaban como aleatorios. La memoria documenta esa versión como un fracaso instructivo, por lo que reproducir el entrenamiento no aporta valor.

---

## V14 — Mentira estratégica

V14 plantea la pregunta inversa a V12: **si la deshonestidad fuera la estrategia óptima, ¿emergería también la mentira?**

El mecanismo del árbitro (7/8-aproximación) es el mismo de V12, pero V14 introduce instancias *ad hoc* donde un agente —el protagonista P— mejora su resultado mintiendo: apoya una variable ajena para satisfacer a un grupo de opositores y neutralizarlos antes de que puedan votar contra su propia variable. P pierde su turno de voto (auto-bloqueo), pero sus aliados deciden la variable por él mientras sus enemigos quedan fuera de juego.

**Archivos de V14** (repartidos por las carpetas, como el resto de versiones):

| Archivo | Carpeta | Rol |
|---|---|---|
| `mi_entorno_3sat_v14_mentiraEstrategica.py` | `entorno/` | Entorno (extiende V12: + presión dinámica + matriz de acoplamiento). |
| `entrenar_mappo_v14.py` | `entrenamiento/` | Entrenamiento MAPPO (`p_inyeccion=0.4`, ~110M pasos). |
| `generador_mentira_v14.py` | `entrenamiento/` | **Dependencia imprescindible** (la usan el entorno y el evaluador). |
| `evaluar_v14.py` | `evaluacion/` | Evaluador: Fase A (aleatorias) + Fase B (mentira-óptimas *held-out*). |
| `mappo_3sat_v14_mentira_final.zip` | `modelos definitivos/` | Modelo final entrenado (~590 KB). |

### Observación ampliada respecto a V12

El actor recibe 52 dimensiones en lugar de 22:

| Bloque | Dims | Contenido |
|---|---|---|
| Mapa de cláusula | 10 | Deseos del agente por variable |
| Estado de variables | 10 | −1 (libre) / 0 / 1 (decidida) |
| Esperanza propia | 1 | Nivel de expectativa actual |
| Paso normalizado | 1 | Fracción del episodio transcurrida |
| **Presión dinámica neta** | **10** | (favor − contra) / N por variable, excluyendo agentes ya bloqueados |
| **Matriz de acoplamiento** | **20** | Para cada (variable, valor): fracción de opositores vivos que se neutralizarían |

La **matriz de acoplamiento** es la clave de V14: le indica al agente cuántos enemigos podría neutralizar apoyando cada variable ajena, haciendo que la mentira estratégica sea una acción identificable y aprendible.

### Resultados

- **Fase A** (aleatorias): ~96,9 % de cláusulas satisfechas y ~96,9 % de veracidad → no degrada el comportamiento honesto donde es lo óptimo.
- **Fase B** (mentira-óptimas *held-out*, `seed=12345`): ~99,7 % de tasa de mentira estratégica → el agente aprende a mentir cuando le conviene.

---

## Cómo usar el código

### 1. Requisitos

El stack tecnológico es estándar para RL multiagente:

```
python >= 3.10
torch
gymnasium
pettingzoo
supersuit
stable-baselines3
numpy
matplotlib
seaborn
```

Instalación rápida:

```bash
pip install torch gymnasium pettingzoo supersuit stable-baselines3 numpy matplotlib seaborn
```

### 2. Montar una carpeta de trabajo

Los scripts usan imports "planos" (p. ej. `from v12 import ...`), por lo que **para ejecutar una versión hay que reunir sus archivos en una misma carpeta de trabajo** y colocar el modelo en una subcarpeta `modelos/` con el nombre que el script espera. La repartición por carpetas del repositorio es para facilitar la lectura, no la ejecución directa.

Ejemplo para **V14** (todos los imports son planos):

```
trabajo_v14/
├── mi_entorno_3sat_v14_mentiraEstrategica.py   (de entorno/)
├── entrenar_mappo_v14.py                        (de entrenamiento/)
├── generador_mentira_v14.py                     (de entrenamiento/)
├── evaluar_v14.py                               (de evaluacion/)
└── modelos/
    └── mappo_3sat_v14_mentira_final.zip         (de "modelos definitivos/")
```

Para **V1–V12** el procedimiento es el mismo: reúne el entorno (`entorno/vN.py`), su script de entrenamiento y de evaluación, y coloca el `.zip` correspondiente de `modelos definitivos/` dentro de una subcarpeta `modelos/`. Comprueba las líneas `import` de cada script por si hay que ajustar el nombre del módulo del entorno.

### 3. Flujo de entrenamiento

```bash
python entrenar_mappo_v14.py        # ejemplo: V14 (largo; el modelo ya viene incluido)
```

Cada script guarda checkpoints en `modelos/` y registra métricas; puedes monitorizarlas en TensorBoard:

```bash
tensorboard --logdir logs/
```

### 4. Flujo de evaluación

```bash
python evaluar_v14.py               # carga modelos/mappo_3sat_v14_mentira_final.zip
```

El evaluador de V14 ejecuta dos fases: **Fase A** sobre 500 instancias aleatorias (satisfacción + veracidad) y **Fase B** sobre 300 instancias mentira-óptimas *held-out* con semilla fija (exactamente reproducible). Para V1–V12, ejecuta el evaluador correspondiente asegurándote de que la ruta/nombre del modelo dentro del script (o pasado como argumento) coincide con el `.zip` colocado en `modelos/`.

---

## Resumen rápido para correctores

1. **Para entender el trabajo:** leer la memoria + abrir `entorno/v1.py`, `entorno/v8.py`, `entorno/v10.py`, `entorno/v11.py`, `entorno/v12.py` y `entorno/mi_entorno_3sat_v14_mentiraEstrategica.py`. Cada uno representa un hito conceptual del trabajo.
2. **Para reproducir un resultado:** reúne los archivos de la versión en una carpeta de trabajo (sección *Montar una carpeta de trabajo*), coloca el `.zip` de `modelos definitivos/` en `modelos/` y ejecuta el evaluador correspondiente.
3. **Para reentrenar desde cero:** ejecuta el script de `entrenamiento/` correspondiente. Los modelos del trabajo se entrenaron entre 5M y 110M de timesteps; el tiempo de ejecución va de ~30 min (V1) a varias horas (V12/V14).
