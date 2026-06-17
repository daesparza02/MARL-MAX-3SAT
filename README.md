# MAX-3SAT como juego de votación multiagente

Este repositorio contiene el código fuente del Trabajo de Fin de Grado *"MAX-3SAT como juego de votación multiagente"*. El proyecto modela el problema clásico de MAX-3SAT como un sistema de decisión colectiva donde 40 agentes, cada uno con una cláusula privada de tres literales, asignan los valores de 10 variables booleanas mediante diferentes mecanismos de votación, todo ello aprendido por refuerzo (PPO / MAPPO).

A lo largo del trabajo se han desarrollado **trece versiones del entorno** que exploran distintos mecanismos de votación, funciones de recompensa y arquitecturas de aprendizaje. Los resultados, análisis y conclusiones de cada versión están detallados en la memoria del TFG. La última versión (V14) cierra el arco del trabajo estudiando la pregunta inversa a V12: si mentir es la estrategia óptima, ¿emerge también la deshonestidad?

---

## Estructura del repositorio

```
.
├── entorno/                                    ←  LÓGICA V1–V12 (núcleo)
├── entrenamiento/                              ←  scripts auxiliares de entrenamiento (V1–V12)
├── evaluacion/                                 ←  scripts auxiliares de evaluación (V1–V12)
├── modelos definitivos/                        ←  modelos ya entrenados V1–V12 (.zip)
│
│   ── V14: experimento avanzado (archivos en la raíz) ──
├── mi_entorno_3sat_v14_mentiraEstrategica.py   ←  entorno V14
├── generador_mentira_v14.py                    ←  generador de instancias (dependencia crítica de V14)
├── entrenar_mappo_v14.py                       ←  entrenamiento V14
├── evaluar_v14.py                              ←  evaluación V14
└── modelos/mappo_3sat_v14_mentira_final.zip    ←  modelo V14 entrenado
```

> **Importante para el corrector:** la lógica conceptual del trabajo — el modelado del problema MAX-3SAT como juego multiagente, los espacios de observación y acción, las dinámicas de votación, las funciones de recompensa y los distintos mecanismos explorados — vive **íntegramente en la carpeta `entorno/`** (V1–V12) y en los cuatro archivos raíz de V14. Las otras carpetas son herramientas auxiliares para entrenar, evaluar o cargar modelos, pero no contienen aportación conceptual del trabajo.

---

## `entorno/` — Núcleo del trabajo

Cada archivo de esta carpeta define una clase `Entorno3SAT` que hereda de `pettingzoo.ParallelEnv` y modela una versión concreta del juego de votación. Aquí está concentrado todo el diseño experimental: la dinámica temporal, los espacios de acción/observación, las reglas de mayoría, la función de recompensa y, en V11 y V12, el mecanismo del árbitro inspirado en el algoritmo de 7/8-aproximación.

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

Para entender realmente qué hace cada modelo, **estos son los archivos que hay que leer**.

> **V14** no sigue la estructura de subcarpetas. Sus archivos están en la raíz del repositorio; ver la sección [V14 — Mentira estratégica](#v14--mentira-estratégica) más abajo.

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

Todos los scripts guardan checkpoints periódicos en `modelos/` y registran métricas para TensorBoard en `logs/`.

> El script de entrenamiento de V14 (`entrenar_mappo_v14.py`) está en la raíz del repositorio, no aquí.

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

> El evaluador de V14 (`evaluar_v14.py`) está en la raíz del repositorio, no aquí.

---

##  `modelos definitivos/` — Modelos entrenados

Contiene los `.zip` finales de cada versión, listos para ser cargados con `PPO.load()`. Son los modelos cuyos resultados aparecen en las tablas comparativas de la memoria.

```
v1.zip, v2.zip, v3.zip, v5.zip, v6.zip, v7.zip, v8.zip
v9Comunitario.zip, v9Egoista.zip
v10.zip, v11.zip, v12.zip
```

> Nota: V4 (en sus dos variantes) no tiene modelo definitivo guardado porque produjo resultados degenerados — los agentes se comportaban como aleatorios. La memoria documenta esa versión como un fracaso instructivo, por lo que reproducir el entrenamiento no aporta valor.

> El modelo de V14 (`modelos/mappo_3sat_v14_mentira_final.zip`) está en la carpeta `modelos/` de la raíz, no aquí.

---

## V14 — Mentira estratégica

V14 plantea la pregunta inversa a V12: **si la deshonestidad es la estrategia óptima, ¿emerge también la mentira?**

El mecanismo del árbitro (7/8-aproximación) sigue intacto, pero V14 introduce instancias *ad hoc* donde un agente —el protagonista P— puede mejorar su resultado mintiendo: apoya una variable ajena para satisfacer a un grupo de opositores, neutralizándolos antes de que puedan votar contra la variable propia de P. El agente pierde su propio turno de voto (auto-bloqueo), pero sus aliados deciden la variable por él mientras sus enemigos quedan fuera de juego.

A diferencia de V1–V12, todos los archivos de V14 están en la **raíz del repositorio**.

### Archivos de V14

| Archivo | Rol |
|---|---|
| `mi_entorno_3sat_v14_mentiraEstrategica.py` | Entorno V14. Extiende V12 añadiendo presión dinámica (10 dims) y matriz de acoplamiento (20 dims) a la observación del actor (`LOCAL_OBS_DIM=52`, `GLOBAL_STATE_DIM=42`). |
| `generador_mentira_v14.py` | **Dependencia imprescindible.** Generador y verificador de instancias *mentira-óptimas*; lo importan tanto el entorno (inyección durante el entrenamiento) como el evaluador (Fase B). Sin este archivo no arranca nada. |
| `entrenar_mappo_v14.py` | Entrenamiento MAPPO. `p_inyeccion=0.4` (40 % de episodios con instancias construidas donde mentir es óptimo), 110M pasos totales. Contiene también las clases `MAPPOPolicy` y `SplitMlpExtractor` que importa el evaluador. |
| `evaluar_v14.py` | Evaluador en dos fases: Fase A (500 partidas aleatorias → satisfacción + veracidad) y Fase B (300 instancias *ad hoc* held-out → tasa de mentira estratégica y éxito del protagonista). |
| `modelos/mappo_3sat_v14_mentira_final.zip` | Modelo final entrenado (110M pasos, ~590 KB). |

Los cinco archivos son necesarios para ejecutar la evaluación. La cadena de imports es:

```
evaluar_v14.py
  ├── mi_entorno_3sat_v14_mentiraEstrategica.py
  │     └── generador_mentira_v14.py
  ├── entrenar_mappo_v14.py   (clases MAPPOPolicy / SplitMlpExtractor)
  └── generador_mentira_v14.py
```

### Observación ampliada respecto a V12

El actor recibe 52 dimensiones en lugar de 22:

| Bloque | Dims | Contenido |
|---|---|---|
| Mapa de cláusula | 10 | Deseos del agente por variable |
| Estado de variables | 10 | −1 (libre) / 0 / 1 (decidida) |
| Esperanza propia | 1 | Nivel de expectativa actual |
| Paso normalizado | 1 | Fracción del episodio transcurrida |
| **Presión dinámica neta** | **10** | (favor − contra) / N por variable, excluyendo agentes ya bloqueados |
| **Matriz de acoplamiento** | **20** | Para cada (variable, valor): fracción de opositores vivos que quedarían neutralizados si el agente apoya ese valor |

La **presión dinámica** se recalcula en cada paso excluyendo agentes ya satisfechos. La **matriz de acoplamiento** es la clave de V14: le indica al agente exactamente cuántos enemigos podría neutralizar apoyando cada variable ajena, haciendo que la mentira estratégica sea una acción identificable y aprendible.

### Resultados

- **Fase A** (instancias aleatorias): ~96,9 % de cláusulas satisfechas, ~96,9 % de veracidad. El modelo no degrada el comportamiento honesto en los contextos donde es lo óptimo.
- **Fase B** (instancias *ad hoc* held-out): ~99,7 % de tasa de mentira estratégica. El modelo aprende a detectar y ejecutar la mentira cuando es rentable.

### Cómo usar V14

Desde la raíz del repositorio (los cinco archivos deben estar en la misma carpeta):

```bash
python evaluar_v14.py          # evalúa; carga modelos/mappo_3sat_v14_mentira_final.zip
python entrenar_mappo_v14.py   # reentrenar desde cero
```

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

### 2. Flujo de entrenamiento

Cada script de `entrenamiento/` produce un modelo entrenado siguiendo siempre el mismo patrón:

1. Carga el entorno correspondiente desde `entorno/`.
2. Convierte el `ParallelEnv` de PettingZoo a un `VecEnv` compatible con SB3 mediante SuperSuit.
3. Instancia un `PPO` (con `MlpPolicy` para IPPO o `MAPPOPolicy` para MAPPO).
4. Entrena durante un número fijo de pasos guardando checkpoints cada 100.000.
5. Guarda el modelo final en la carpeta `modelos/`.

Para entrenar una versión concreta:

```bash
cd entrenamiento/
python entrenar_mappo_V12.py        # ejemplo: entrenar V12 (80M pasos, varias horas)
```

Si quieres entrenar una versión que comparte plantilla (p. ej. V7), abre `entrenar_mappo.py` y cambia la línea del `import` para que apunte al entorno deseado, así como el `name_prefix` y el `nombre_final` para no sobrescribir otros modelos.

Durante el entrenamiento puedes monitorizar la evolución en TensorBoard:

```bash
tensorboard --logdir logs/
```

### 3. Flujo de evaluación

Cada script de `evaluacion/` carga un modelo entrenado y produce dos tipos de resultados:

- **Fase 1 — Casos personalizados:** evalúa el modelo sobre los 9 casos *ad hoc* (utopía, conflicto, escalera, pares/impares, cadena de rescate y 4 instancias densas). Genera gráficas con la votación de cada agente y, en versiones multipaso, la evolución turno a turno.
- **Fase 2 — Estadísticas globales:** ejecuta 500 partidas aleatorias y calcula la tasa media de satisfacción (y de veracidad en V11/V12).

Para evaluar una versión:

```bash
cd evaluacion/
python evaluar_V12.py        # ejemplo: evaluar V12
```

Antes de ejecutar, asegúrate de que la ruta del modelo dentro del script coincide con el `.zip` correspondiente en `modelos/`. Por defecto los scripts buscan el modelo en `modelos/` por el nombre con que lo guardó el script de entrenamiento; si quieres usar los modelos finales de `modelos definitivos/`, copia el `.zip` a `modelos/` con el nombre esperado o ajusta la variable `nombre_archivo` dentro del evaluador.

---

## Resumen rápido para correctores

1. **Para entender el trabajo:** leer la memoria + abrir `entorno/v1.py`, `entorno/v8.py`, `entorno/v10.py`, `entorno/v11.py`, `entorno/v12.py` y `mi_entorno_3sat_v14_mentiraEstrategica.py`. Cada uno representa un hito conceptual del trabajo.
2. **Para reproducir un resultado concreto (V1–V12):** cargar el `.zip` correspondiente de `modelos definitivos/` con el evaluador adecuado de `evaluacion/`.
3. **Para reproducir V14:** ejecutar `python evaluar_v14.py` desde la raíz (carga `modelos/mappo_3sat_v14_mentira_final.zip` automáticamente). Los cinco archivos de V14 deben estar en la misma carpeta.
4. **Para reentrenar desde cero:** ejecutar el script de `entrenamiento/` correspondiente (V1–V12) o `python entrenar_mappo_v14.py` (V14). Los modelos del trabajo se entrenaron entre 5M y 110M de timesteps; el tiempo de ejecución va de ~30 min (V1) a varias horas (V12/V14).
