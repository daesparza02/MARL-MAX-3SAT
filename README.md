# MAX-3SAT como juego de votación multiagente

Este repositorio contiene el código fuente del Trabajo de Fin de Grado *"MAX-3SAT como juego de votación multiagente"*. El proyecto modela el problema clásico de MAX-3SAT como un sistema de decisión colectiva donde 40 agentes, cada uno con una cláusula privada de tres literales, asignan los valores de 10 variables booleanas mediante diferentes mecanismos de votación, todo ello aprendido por refuerzo (PPO / MAPPO).

A lo largo del trabajo se han desarrollado **doce versiones del entorno** que exploran distintos mecanismos de votación, funciones de recompensa y arquitecturas de aprendizaje. Los resultados, análisis y conclusiones de cada versión están detallados en la memoria del TFG.

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

---

## 📂 `modelos definitivos/` — Modelos entrenados

Contiene los `.zip` finales de cada versión, listos para ser cargados con `PPO.load()`. Son los modelos cuyos resultados aparecen en las tablas comparativas de la memoria.

```
v1.zip, v2.zip, v3.zip, v5.zip, v6.zip, v7.zip, v8.zip
v9Comunitario.zip, v9Egoista.zip
v10.zip, v11.zip, v12.zip
```

> Nota: V4 (en sus dos variantes) no tiene modelo definitivo guardado porque produjo resultados degenerados — los agentes se comportaban como aleatorios. La memoria documenta esa versión como un fracaso instructivo, por lo que reproducir el entrenamiento no aporta valor.

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

1. **Para entender el trabajo:** leer la memoria + abrir `entorno/v1.py`, `entorno/v8.py`, `entorno/v10.py`, `entorno/v11.py` y `entorno/v12.py`. Cada uno representa un hito conceptual del trabajo.
2. **Para reproducir un resultado concreto:** cargar el `.zip` correspondiente de `modelos definitivos/` con el evaluador adecuado de `evaluacion/`.
3. **Para reentrenar desde cero:** ejecutar el script de `entrenamiento/` correspondiente. Los modelos del trabajo se entrenaron entre 5M y 100M de timesteps; el tiempo de ejecución va de ~30 min (V1) a varias horas (V12).
