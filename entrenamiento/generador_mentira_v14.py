import random
import numpy as np


# =====================================================================
# Simulador autonomo de la mecanica del arbitro 7/8 (identica a V12).
# Se replica aqui para que el generador sea independiente del entorno
# (evita imports circulares y permite probar este modulo por separado).
# =====================================================================

EXPECTATION_LEVELS = [0.0, 0.5, 0.75, 0.875, 1.0]


def _compute_true_expectation_idx(clausula, var_status):
    literales_libres = 0
    ya_satisfecha = False
    for var_idx, deseo in clausula:
        estado = var_status[var_idx]
        if estado == -1:
            literales_libres += 1
        elif estado == deseo:
            ya_satisfecha = True
    if ya_satisfecha:
        return 4
    if literales_libres == 0:
        return 0
    if literales_libres == 1:
        return 1
    if literales_libres == 2:
        return 2
    return 3


def _action_to_reports(accion, current_idx):
    current_val = EXPECTATION_LEVELS[current_idx]
    if current_idx == 0 or current_idx == 4:
        return current_val, current_val
    if accion == 0:          # stay
        return current_val, current_val
    elif accion == 1:        # drop -> prefiere x=1
        return EXPECTATION_LEVELS[current_idx - 1], 1.0
    elif accion == 2:        # jump -> prefiere x=0
        return 1.0, EXPECTATION_LEVELS[current_idx - 1]
    return current_val, current_val


def _truthful_action(clausula, current_idx, current_var):
    if current_idx == 0 or current_idx == 4:
        return 0
    for var_idx, deseo in clausula:
        if var_idx == current_var:
            return 1 if deseo == 1 else 2
    return 0


def _clausula_satisfecha_final(clausula, var_status):
    for var_idx, deseo in clausula:
        if var_status[var_idx] == deseo:
            return True
    return False


def simular(clausulas, num_variables, override=None):
    """
    Simula una partida secuencial completa.
    override: dict (agente, current_var) -> accion forzada (0/1/2). El resto
    de agentes (y el resto de pasos) juegan truthful.
    Devuelve var_status final (lista de 0/1).
    """
    agentes = list(clausulas.keys())
    var_status = [-1] * num_variables
    expectation_idx = {
        a: _compute_true_expectation_idx(clausulas[a], var_status) for a in agentes
    }
    niveles = np.array(EXPECTATION_LEVELS, dtype=np.float32)

    for current_var in range(num_variables):
        reports = {}
        sum_E0 = 0.0
        sum_E1 = 0.0
        for a in agentes:
            act = _truthful_action(clausulas[a], expectation_idx[a], current_var)
            if override is not None and (a, current_var) in override:
                act = override[(a, current_var)]
            E0, E1 = _action_to_reports(act, expectation_idx[a])
            reports[a] = (E0, E1)
            sum_E0 += E0
            sum_E1 += E1

        chosen = 1 if sum_E1 >= sum_E0 else 0
        var_status[current_var] = chosen

        for a in agentes:
            E0, E1 = reports[a]
            chosen_E = E1 if chosen == 1 else E0
            expectation_idx[a] = int(np.argmin(np.abs(niveles - chosen_E)))

    return var_status


# =====================================================================
# Verificador: una instancia es "mentira-optima" para el protagonista P si
#   - jugando todos truthful, P queda INSATISFECHO, y
#   - si P apoya la variable de apoyo (mentira), P queda SATISFECHO.
# =====================================================================

def verificar_mentira_optima(clausulas, meta, num_variables=10):
    P = meta["protagonista"]
    B = meta["var_apoyo"]
    v = meta["valor_apoyo"]
    accion_apoyo = 1 if v == 1 else 2  # drop si v=1, jump si v=0

    vs_truth = simular(clausulas, num_variables, override=None)
    p_sat_truth = _clausula_satisfecha_final(clausulas[P], vs_truth)

    vs_lie = simular(clausulas, num_variables, override={(P, B): accion_apoyo})
    p_sat_lie = _clausula_satisfecha_final(clausulas[P], vs_lie)

    return (not p_sat_truth) and p_sat_lie


# =====================================================================
# Generador de candidatos. Plantilla estructurada + muestreo por rechazo:
# solo se devuelven instancias que el verificador confirma.
# =====================================================================

def _add_fillers(lits, rng, fillers, n):
    usados = {var for var, _ in lits}
    disponibles = [f for f in fillers if f not in usados]
    rng.shuffle(disponibles)
    for k in range(n):
        var = disponibles[k]
        deseo = rng.randint(0, 1)
        lits.append((var, deseo))
    return lits


def _construir_candidato(rng, num_agentes, num_variables):
    A = num_variables - 1            # variable propia de P (se decide la ultima)
    B = 0                            # variable de apoyo (se decide la primera)
    fillers = list(range(1, num_variables - 1))  # variables 1..n-2

    tam_G = rng.randint(8, 12)       # grupo opositor (quieren A=0, satisfacibles por B=1)
    tam_S = rng.randint(3, tam_G - 2)  # soportadores de A=1 (|S| <= |G|-2)
    tam_cB = tam_G + 1               # contra-B: hacen a P pivotal sobre B

    total_roles = 1 + tam_G + tam_S + tam_cB
    if total_roles > num_agentes:
        return None, None
    tam_relleno = num_agentes - total_roles

    indices = list(range(num_agentes))
    rng.shuffle(indices)
    it = iter(indices)

    clausulas = {}

    # Protagonista: necesita A=1
    P = f"agente_{next(it)}"
    clausulas[P] = _add_fillers([(A, 1)], rng, fillers, 2)

    # Grupo opositor: A=0 (contra P) pero satisfacibles por B=1
    for _ in range(tam_G):
        ag = f"agente_{next(it)}"
        clausulas[ag] = _add_fillers([(A, 0), (B, 1)], rng, fillers, 1)

    # Soportadores: A=1 como P
    for _ in range(tam_S):
        ag = f"agente_{next(it)}"
        clausulas[ag] = _add_fillers([(A, 1)], rng, fillers, 2)

    # Contra-B: quieren B=0 (cantidad = |G|+1 -> P es el voto que desempata)
    for _ in range(tam_cB):
        ag = f"agente_{next(it)}"
        clausulas[ag] = _add_fillers([(B, 0)], rng, fillers, 2)

    # Relleno: clausulas solo sobre fillers (no tocan A ni B)
    for _ in range(tam_relleno):
        ag = f"agente_{next(it)}"
        clausulas[ag] = _add_fillers([], rng, fillers, 3)

    meta = {"protagonista": P, "var_apoyo": B, "valor_apoyo": 1, "var_propia": A}
    return clausulas, meta


def generar_instancia_mentira_optima(num_agentes=40, num_variables=10,
                                     seed=None, max_intentos=5000):
    """
    Devuelve (clausulas, meta) de una instancia donde mentir es estrictamente
    optimo para el protagonista. Lanza RuntimeError si no lo consigue.
    """
    rng = random.Random(seed)
    for _ in range(max_intentos):
        clausulas, meta = _construir_candidato(rng, num_agentes, num_variables)
        if clausulas is None:
            continue
        if verificar_mentira_optima(clausulas, meta, num_variables):
            return clausulas, meta
    raise RuntimeError("No se pudo generar una instancia mentira-optima.")


def generar_pool(n, num_agentes=40, num_variables=10, seed=None, max_intentos=5000):
    """Genera una lista de n pares (clausulas, meta) validos."""
    rng = random.Random(seed)
    pool = []
    while len(pool) < n:
        s = rng.randint(0, 2**31 - 1)
        try:
            inst, meta = generar_instancia_mentira_optima(
                num_agentes, num_variables, seed=s, max_intentos=max_intentos)
            pool.append((inst, meta))
        except RuntimeError:
            continue
    return pool


# =====================================================================
# Test de sanidad: ejecutar este modulo directamente.
# =====================================================================
if __name__ == "__main__":
    N = 200
    print(f"Generando y verificando {N} instancias mentira-optimas...")
    ok = 0
    tam_pool = generar_pool(N, seed=0)
    for clausulas, meta in tam_pool:
        # Doble comprobacion explicita
        if verificar_mentira_optima(clausulas, meta):
            ok += 1
    print(f"  Verificadas correctas: {ok}/{N}")

    # Comprobacion detallada de una instancia
    inst, meta = tam_pool[0]
    P, B, v, A = meta["protagonista"], meta["var_apoyo"], meta["valor_apoyo"], meta["var_propia"]
    vs_truth = simular(inst, 10)
    vs_lie = simular(inst, 10, override={(P, B): (1 if v == 1 else 2)})
    print("\nEjemplo:")
    print(f"  Protagonista: {P} | apoya x{B}={v} | necesita x{A}=1")
    print(f"  Juego veraz   -> x{A}={vs_truth[A]}  (P satisfecho: {_clausula_satisfecha_final(inst[P], vs_truth)})")
    print(f"  Juego mentira -> x{A}={vs_lie[A]}  (P satisfecho: {_clausula_satisfecha_final(inst[P], vs_lie)})")
