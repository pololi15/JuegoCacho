"""
Torneo IA vs IA para el Cacho (sin interfaz gráfica).

Reglas usadas (consigna, sección 8.1): 5 dados, hasta 3 tiradas por turno (una tirada
inicial + 2 relanzamientos), 10 categorías, 10 rondas por jugador, 2 jugadores.

Cada agente recibe una vista del estado en la que él es el jugador 0 (así el agente
base del docente, que asume ser MAX = jugador 0, juega correctamente desde cualquier
asiento). Quién empieza se alterna en cada partida y las semillas quedan registradas.

Uso:
    python torneo_cacho.py --a Base --b Mejorada --n 500 --semilla 2026
    python torneo_cacho.py --a Base --b Mejorada --n 500 --csv ../../datos/resultados_cacho.csv
"""

import argparse
import csv
import os
import random
import time

from AgenteCacho import CATEGORIAS, normalizar, puntuar
from AgenteCachoIA import AgenteCachoIA
from AgenteIA.AgenteJugador import ElEstado
from agente_ia import AgenteCachoBaseCorregido, AgenteCachoMejorado
from funcion_evaluacion import FuncionEvaluacionMejorada, evaluacion_base

TIRADAS_EXTRA = 2          # relanzamientos tras la tirada inicial (3 tiradas en total)
N_RONDAS = 10

# Pesos finales elegidos por calibración + validación (ver calibrar_cacho.py,
# datos/calibracion_cacho.csv y datos/validacion_cacho.csv)
PESOS_MEJORADA = dict(rho_ini=1.0, rho_fin=1.0, kappa=0.0)
PESOS_ETAPAS = dict(rho_ini=0.8, rho_fin=0.4, kappa=0.0)   # hipótesis por etapas descartada


class AgenteCachoIAContado(AgenteCachoIA):
    """Agente base del docente sin cambios de lógica; solo cuenta llamadas a la evaluación."""

    def __init__(self, *a, **k):
        AgenteCachoIA.__init__(self, *a, **k)
        self.evaluaciones = 0

    def funcion_evaluacion(self, estado):
        self.evaluaciones += 1
        return AgenteCachoIA.funcion_evaluacion(self, estado)


def crear_agente(nombre):
    if nombre == 'Base':
        return AgenteCachoIAContado(nombre='Base', altura=1)
    if nombre == 'BaseCorregida':
        return AgenteCachoBaseCorregido()
    if nombre == 'Mejorada':
        return AgenteCachoMejorado('Mejorada', FuncionEvaluacionMejorada(**PESOS_MEJORADA))
    if nombre == 'MejoradaH1':      # misma función, búsqueda miope (altura 1)
        return AgenteCachoMejorado('MejoradaH1', FuncionEvaluacionMejorada(**PESOS_MEJORADA), horizonte=1)
    if nombre == 'MejoradaEtapas':   # pesos adaptativos por etapa (0.8 -> 0.4)
        return AgenteCachoMejorado('MejoradaEtapas', FuncionEvaluacionMejorada(**PESOS_ETAPAS))
    if nombre == 'MejoradaEVEstatico':  # misma búsqueda, pero EV estáticos de la base
        return AgenteCachoMejorado('MejoradaEVEstatico',
                                   FuncionEvaluacionMejorada(**PESOS_MEJORADA, valores='estaticos'))
    if nombre == 'MejoradaRiesgo':   # con término de varianza dependiente del marcador
        p = dict(PESOS_MEJORADA, kappa=0.5)
        return AgenteCachoMejorado('MejoradaRiesgo', FuncionEvaluacionMejorada(**p))
    raise ValueError(nombre)


def vista(tablero, yo):
    """Estado visto por el jugador 'yo' (siempre como jugador 0)."""
    o = 1 - yo
    t = {
        'dados': tablero['dados'],
        'tiradas_restantes': tablero['tiradas_restantes'],
        'puntajes': [tablero['puntajes'][yo], tablero['puntajes'][o]],
        'usadas': [set(tablero['usadas'][yo]), set(tablero['usadas'][o])],
        'ronda': len(tablero['usadas'][yo]),
    }
    return ElEstado(jugador=0, get_utilidad=0, tablero=t, movidas=None)


def _accion_valida(accion, tablero, yo):
    if not (isinstance(accion, tuple) and len(accion) == 2):
        return False
    if accion[0] == 'tirar':
        return tablero['tiradas_restantes'] > 0 and len(accion[1]) > 0
    return accion[0] == 'anotar' and accion[1] in CATEGORIAS and accion[1] not in tablero['usadas'][yo]


def jugar_partida(agentes, semilla, empieza=0, registro=None):
    """
    agentes: [agente_0, agente_1]. Devuelve dict con puntajes y métricas.
    registro: lista opcional donde se guardan (ronda, jugador, E_base, E_nueva) al final de cada ronda.
    """
    rng = random.Random(semilla)
    tab = {'dados': None, 'tiradas_restantes': 0, 'puntajes': [0, 0],
           'usadas': [set(), set()]}
    tiempos = [[], []]
    evals = [[], []]
    jugador = empieza
    fe_ref = FuncionEvaluacionMejorada(**PESOS_MEJORADA)
    for _ in range(2 * N_RONDAS):
        tab['dados'] = normalizar([rng.randint(1, 6) for _ in range(5)])
        tab['tiradas_restantes'] = TIRADAS_EXTRA
        while True:
            ag = agentes[jugador]
            ag.estado = vista(tab, jugador)
            if hasattr(ag, 'evaluaciones'):
                ag.evaluaciones = 0
            t0 = time.perf_counter()
            accion = ag.elegir_accion(ag.estado)
            tiempos[jugador].append(time.perf_counter() - t0)
            evals[jugador].append(getattr(ag, 'evaluaciones', 0))
            if not _accion_valida(accion, tab, jugador):
                disp = [c for c in CATEGORIAS if c not in tab['usadas'][jugador]]
                accion = ('anotar', max(disp, key=lambda c: puntuar(c, tab['dados'])))
            if accion[0] == 'tirar':
                d = list(tab['dados'])
                for i in accion[1]:
                    d[i] = rng.randint(1, 6)
                tab['dados'] = normalizar(d)
                tab['tiradas_restantes'] -= 1
                continue
            cat = accion[1]
            tab['puntajes'][jugador] += puntuar(cat, tab['dados'])
            tab['usadas'][jugador].add(cat)
            break
        if registro is not None and jugador != empieza:   # fin de una ronda completa
            r = len(tab['usadas'][0])
            registro.append((r,
                             evaluacion_base(tab['puntajes'], tab['usadas'], 0),
                             fe_ref.evaluar(tab['puntajes'], tab['usadas'], 0, r)))
        jugador = 1 - jugador
    return {
        'puntajes': list(tab['puntajes']),
        'tiempo': [sum(t) / len(t) for t in tiempos],
        'evals': [sum(e) / len(e) for e in evals],
    }


def torneo(nombre_a, nombre_b, n, semilla=2026, csv_path=None, verbose=True):
    filas = []
    a, b = crear_agente(nombre_a), crear_agente(nombre_b)
    for k in range(n):
        s = semilla + k
        empieza = k % 2                 # alterna quién abre la partida
        res = jugar_partida([a, b], s, empieza)
        pa, pb = res['puntajes']
        filas.append({
            'agente_a': nombre_a, 'agente_b': nombre_b,
            'puntaje_a': pa, 'puntaje_b': pb,
            'version_funcion': nombre_b.lower(),
            'tiempo_decision_promedio': round((res['tiempo'][0] + res['tiempo'][1]) / 2, 6),
            'tiempo_a': round(res['tiempo'][0], 6), 'tiempo_b': round(res['tiempo'][1], 6),
            'evals_a': round(res['evals'][0], 1), 'evals_b': round(res['evals'][1], 1),
            'ganador': nombre_a if pa > pb else nombre_b if pb > pa else 'empate',
            'empieza': nombre_a if empieza == 0 else nombre_b,
            'semilla': s,
        })
        if verbose and (k + 1) % 50 == 0:
            ga = sum(f['ganador'] == nombre_a for f in filas)
            gb = sum(f['ganador'] == nombre_b for f in filas)
            print(f"  {k + 1}/{n}: {nombre_a} {ga} - {gb} {nombre_b}")
    if csv_path:
        nuevo = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            if nuevo:
                w.writeheader()
            w.writerows(filas)
    return filas


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='Base')
    ap.add_argument('--b', default='Mejorada')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--semilla', type=int, default=2026)
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()
    torneo(args.a, args.b, args.n, args.semilla, args.csv)
