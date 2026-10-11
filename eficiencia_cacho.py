"""
Métricas de eficiencia de la selección de jugadas (antes / después de las optimizaciones).

Se toman estados de decisión reales (partidas Mejorada vs Mejorada, semillas 50_000..)
y se mide, para cada variante del buscador, el tiempo por decisión, las evaluaciones de
hoja y los nodos expandidos. Todas las variantes de la versión mejorada devuelven la
misma acción (se verifica), así que la diferencia es solo de costo.

Variantes:
    Base                 agente del docente (altura 1)
    Rec                  recursivo, sin poda ni memoización
    Rec+Poda             reservas equivalentes fusionadas (31 -> <=31 distintas)
    Rec+Memo             tabla (dados, tiradas) por decisión
    Rec+Poda+Memo
    DP (final)           programación dinámica vectorizada con caché por contexto

Uso: python eficiencia_cacho.py --n1 40 --n2 6 --csv ../../datos/eficiencia_cacho.csv
"""

import argparse
import csv
import random
import statistics
import time

from AgenteIA.AgenteJugador import ElEstado
from agente_ia import AgenteCachoMejorado
from funcion_evaluacion import FuncionEvaluacionMejorada
from torneo_cacho import PESOS_MEJORADA, crear_agente, jugar_partida


def recolectar_estados(n_partidas=10, semilla=50_000):
    estados = []
    ag = crear_agente('Mejorada')
    orig = ag.elegir_accion

    def espia(e):
        t = e.tablero
        estados.append({'dados': t['dados'], 'tiradas_restantes': t['tiradas_restantes'],
                        'puntajes': list(t['puntajes']),
                        'usadas': [set(t['usadas'][0]), set(t['usadas'][1])],
                        'ronda': t['ronda']})
        return orig(e)

    ag.elegir_accion = espia
    for k in range(n_partidas):
        jugar_partida([ag, crear_agente('Mejorada')], semilla + k, k % 2)
    return estados


def variantes():
    fe = FuncionEvaluacionMejorada(**PESOS_MEJORADA)
    return {
        'Rec': AgenteCachoMejorado('Rec', fe, modo='recursivo', usar_memo=False, usar_poda=False),
        'Rec+Poda': AgenteCachoMejorado('Rec+Poda', fe, modo='recursivo', usar_memo=False, usar_poda=True),
        'Rec+Memo': AgenteCachoMejorado('Rec+Memo', fe, modo='recursivo', usar_memo=True, usar_poda=False),
        'Rec+Poda+Memo': AgenteCachoMejorado('Rec+Poda+Memo', fe, modo='recursivo', usar_memo=True, usar_poda=True),
        'DP': AgenteCachoMejorado('DP', fe, modo='dp'),
    }


def clave_accion(a, dados):
    if a[0] == 'anotar':
        return a
    return ('tirar', tuple(sorted(dados[i] for i in range(5) if i not in a[1])))


def medir(estados, por_tiradas, csv_path=None):
    filas = []
    base = crear_agente('Base')
    for r, n in por_tiradas.items():
        sub = [e for e in estados if e['tiradas_restantes'] == r][:n]
        acciones_ref = None
        for nombre, ag in [('Base', base)] + list(variantes().items()):
            tiempos, evals, nodos, acciones = [], [], [], []
            for e in sub:
                if nombre == 'DP':
                    ag._cache_ctx.clear()          # sin reutilizar entre decisiones: peor caso
                est = ElEstado(0, 0, e, None)
                ag.evaluaciones = 0
                ag.nodos = 0
                t0 = time.perf_counter()
                a = ag.elegir_accion(est)
                tiempos.append(time.perf_counter() - t0)
                evals.append(ag.evaluaciones)
                nodos.append(getattr(ag, 'nodos', 0))
                acciones.append(clave_accion(a, e['dados']))
            if nombre == 'Rec':
                acciones_ref = acciones
            iguales = (sum(x == y for x, y in zip(acciones, acciones_ref)) if acciones_ref and nombre != 'Base'
                       else '')
            fila = {'tiradas_restantes': r, 'variante': nombre, 'n_estados': len(sub),
                    'tiempo_ms_media': round(1000 * statistics.mean(tiempos), 3),
                    'tiempo_ms_max': round(1000 * max(tiempos), 3),
                    'evaluaciones_media': round(statistics.mean(evals), 1),
                    'nodos_media': round(statistics.mean(nodos), 1),
                    'misma_accion_que_Rec': iguales}
            filas.append(fila)
            print(fila, flush=True)
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
    return filas


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n0', type=int, default=40)
    ap.add_argument('--n1', type=int, default=40)
    ap.add_argument('--n2', type=int, default=6)
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()
    est = recolectar_estados()
    random.Random(0).shuffle(est)
    medir(est, {0: args.n0, 1: args.n1, 2: args.n2}, args.csv)
