"""
Ajuste experimental de los pesos w2(k) = rho_ini -> rho_fin (búsqueda en rejilla).

Con kappa = 0 las decisiones de la función mejorada dependen solo de la planilla propia,
por lo que maximizar el puntaje esperado en partidas de un jugador equivale a maximizar
la diferencia esperada contra cualquier rival. Se usan números aleatorios comunes
(mismas semillas para cada configuración) para reducir la varianza de la comparación.
Semillas de calibración: 10_000..10_000+N-1 (disjuntas de las del torneo, 2026..).

Uso: python calibrar_cacho.py --n 300 --csv ../../datos/calibracion_cacho.csv
"""

import argparse
import csv
import random
import statistics

from AgenteCacho import CATEGORIAS, normalizar, puntuar
from AgenteIA.AgenteJugador import ElEstado
from agente_ia import AgenteCachoMejorado
from funcion_evaluacion import FuncionEvaluacionMejorada

TIRADAS_EXTRA = 2


def partida_solitario(agente, semilla):
    rng = random.Random(semilla)
    tab = {'puntajes': [0, 0], 'usadas': [set(), set()]}
    for ronda in range(10):
        tab['dados'] = normalizar([rng.randint(1, 6) for _ in range(5)])
        tab['tiradas_restantes'] = TIRADAS_EXTRA
        while True:
            tab['ronda'] = ronda
            acc = agente.elegir_accion(ElEstado(0, 0, tab, None))
            if acc[0] == 'tirar':
                d = list(tab['dados'])
                for i in acc[1]:
                    d[i] = rng.randint(1, 6)
                tab['dados'] = normalizar(d)
                tab['tiradas_restantes'] -= 1
                continue
            tab['puntajes'][0] += puntuar(acc[1], tab['dados'])
            tab['usadas'][0].add(acc[1])
            break
    return tab['puntajes'][0]


def evaluar_config(rho_ini, rho_fin, n, base_semilla=10_000):
    ag = AgenteCachoMejorado('cal', FuncionEvaluacionMejorada(rho_ini, rho_fin, 0.0))
    return [partida_solitario(ag, base_semilla + k) for k in range(n)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--ini', default='0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0')
    ap.add_argument('--fin', default='0,0.2,0.4,0.6,0.8,1.0,1.2,1.4')
    ap.add_argument('--anexar', action='store_true', help='añadir al CSV existente')
    ap.add_argument('--pares', default=None,
                    help='validación: lista "ri:rf;ri:rf" evaluada con semillas nuevas (20_000..)')
    args = ap.parse_args()
    if args.pares:
        configs = [tuple(float(x) for x in p.split(':')) for p in args.pares.split(';')]
        semilla0 = 20_000
    else:
        configs = [(float(ri), float(rf)) for ri in args.ini.split(',') for rf in args.fin.split(',')]
        semilla0 = 10_000
    filas = []
    for ri, rf in configs:
        p = evaluar_config(ri, rf, args.n, semilla0)
        m, sd = statistics.mean(p), statistics.stdev(p)
        filas.append({'rho_ini': ri, 'rho_fin': rf, 'media': round(m, 3),
                      'desv': round(sd, 3), 'n': args.n})
        print(f"rho_ini={ri:.1f} rho_fin={rf:.1f}  media={m:6.2f}  sd={sd:5.2f}", flush=True)
    filas.sort(key=lambda f: -f['media'])
    print("Mejor:", filas[0])
    if args.csv:
        with open(args.csv, 'a' if args.anexar else 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            if not args.anexar:
                w.writeheader()
            w.writerows(filas)
