"""
Análisis de decisiones: ¿en qué estados la IA mejorada decide distinto que la base y
cuánto vale esa diferencia?

1. Se recolectan estados de decisión reales (partidas Mejorada vs Mejorada).
2. En cada estado se consulta a Base, BaseCorregida y Mejorada.
3. Para una muestra de estados donde difieren, se estima el valor de cada acción por
   simulación: se aplica la acción y el resto de la partida (solitario) lo juega la
   política mejorada; se usan las mismas semillas para ambas acciones (números
   aleatorios comunes). La diferencia de puntaje final medio es la pérdida de la
   decisión de la base.

Uso: python decisiones_cacho.py --partidas 20 --muestra 60 --rollouts 300
"""

import argparse
import csv
import random
import statistics

from AgenteCacho import CATEGORIAS, normalizar, puntuar
from AgenteIA.AgenteJugador import ElEstado
from eficiencia_cacho import recolectar_estados, clave_accion
from torneo_cacho import crear_agente


def tipo_accion(a, dados):
    if a[0] == 'anotar':
        return f"anotar {a[1]} ({puntuar(a[1], dados)} pts)"
    reserva = tuple(sorted(dados[i] for i in range(5) if i not in a[1]))
    return f"reservar {list(reserva)}"


def simular(estado, accion, politica, semilla):
    rng = random.Random(semilla)
    tab = {'dados': estado['dados'], 'tiradas_restantes': estado['tiradas_restantes'],
           'puntajes': list(estado['puntajes']),
           'usadas': [set(estado['usadas'][0]), set(estado['usadas'][1])],
           'ronda': estado['ronda']}
    base_pts = tab['puntajes'][0]
    primera = True
    while len(tab['usadas'][0]) < len(CATEGORIAS):
        while True:
            tab['ronda'] = len(tab['usadas'][0])
            acc = accion if primera else politica.elegir_accion(ElEstado(0, 0, tab, None))
            primera = False
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
        tab['dados'] = normalizar([rng.randint(1, 6) for _ in range(5)])
        tab['tiradas_restantes'] = 2
    return tab['puntajes'][0] - base_pts


def main(n_partidas, muestra, rollouts, csv_resumen, csv_casos):
    estados = recolectar_estados(n_partidas, semilla=60_000)
    base, corr, mej = crear_agente('Base'), crear_agente('BaseCorregida'), crear_agente('Mejorada')
    filas = []
    for e in estados:
        est = ElEstado(0, 0, e, None)
        a_b, a_c, a_m = (ag.elegir_accion(est) for ag in (base, corr, mej))
        filas.append((e, a_b, a_c, a_m))

    def resumen(idx, nombre):
        dif = [f for f in filas if clave_accion(f[idx], f[0]['dados']) != clave_accion(f[3], f[0]['dados'])]
        tipos = {'anota en vez de tirar': 0, 'tira en vez de anotar': 0,
                 'otra categoría': 0, 'otra reserva': 0}
        ceros = 0
        for e, *acc in dif:
            a, m = acc[idx - 1], acc[2]
            if a[0] == 'anotar' and m[0] == 'tirar':
                tipos['anota en vez de tirar'] += 1
            elif a[0] == 'tirar' and m[0] == 'anotar':
                tipos['tira en vez de anotar'] += 1
            elif a[0] == 'anotar':
                tipos['otra categoría'] += 1
            else:
                tipos['otra reserva'] += 1
            if a[0] == 'anotar' and puntuar(a[1], e['dados']) == 0 and e['tiradas_restantes'] > 0:
                ceros += 1
        return dif, dict(total=len(filas), distintas=len(dif), anota_cero_con_tiradas=ceros, **tipos)

    out_resumen, out_casos = [], []
    for idx, nombre in [(1, 'Base'), (2, 'BaseCorregida')]:
        dif, res = resumen(idx, nombre)
        res['agente'] = nombre
        rng = random.Random(7)
        sub = rng.sample(dif, min(muestra, len(dif)))
        perdidas = []
        for e, *acc in sub:
            a, m = acc[idx - 1], acc[2]
            va = [simular(e, a, mej, 90_000 + k) for k in range(rollouts)]
            vm = [simular(e, m, mej, 90_000 + k) for k in range(rollouts)]
            d = [y - x for x, y in zip(va, vm)]
            perdidas.append(statistics.mean(d))
            out_casos.append({'agente': nombre, 'dados': e['dados'],
                              'tiradas_restantes': e['tiradas_restantes'],
                              'usadas': ' '.join(sorted(e['usadas'][0])) or '-',
                              'accion_agente': tipo_accion(a, e['dados']),
                              'accion_mejorada': tipo_accion(m, e['dados']),
                              'valor_agente': round(statistics.mean(va), 2),
                              'valor_mejorada': round(statistics.mean(vm), 2),
                              'ganancia_mejorada': round(statistics.mean(d), 2),
                              'ee_ganancia': round(statistics.stdev(d) / len(d) ** 0.5, 2)})
        res['muestra_simulada'] = len(sub)
        res['ganancia_media_por_decision'] = round(statistics.mean(perdidas), 2)
        res['frac_decisiones_mejorada_mejor'] = round(sum(p > 0 for p in perdidas) / len(perdidas), 3)
        out_resumen.append(res)
        print(res, flush=True)

    for path, rows in [(csv_resumen, out_resumen), (csv_casos, out_casos)]:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--partidas', type=int, default=20)
    ap.add_argument('--muestra', type=int, default=60)
    ap.add_argument('--rollouts', type=int, default=300)
    ap.add_argument('--csv', default='../../datos/decisiones_cacho_resumen.csv')
    ap.add_argument('--casos', default='../../datos/decisiones_cacho_casos.csv')
    a = ap.parse_args()
    main(a.partidas, a.muestra, a.rollouts, a.csv, a.casos)
