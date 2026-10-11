"""
Análisis estadístico de los torneos de Cacho y generación de figuras/tablas.

Entradas (carpeta datos/): resultados_cacho.csv, calibracion_cacho.csv,
eficiencia_cacho.csv, decisiones_cacho_resumen.csv
Salidas: datos/estadisticas_cacho.csv, datos/correlacion_cacho.csv,
articulo/figuras/*.pdf

Métodos: test binomial exacto (H0: p = 0.5, empates excluidos), intervalo de Wilson al
95 %, bootstrap (10 000 remuestreos) para la diferencia media de puntaje, test de
Welch, chi-cuadrado de homogeneidad entre emparejamientos y correlación de
Pearson/Spearman entre el valor de la función y la diferencia final.
"""

import csv
import os
import sys
from collections import defaultdict
from math import sqrt

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, '..', '..', 'datos')
FIGS = os.path.join(AQUI, '..', '..', 'articulo', 'figuras')
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})
AZUL, NARANJA, GRIS, VERDE = '#2a6f97', '#d17a22', '#7f7f7f', '#4c956c'


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - m, c + m


def bootstrap_media(x, B=10_000, semilla=0):
    rng = np.random.default_rng(semilla)
    x = np.asarray(x, float)
    medias = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def leer(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def estadisticas_torneos():
    filas = leer(os.path.join(DATOS, 'resultados_cacho.csv'))
    grupos = defaultdict(list)
    for f in filas:
        grupos[(f['agente_a'], f['agente_b'])].append(f)
    salida = []
    for (a, b), fs in grupos.items():
        pa = np.array([int(f['puntaje_a']) for f in fs])
        pb = np.array([int(f['puntaje_b']) for f in fs])
        n = len(fs)
        gb = int((pb > pa).sum()); ga = int((pa > pb).sum()); emp = n - ga - gb
        nd = ga + gb
        test = stats.binomtest(gb, nd, 0.5) if nd else None
        lo, hi = wilson(gb, nd)
        dif = pb - pa
        blo, bhi = bootstrap_media(dif)
        welch = stats.ttest_ind(pb, pa, equal_var=False)
        d_cohen = dif.mean() / np.sqrt((pa.var(ddof=1) + pb.var(ddof=1)) / 2) if n > 1 else 0
        # ventaja de abrir la partida
        abre_b = [f for f in fs if (int(f['semilla']) - 2026) % 2 == 1]   # partidas impares: abre b
        gana_abriendo = sum(int(f['puntaje_b']) > int(f['puntaje_a']) for f in abre_b)
        salida.append({
            'agente_a': a, 'agente_b': b, 'n': n, 'victorias_b': gb, 'victorias_a': ga, 'empates': emp,
            'tasa_b': round(gb / nd, 4) if nd else '', 'wilson_lo': round(lo, 4), 'wilson_hi': round(hi, 4),
            'p_binomial': float(f"{test.pvalue:.3g}") if test else '',
            'media_a': round(pa.mean(), 2), 'media_b': round(pb.mean(), 2),
            'sd_a': round(pa.std(ddof=1), 2), 'sd_b': round(pb.std(ddof=1), 2),
            'dif_media': round(dif.mean(), 2), 'boot_lo': round(blo, 2), 'boot_hi': round(bhi, 2),
            'p_welch': float(f"{welch.pvalue:.3g}"), 'd_cohen': round(d_cohen, 3),
            't_dec_a_ms': round(1000 * np.mean([float(f['tiempo_a']) for f in fs]), 3),
            't_dec_b_ms': round(1000 * np.mean([float(f['tiempo_b']) for f in fs]), 3),
            'evals_a': round(np.mean([float(f['evals_a']) for f in fs]), 1),
            'evals_b': round(np.mean([float(f['evals_b']) for f in fs]), 1),
            'tasa_b_abriendo': round(gana_abriendo / len(abre_b), 3) if abre_b else '',
        })
    with open(os.path.join(DATOS, 'estadisticas_cacho.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(salida[0].keys()))
        w.writeheader()
        w.writerows(salida)
    for s in salida:
        print(s)
    return salida, grupos


def chi2_homogeneidad(salida, pares):
    tabla = [[s['victorias_b'], s['victorias_a']] for s in salida if (s['agente_a'], s['agente_b']) in pares]
    chi2, p, gl, _ = stats.chi2_contingency(tabla)
    print(f"Chi-cuadrado homogeneidad {pares}: chi2={chi2:.2f}, gl={gl}, p={p:.3g}")
    return chi2, gl, p


def correlaciones(n=1000, semilla=70_000):
    sys.path.insert(0, AQUI)
    from torneo_cacho import crear_agente, jugar_partida
    ruta = os.path.join(DATOS, 'correlacion_cacho.csv')
    a, b = crear_agente('Mejorada'), crear_agente('BaseCorregida')
    por_ronda = defaultdict(lambda: ([], [], []))
    for k in range(n):
        reg = []
        res = jugar_partida([a, b], semilla + k, k % 2, registro=reg)
        final = res['puntajes'][0] - res['puntajes'][1]
        for r, eb, en in reg:
            por_ronda[r][0].append(eb); por_ronda[r][1].append(en); por_ronda[r][2].append(final)
    filas = []
    for r in sorted(por_ronda):
        eb, en, fin = map(np.array, por_ronda[r])
        filas.append({'ronda': r, 'n': len(fin),
                      'pearson_base': round(stats.pearsonr(eb, fin)[0], 4),
                      'spearman_base': round(stats.spearmanr(eb, fin)[0], 4),
                      'pearson_nueva': round(stats.pearsonr(en, fin)[0], 4),
                      'spearman_nueva': round(stats.spearmanr(en, fin)[0], 4)})
    with open(ruta, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    for f in filas:
        print(f)
    return filas


# ----------------------------------------------------------------------------- figuras

def fig_winrate(salida, etiquetas):
    sel = [s for s in salida if (s['agente_a'], s['agente_b']) in etiquetas]
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    y = np.arange(len(sel))
    for i, s in enumerate(sel):
        t = s['tasa_b']
        ax.errorbar(t, i, xerr=[[t - s['wilson_lo']], [s['wilson_hi'] - t]], fmt='o', color=AZUL,
                    ecolor=AZUL, capsize=3, ms=5)
        if t > 0.75:
            ax.text(s['wilson_lo'] - 0.03, i, f"{100 * t:.1f} %", va='center', ha='right', fontsize=8)
        else:
            ax.text(s['wilson_hi'] + 0.03, i, f"{100 * t:.1f} %", va='center', ha='left', fontsize=8)
    ax.axvline(0.5, color=GRIS, ls='--', lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([etiquetas[(s['agente_a'], s['agente_b'])] for s in sel])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Tasa de victorias del agente de la derecha (IC Wilson 95 %)', fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'cacho_winrate.pdf'))
    plt.close(fig)


def fig_puntajes(grupos):
    datos = {}
    for (a, b), fs in grupos.items():
        datos.setdefault(b, []).extend(int(f['puntaje_b']) for f in fs)
        datos.setdefault(a, []).extend(int(f['puntaje_a']) for f in fs)
    orden = [k for k in ['Base', 'BaseCorregida', 'MejoradaEVEstatico', 'MejoradaEtapas', 'MejoradaH1',
                         'MejoradaRiesgo', 'Mejorada']
             if k in datos]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    bp = ax.boxplot([datos[k] for k in orden], orientation="horizontal", widths=0.55, patch_artist=True,
                    medianprops=dict(color='black'), flierprops=dict(ms=2, alpha=.4))
    for patch, k in zip(bp['boxes'], orden):
        patch.set_facecolor(NARANJA if k.startswith('Base') else AZUL)
        patch.set_alpha(.55)
    ax.set_yticks(range(1, len(orden) + 1))
    ax.set_yticklabels(orden)
    ax.set_xlabel('Puntaje final por partida')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'cacho_puntajes.pdf'))
    plt.close(fig)


def fig_calibracion():
    filas = leer(os.path.join(DATOS, 'calibracion_cacho.csv'))
    ini = sorted({float(f['rho_ini']) for f in filas})
    fin = sorted({float(f['rho_fin']) for f in filas})
    M = np.full((len(fin), len(ini)), np.nan)
    for f in filas:
        M[fin.index(float(f['rho_fin'])), ini.index(float(f['rho_ini']))] = float(f['media'])
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    im = ax.imshow(M, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(ini))); ax.set_xticklabels([f"{x:.1f}" for x in ini])
    ax.set_yticks(range(len(fin))); ax.set_yticklabels([f"{x:.1f}" for x in fin])
    ax.set_xlabel(r'$\rho_{ini}$ (peso del futuro en la ronda 1)')
    ax.set_ylabel(r'$\rho_{fin}$ (ronda 10)')
    j, i = np.unravel_index(np.nanargmax(M), M.shape)
    ax.plot(i, j, marker='*', color='white', ms=10, mec='black', ls='none', label='máximo de la rejilla')
    ax.plot(ini.index(1.0), fin.index(1.0), marker='o', color='none', ms=11, mec='white', mew=1.6, ls='none',
            label='elegido tras validación')
    ax.legend(loc='lower left', fontsize=7, framealpha=.85)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Puntaje medio (1000 partidas)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'cacho_calibracion.pdf'))
    plt.close(fig)


def fig_eficiencia():
    filas = leer(os.path.join(DATOS, 'eficiencia_cacho.csv'))
    variantes = ['Base', 'Rec', 'Rec+Poda', 'Rec+Memo', 'Rec+Poda+Memo', 'DP']
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.5))
    for ax, campo, titulo in [(axs[0], 'tiempo_ms_media', 'Tiempo por decisión (ms)'),
                              (axs[1], 'evaluaciones_media', 'Evaluaciones de la función')]:
        for k, (r, col) in enumerate([(1, NARANJA), (2, AZUL)]):
            vals = []
            for v in variantes:
                x = [f for f in filas if f['variante'] == v and int(f['tiradas_restantes']) == r]
                vals.append(float(x[0][campo]) if x else np.nan)
            ax.bar(np.arange(len(variantes)) + (k - .5) * .38, vals, .38, color=col,
                   label=f'{r} tirada(s) restante(s)')
        ax.set_yscale('log')
        ax.set_xticks(range(len(variantes)))
        ax.set_xticklabels(variantes, rotation=35, ha='right', fontsize=7)
        ax.set_title(titulo, fontsize=9)
    axs[0].legend(fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'cacho_eficiencia.pdf'))
    plt.close(fig)


def fig_correlacion(filas):
    filas = [f for f in filas if f['ronda'] < 10]     # la ronda 10 es el resultado final (rho = 1)
    r = [f['ronda'] for f in filas]
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.plot(r, [f['spearman_base'] for f in filas], 'o--', color=NARANJA, label=r'$E_{base}$')
    ax.plot(r, [f['spearman_nueva'] for f in filas], 'o-', color=AZUL, label=r'$E_{nueva}$')
    ax.set_xlabel('Ronda completada')
    ax.set_ylabel(r'$\rho$ de Spearman con la dif. final')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'cacho_correlacion.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    salida, grupos = estadisticas_torneos()
    principales = {('Base', 'Mejorada'): 'Base vs Mejorada',
                   ('BaseCorregida', 'Mejorada'): 'B. corregida vs Mejorada',
                   ('Base', 'BaseCorregida'): 'Base vs B. corregida',
                   ('MejoradaH1', 'Mejorada'): 'Mej. h=1 vs Mejorada',
                   ('MejoradaEVEstatico', 'Mejorada'): 'Mej. EV estático vs Mejorada',
                   ('MejoradaEtapas', 'Mejorada'): 'Mej. por etapas vs Mejorada',
                   ('MejoradaRiesgo', 'Mejorada'): 'Mej. riesgo vs Mejorada',
                   ('Mejorada', 'Mejorada'): 'Mejorada vs Mejorada'}
    abl = [('MejoradaH1', 'Mejorada'), ('MejoradaEVEstatico', 'Mejorada'),
           ('MejoradaEtapas', 'Mejorada'), ('MejoradaRiesgo', 'Mejorada'), ('Mejorada', 'Mejorada')]
    pruebas = []
    for nombre, pares in [('ablaciones_y_espejo', abl), ('solo_ablaciones', abl[:4])]:
        chi2, gl, p = chi2_homogeneidad(salida, set(pares))
        pruebas.append({'prueba': 'chi2_homogeneidad_' + nombre, 'chi2': round(chi2, 3), 'gl': gl,
                        'p': float(f"{p:.3g}")})
    # bondad de ajuste a 50/50 para cada emparejamiento
    for s_ in salida:
        o = [s_['victorias_b'], s_['victorias_a']]
        c, p = stats.chisquare(o)
        pruebas.append({'prueba': f"chi2_50_50_{s_['agente_a']}_vs_{s_['agente_b']}", 'chi2': round(c, 3),
                        'gl': 1, 'p': float(f"{p:.3g}")})
    with open(os.path.join(DATOS, 'pruebas_chi2_cacho.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pruebas[0].keys()))
        w.writeheader()
        w.writerows(pruebas)
    fig_winrate(salida, principales)
    fig_puntajes(grupos)
    if os.path.exists(os.path.join(DATOS, 'calibracion_cacho.csv')):
        fig_calibracion()
    if os.path.exists(os.path.join(DATOS, 'eficiencia_cacho.csv')):
        fig_eficiencia()
    ruta_corr = os.path.join(DATOS, 'correlacion_cacho.csv')
    filas_corr = correlaciones() if not os.path.exists(ruta_corr) else [
        {k: (float(v) if k != 'ronda' else int(v)) for k, v in f.items()} for f in leer(ruta_corr)]
    fig_correlacion(filas_corr)
