"""
Función de evaluación base y función de evaluación mejorada para el Cacho.

E_base(s)   = (p_i + 0.6 * V_fut,i) - p_-i                       (consigna, ec. 1)

E_nueva(s)  = w1 f1(s) + w2(k) f2(s) + w3(s) f3(s)

    f1(s) = p_i - p_-i                         diferencia de puntaje actual
    f2(s) = V*_i - V*_-i                       diferencia de valor futuro dinámico
            V*_j = sum_{c no usada por j} EV*(c)
            EV*(c) = valor esperado de la categoría c jugando un turno
                     completo (3 tiradas) de forma óptima para c
    f3(s) = -(S_i - S_-i)                      diferencia de riesgo
            S_j = sum_{c no usada por j} sigma*(c)
    w1 = 1
    w2(k) = rho_ini + (rho_fin - rho_ini) * k / 9      pesos adaptativos por etapa
    w3(s) = kappa * tanh(Delta_proy / 20)              aversión al riesgo si se va
                                                       ganando, búsqueda de riesgo
                                                       si se va perdiendo

Todas las tablas de probabilidad se precalculan una sola vez (252 estados de
dados ordenados, 462 multiconjuntos de dados reservados).
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
from math import factorial, tanh, sqrt

import numpy as np

from AgenteCacho import CATEGORIAS, puntuar

N_RONDAS = 10

# ------------------------------------------------------------------
# Tablas combinatorias
# ------------------------------------------------------------------

ESTADOS = list(combinations_with_replacement(range(1, 7), 5))          # 252
IDX_ESTADO = {e: i for i, e in enumerate(ESTADOS)}
RESERVAS = [r for m in range(6) for r in combinations_with_replacement(range(1, 7), m)]  # 462
IDX_RESERVA = {r: i for i, r in enumerate(RESERVAS)}


def _prob_multiconjunto(combo):
    n = len(combo)
    formas = factorial(n)
    for v in Counter(combo).values():
        formas //= factorial(v)
    return formas / 6 ** n


def _construir_transiciones():
    """T[k, s] = P(estado final s | se reservan los dados k y se relanza el resto)."""
    T = np.zeros((len(RESERVAS), len(ESTADOS)))
    for k, reserva in enumerate(RESERVAS):
        n = 5 - len(reserva)
        for combo in combinations_with_replacement(range(1, 7), n):
            final = tuple(sorted(reserva + combo))
            T[k, IDX_ESTADO[final]] += _prob_multiconjunto(combo)
    return T


def _sub_multiconjuntos(dados):
    """Reservas distintas posibles desde 'dados' (poda de acciones equivalentes)."""
    distintas = set()
    for mascara in range(32):
        distintas.add(tuple(dados[i] for i in range(5) if mascara & (1 << i)))
    return sorted(distintas)


T = _construir_transiciones()
RESERVAS_DE = [[IDX_RESERVA[r] for r in _sub_multiconjuntos(e) if len(r) < 5] for e in ESTADOS]
PUNTOS = np.array([[puntuar(c, e) for e in ESTADOS] for c in CATEGORIAS], dtype=float)  # 10 x 252
P_INICIAL = T[IDX_RESERVA[()]]           # distribución de la primera tirada


def _valor_turno(hoja, tiradas):
    """Programación dinámica del turno: V_r(s) = max(hoja(s), max_k T_k · V_{r-1})."""
    V = hoja.copy()
    for _ in range(tiradas):
        EK = T @ V
        mejor_relanzar = np.array([EK[ks].max() for ks in RESERVAS_DE])
        V = np.maximum(hoja, mejor_relanzar)
    return V


def _politica_turno(hoja, tiradas):
    """Igual que _valor_turno pero devuelve también E[X^2] para obtener la varianza."""
    V, M2 = hoja.copy(), hoja ** 2
    for _ in range(tiradas):
        EK, EK2 = T @ V, T @ M2
        nuevoV, nuevoM2 = V.copy(), M2.copy()
        for s, ks in enumerate(RESERVAS_DE):
            j = int(np.argmax(EK[ks]))
            if EK[ks[j]] > hoja[s]:
                nuevoV[s], nuevoM2[s] = EK[ks[j]], EK2[ks[j]]
            else:
                nuevoV[s], nuevoM2[s] = hoja[s], hoja[s] ** 2
        V, M2 = nuevoV, nuevoM2
    return V, M2


def _estadisticas_categorias(tiradas_extra=2):
    ev, sd = {}, {}
    for c_i, c in enumerate(CATEGORIAS):
        V, M2 = _politica_turno(PUNTOS[c_i], tiradas_extra)
        media = float(P_INICIAL @ V)
        var = float(P_INICIAL @ M2) - media ** 2
        ev[c], sd[c] = media, sqrt(max(var, 0.0))
    return ev, sd


EV_ESTATICO = {c: float(PUNTOS[i].mean()) for i, c in enumerate(CATEGORIAS)}   # base (1 tirada, uniforme sobre multiconjuntos)
EV_DINAMICO, SD_DINAMICO = _estadisticas_categorias()


# ------------------------------------------------------------------
# Función base (consigna, ecuación 1)
# ------------------------------------------------------------------

def evaluacion_base(puntajes, usadas, jugador, alpha=0.6):
    rival = 1 - jugador
    futuro = sum(EV_ESTATICO[c] for c in CATEGORIAS if c not in usadas[jugador])
    return (puntajes[jugador] + alpha * futuro) - puntajes[rival]


# ------------------------------------------------------------------
# Función mejorada
# ------------------------------------------------------------------

class FuncionEvaluacionMejorada:
    """E_nueva(s) = w1 f1 + w2(k) f2 + w3(s) f3   (ver docstring del módulo)."""

    def __init__(self, rho_ini=1.0, rho_fin=1.0, kappa=0.0, w1=1.0, valores='dinamicos'):
        self.w1 = w1
        # 'estaticos' usa los EV de la base (una tirada) para aislar el efecto de EV*
        self.EV = EV_DINAMICO if valores == 'dinamicos' else EV_ESTATICO
        self.SD = SD_DINAMICO
        self.rho_ini = rho_ini
        self.rho_fin = rho_fin
        self.kappa = kappa

    # --- pesos ---------------------------------------------------
    def w2(self, ronda):
        ronda = min(max(ronda, 0), N_RONDAS - 1)
        return self.rho_ini + (self.rho_fin - self.rho_ini) * ronda / (N_RONDAS - 1)

    def w3(self, puntajes, usadas, jugador, ronda):
        if self.kappa == 0:
            return 0.0
        delta = self.f1(puntajes, jugador) + self.w2(ronda) * self.f2(usadas, jugador)
        return self.kappa * tanh(delta / 20.0)

    # --- características -----------------------------------------
    @staticmethod
    def f1(puntajes, jugador):
        return puntajes[jugador] - puntajes[1 - jugador]

    def f2(self, usadas, jugador):
        vi = sum(self.EV[c] for c in CATEGORIAS if c not in usadas[jugador])
        vr = sum(self.EV[c] for c in CATEGORIAS if c not in usadas[1 - jugador])
        return vi - vr

    @staticmethod
    def f3(usadas, jugador):
        si = sum(SD_DINAMICO[c] for c in CATEGORIAS if c not in usadas[jugador])
        sr = sum(SD_DINAMICO[c] for c in CATEGORIAS if c not in usadas[1 - jugador])
        return -(si - sr)

    # --- evaluación completa de un estado -------------------------
    def evaluar(self, puntajes, usadas, jugador, ronda):
        return (self.w1 * self.f1(puntajes, jugador)
                + self.w2(ronda) * self.f2(usadas, jugador)
                + self.w3(puntajes, usadas, jugador, ronda) * self.f3(usadas, jugador))

    # --- vector de hojas para el turno actual ---------------------
    def hojas_anotar(self, puntajes, usadas, jugador, ronda):
        """
        Para cada estado de dados s devuelve max_c E_nueva(s tras anotar c) - E_nueva(s actual)
        (las constantes no cambian la decisión). Vectorizado sobre los 252 estados.
        Devuelve (valores, indice_categoria_optima).
        """
        disp = [i for i, c in enumerate(CATEGORIAS) if c not in usadas[jugador]]
        w2 = self.w2(ronda)
        w3 = self.w3(puntajes, usadas, jugador, ronda)
        filas = []
        for i in disp:
            c = CATEGORIAS[i]
            # anotar c: +puntos en f1, se pierde EV*(c) de f2 y sigma*(c) de f3
            filas.append(self.w1 * PUNTOS[i] - w2 * self.EV[c] + w3 * self.SD[c])
        M = np.vstack(filas)
        arg = M.argmax(axis=0)
        return M.max(axis=0), np.array(disp)[arg]
