"""
Agentes de Cacho usados en la práctica.

- AgenteCachoIA            (archivo AgenteCachoIA.py): agente base del docente, sin cambios.
- AgenteCachoBaseCorregido : misma función base (alpha = 0.6, altura = 1) pero evaluando
                             siempre desde la perspectiva del jugador que decide.
- AgenteCachoMejorado      : función de evaluación nueva + selección de jugadas optimizada
                             (poda de reservas equivalentes, memoización / programación
                             dinámica del turno y profundidad adaptativa = tiradas restantes).

Todos los agentes reciben un ElEstado y devuelven ('tirar', frozenset(indices)) o
('anotar', categoria), con la misma interfaz que el programa base.
"""

import time
from functools import lru_cache

import numpy as np

from AgenteCacho import AgenteCacho, CATEGORIAS, puntuar, probabilidad_tirada, acciones_reservar, normalizar
from AgenteIA.AgenteJugador import ElEstado
from funcion_evaluacion import (ESTADOS, IDX_ESTADO, RESERVAS, RESERVAS_DE, T,
                                FuncionEvaluacionMejorada, evaluacion_base, PUNTOS)


def indices_para_reserva(dados, reserva):
    """Índices (sobre los dados ordenados) que hay que relanzar para quedarse con 'reserva'."""
    pendientes = list(reserva)
    tirar = []
    for i, v in enumerate(dados):
        if v in pendientes:
            pendientes.remove(v)
        else:
            tirar.append(i)
    return frozenset(tirar)


# ======================================================================
# Base corregida (solo arregla la perspectiva de la evaluación)
# ======================================================================

class AgenteCachoBaseCorregido(AgenteCacho):
    """Misma E_base y altura=1, pero la hoja se evalúa siempre para quien decide."""

    def __init__(self, nombre="BaseCorregida", altura=1, alpha=0.6):
        AgenteCacho.__init__(self, nombre=nombre, altura=altura)
        self.alpha = alpha
        self.evaluaciones = 0

    def _eval(self, tablero, yo):
        self.evaluaciones += 1
        return evaluacion_base(tablero['puntajes'], tablero['usadas'], yo, self.alpha)

    def elegir_accion(self, estado):
        yo = estado.jugador
        tab = estado.tablero
        dados = tab['dados']
        mejor, mejor_v = None, float('-inf')
        for accion in self.jugadas(estado):
            if accion[0] == 'anotar':
                nuevo = self.getResultado(estado, accion)
                v = self._eval(nuevo.tablero, yo)
            else:
                v = 0.0
                for nuevos, p in probabilidad_tirada(dados, accion[1]).items():
                    t = self._copiar_tablero(tab)
                    t['dados'] = nuevos
                    v += p * self._eval(t, yo)
            if v > mejor_v:
                mejor, mejor_v = accion, v
        if tab['tiradas_restantes'] == 0 or mejor is None:
            cands = [a for a in self.jugadas(estado) if a[0] == 'anotar']
            mejor = max(cands, key=lambda a: puntuar(a[1], dados))
        return mejor

    def programa(self):
        self.set_acciones([])
        self.set_acciones(self.elegir_accion(self.estado))


# ======================================================================
# Agente mejorado
# ======================================================================

class AgenteCachoMejorado(AgenteCacho):
    """
    Expectiminimax sobre el turno completo:
        nodo MAX   : elegir reserva (o anotar)
        nodo AZAR  : relanzar los dados no reservados
        hoja       : anotar la mejor categoría según E_nueva
    El nodo MIN del programa base se elimina dentro del turno porque el rival no puede
    afectar los dados ni la planilla propia; su influencia entra por f1, f2 y w3.

    modo='dp'         : implementación vectorizada (memoización completa por programación
                        dinámica sobre los 252 estados). Es la usada en el torneo.
    modo='recursivo'  : implementación recursiva clásica con banderas usar_memo y usar_poda,
                        usada solo para medir el efecto de cada optimización.
    horizonte         : None = profundidad adaptativa (todas las tiradas restantes);
                        un entero h limita la búsqueda a h tiradas (h=1 replica la miopía).
    """

    def __init__(self, nombre="Mejorada", fe=None, horizonte=None, modo='dp',
                 usar_memo=True, usar_poda=True):
        AgenteCacho.__init__(self, nombre=nombre, altura=3)
        self.fe = fe or FuncionEvaluacionMejorada()
        self.horizonte = horizonte
        self.modo = modo
        self.usar_memo = usar_memo
        self.usar_poda = usar_poda
        self.evaluaciones = 0      # evaluaciones de hoja (E_nueva) en la última decisión
        self.nodos = 0             # nodos expandidos en la última decisión
        self._cache_ctx = {}

    # ------------------------------------------------------------------
    def _contexto(self, tab, yo):
        usadas = tab['usadas']
        ronda = len(usadas[yo])
        clave = (frozenset(usadas[yo]), frozenset(usadas[1 - yo]), ronda,
                 tab['puntajes'][yo] - tab['puntajes'][1 - yo] if self.fe.kappa else 0)
        return clave, ronda

    def _niveles(self, tab, yo, h):
        """V[0..h] para el contexto actual (programación dinámica con caché por contexto)."""
        clave, ronda = self._contexto(tab, yo)
        if clave not in self._cache_ctx:
            hoja, cat = self.fe.hojas_anotar(tab['puntajes'], tab['usadas'], yo, ronda)
            self.evaluaciones += len(ESTADOS) * (len(CATEGORIAS) - len(tab['usadas'][yo]))
            niveles = [hoja]
            V = hoja
            for _ in range(3):
                EK = T @ V
                V = np.maximum(hoja, np.array([EK[ks].max() for ks in RESERVAS_DE]))
                niveles.append(V)
            if len(self._cache_ctx) > 64:
                self._cache_ctx.clear()
            self._cache_ctx[clave] = (niveles, cat)
        return self._cache_ctx[clave]

    def _decidir_dp(self, estado):
        tab, yo = estado.tablero, estado.jugador
        dados = normalizar(tab['dados'])
        r = tab['tiradas_restantes']
        h = r if self.horizonte is None else min(r, self.horizonte)
        if h == 0 and self._contexto(tab, yo)[0] not in self._cache_ctx:
            # sin tiradas: basta evaluar la hoja del estado actual (no hace falta la tabla)
            _, ronda = self._contexto(tab, yo)
            fe = self.fe
            w2, w3 = fe.w2(ronda), fe.w3(tab['puntajes'], tab['usadas'], yo, ronda)
            disp = [c for c in CATEGORIAS if c not in tab['usadas'][yo]]
            self.evaluaciones += len(disp)
            v, c = max((puntuar(c, dados) - w2 * fe.EV[c] + w3 * fe.SD[c], c) for c in disp)
            return ('anotar', c), v
        niveles, cat = self._niveles(tab, yo, 3)
        s = IDX_ESTADO[dados]
        hoja = niveles[0]
        mejor = ('anotar', CATEGORIAS[cat[s]])
        mejor_v = hoja[s]
        if h > 0:
            EK = T @ niveles[h - 1]
            ks = RESERVAS_DE[s]
            self.nodos += len(ks)
            j = int(np.argmax(EK[ks]))
            if EK[ks[j]] > mejor_v + 1e-9:
                mejor = ('tirar', indices_para_reserva(dados, RESERVAS[ks[j]]))
                mejor_v = EK[ks[j]]
        return mejor, mejor_v

    # ------------------------------------------------------------------
    # Versión recursiva (para medir memoización y poda)
    # ------------------------------------------------------------------
    def _decidir_recursivo(self, estado):
        tab, yo = estado.tablero, estado.jugador
        _, ronda = self._contexto(tab, yo)
        disp = [c for c in CATEGORIAS if c not in tab['usadas'][yo]]
        w2 = self.fe.w2(ronda)
        w3 = self.fe.w3(tab['puntajes'], tab['usadas'], yo, ronda)
        EV, SD = self.fe.EV, self.fe.SD

        def hoja(d):
            self.evaluaciones += len(disp)
            return max((puntuar(c, d) - w2 * EV[c] + w3 * SD[c], c) for c in disp)

        def acciones_tirar(d):
            if self.usar_poda:
                vistas = {}
                for reserva, tirar in acciones_reservar(d):
                    if tirar and reserva not in vistas:
                        vistas[reserva] = tirar
                return list(vistas.values())
            return [tirar for _, tirar in acciones_reservar(d) if tirar]

        def valor(d, r):
            self.nodos += 1
            v, _ = hoja(d)
            if r == 0:
                return v
            for tirar in acciones_tirar(d):
                ve = sum(p * valor_m(nd, r - 1) for nd, p in probabilidad_tirada(d, tirar).items())
                v = max(v, ve)
            return v

        memo = {}

        def valor_m(d, r):
            if not self.usar_memo:
                return valor(d, r)
            k = (d, r)
            if k not in memo:
                memo[k] = valor(d, r)
            return memo[k]

        dados = normalizar(tab['dados'])
        r = tab['tiradas_restantes']
        h = r if self.horizonte is None else min(r, self.horizonte)
        v_anotar, c = hoja(dados)
        mejor, mejor_v = ('anotar', c), v_anotar
        if h > 0:
            for tirar in acciones_tirar(dados):
                ve = sum(p * valor_m(nd, h - 1) for nd, p in probabilidad_tirada(dados, tirar).items())
                if ve > mejor_v + 1e-9:
                    mejor, mejor_v = ('tirar', tirar), ve
        return mejor, mejor_v

    # ------------------------------------------------------------------
    def elegir_accion(self, estado):
        self.evaluaciones = 0
        self.nodos = 0
        if self.modo == 'dp':
            accion, _ = self._decidir_dp(estado)
        else:
            accion, _ = self._decidir_recursivo(estado)
        return accion

    def programa(self):
        self.set_acciones([])
        self.set_acciones(self.elegir_accion(self.estado))

