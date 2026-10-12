from EntornoCacho import EntornoCacho
from AgenteCachoHumano import AgenteCachoHumano
from VistaCacho import VistaCacho

# IA mejorada (funcion_evaluacion.py + agente_ia.py). Para jugar contra la IA base del
# docente, usar en su lugar:  from AgenteCachoIA import AgenteCachoIA
#                             ia = AgenteCachoIA(nombre="IA", altura=1)
from agente_ia import AgenteCachoMejorado


if __name__ == "__main__":
    entorno = EntornoCacho()

    ia = AgenteCachoMejorado(nombre="IA")
    humano = AgenteCachoHumano(nombre="Tú")

    entorno.insertar(ia)
    entorno.insertar(humano)

    vista = VistaCacho(entorno)
    vista.loop()
