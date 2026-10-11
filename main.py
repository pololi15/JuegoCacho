from EntornoCacho import EntornoCacho
from AgenteCachoIA import AgenteCachoIA
from AgenteCachoHumano import AgenteCachoHumano
from VistaCacho import VistaCacho


if __name__ == "__main__":
    entorno = EntornoCacho()

    ia = AgenteCachoIA(nombre="IA", altura=1)
    humano = AgenteCachoHumano(nombre="Tú")

    entorno.insertar(ia)
    entorno.insertar(humano)

    vista = VistaCacho(entorno)
    vista.loop()