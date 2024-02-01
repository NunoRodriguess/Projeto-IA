
from Graph import Grafo
import random
class Posto:
    def __init__(self, nr_carros, nr_motas, nr_bicicleta, estafetas, entregas,grafo): #entregas is a list like [(weight,endNode)]
        self._nr_carros = nr_carros
        self._nr_motas = nr_motas
        self._nr_bicicleta = nr_bicicleta
        self._estafetas = estafetas
        self._entregas = entregas
        self._meu_grafo = grafo

    def getGrafo(self):
        return self._meu_grafo

    def getEntregas(self):
        return self._entregas



g = Grafo.load_graph("625P")
entregas = {
    1 : (2,'Node_0_10'),
    2 : (5,'Node_1_10'),
    3 : (20,'Node_0_22'),
}
p = Posto(nr_carros=1, nr_motas=1, nr_bicicleta=3, estafetas=2, entregas=entregas, grafo=g)

