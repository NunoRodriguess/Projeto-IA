#LICENCIATURA EM ENGENHARIA INFORMÁTICA
#MESTRADO integrado EM ENGENHARIA INFORMÁTICA

#Inteligência Artificial
#2022/23

#Draft Ficha 3


# Classe nodo para definiçao dos nodos
# cada nodo tem um nome e um id, poderia ter também informação sobre um ob jeto a guardar.....
class Node:
    def __init__(self, name,tipo,coordenadas,id=-1):     #  construtor do nodo....."
        self.m_id = id
        self.m_name = name
        self.m_tipo = tipo # Caso seja uma Porta "nodo verde" ou uma Rua "nodo azul"
        self.m_coordenada = coordenadas

        # posteriormente podera ser colocodo um objeto que armazena informação em cada nodo.....

    def __str__(self):
        return "node " + self.m_name + " " + str(self.m_coordenada)

    def __repr__(self):
        return "node " + self.m_name

    def setId(self, id):
        self.m_id = id


    def getId(self):
        return self.m_id

    def getName(self):
        return self.m_name

    def setType(self, tipo):
        self.m_tipo = tipo

    def setCoordenada(self,coordenada):
        self.m_coordenada = coordenada

    def getCoordenada(self):
        return self.m_coordenada

    def getCoordinates(self):
        return self.m_coordenada
    def getType(self):
        return self.m_tipo

    def __eq__(self, other):
        return self.m_name == other.m_name  # são iguais se nome igual, não usa o id

    def __hash__(self):
        return hash(self.m_name)
