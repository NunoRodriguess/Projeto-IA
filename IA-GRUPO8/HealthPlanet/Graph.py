#LICENCIATURA EM ENGENHARIA INFORMÁTICA
#MESTRADO integrado EM ENGENHARIA INFORMÁTICA

#Inteligência Artificial
#2022/23

#Draft Ficha 3

# Classe grafo para representaçao de grafos,
import math
import random
import threading
import time
from queue import Queue, PriorityQueue
import pickle
import copy
import osmnx as ox
import pandas as pd
import networkx as nx  # biblioteca de tratamento de grafos necessária para desnhar graficamente o grafo
import matplotlib.pyplot as plt  # idem

from nodo import Node


class Grafo:

    def __init__(self, directed=True):
        self.m_nodes = []
        self.m_directed = directed
        self.m_graph = {}  # dicionario para armazenar os nodos e arestas
        self.m_h = {}  # dicionario para posterirmente armazenar as heuristicas para cada nodo -< pesquisa informada
        self.m_blocked={} #arestas temporareamente bloquadas
        self.m_arestas = {} # atribuir um nome às arestas

    #############
    # Escrever o grafo como string
    #############
    def __str__(self):
        out = ""
        for key in self.m_graph.keys():
            out = out + "node" + str(key) + ": " + str(self.m_graph[key]) + "\n"
            return out

    ################################
    # Encontrar nodo pelo nome
    ################################
    def get_node_by_name(self, name):

        for node in self.m_nodes:
            if node.getName() == name:
                return node


        return None

    ##############################
    # Imprimir arestas
    ##############################
    def congestaoTotal(self,path):
        total_congestion = 0
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]

            # Find the edge connecting node1 and node2
            edge = None
            for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[node1]:
                if adjacente == node2:
                    edge = (node1, node2, status, peso, taxa_pegada, taxa_congestao)
                    break

            if edge:
                total_congestion += edge[5]  # Add the taxa_congestao value to the total

        return total_congestion

    #############################
    # Heuristica
    #############################
    def manhattan_distance(self,point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = abs(x2 - x1) + abs(y2 - y1)
        return distance

    def health_heuristic(self,inicio, fim):
        # caminho mais curto possível teórico menor distância teórica
        fimNode = self.get_node_by_name(fim)
        inicioNode = self.get_node_by_name(inicio)
        #dist = self.calcula_pegada(self.procura_BFS(inicio, fim)[0],"carro",50)
        dist = self.manhattan_distance(inicioNode.getCoordenada(),fimNode.getCoordenada())
        # Distância de Manhathan
        heu = dist
        return heu

    def all_heuristics(self,fim):
        for n in self.m_nodes:
            self.add_heuristica(n.getName(),n.getType(),n.getCoordenada(),self.health_heuristic(n.getName(),fim))

    #############################
    # Imprimir Aresta
    #############################
    def imprime_aresta(self):
        listaA = ""
        lista = self.m_graph.keys()
        for nodo in lista:
            for (nodo2,B,custo,taxa_pegada,taxa_congestao) in self.m_graph[nodo]:
                listaA = listaA + nodo + " ->" + nodo2 + " custo:" + str(custo) + " pegada:" + str(taxa_pegada) +" congestao:"+ str(taxa_congestao) + "\n"
        return listaA

    #############################
    # Adicionar nodo ao grafo
    #############################
    def add_node(self, node_data):
        self.m_nodes.append(node_data)
        self.m_graph[node_data.getName()] = []

    #############################
    # Adicionar   aresta no grafo
    #############################
    def add_edge(self, node1, tipo1,coordenada1, node2, tipo2,coordenada2, weight,biDirected,taxa_pegada = 0, taxa_congestao = 0):
        n1 = Node(node1, tipo1,coordenada1)
        n2 = Node(node2, tipo2,coordenada2)

        if n1 not in self.m_nodes:
            self.m_nodes.append(n1)
            self.m_graph[node1] = list()
        else:
            n1 = self.get_node_by_name(node1)

        if n2 not in self.m_nodes:
            self.m_nodes.append(n2)
            self.m_graph[node2] = list()
        else:
            n2 = self.get_node_by_name(node2)

        self.m_graph[node1].append((node2,True,weight,taxa_pegada,taxa_congestao))

        if not self.m_directed or biDirected == True:
            self.m_graph[node2].append((node1,True,weight,taxa_pegada,taxa_congestao))


    #############################
    # Devolver nodos do Grafo
    ############################

    def getNodes(self):
        return self.m_nodes

    ###############################
    # Devolver o custo de uma aresta
    ################################
    def get_arc_cost(self, node1, node2):
        custoT = math.inf
        a = self.m_graph[node1]  # lista de arestas para aquele nodo
        for (nodo,status,custo,taxa_pegada,taxa_congestao) in a:
            if nodo == node2:
                custoT = custo

        return custoT
    ##############################
    #  Dado um caminho calcula o seu custo:
    ###############################

    def calcula_custo(self, caminho):
        teste = caminho
        custo = 0
        i = 0
        while i + 1 < len(teste):
            custo = custo + self.get_arc_cost(teste[i], teste[i + 1])
            i = i + 1
        return custo

    def calcula_tempo(self, caminho, vehicle,peso):

        if vehicle == "mota":
            vel = 20-peso*0.5  # Km/h
        elif vehicle == "carro":
            vel = 50-peso*0.1
        else:
            vel = 10-peso*0.6

        total_time = 0
        i = 0
        while i + 1 < len(caminho):
            node1 = caminho[i]
            node2 = caminho[i + 1]

            edge = None
            for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[node1]:
                if adjacente == node2:
                    edge = (node1, node2, status, peso, taxa_pegada, taxa_congestao)
                    break

            if edge:

                distance = edge[3]
                congestion_tax = edge[5]
                time_for_edge = distance / vel
                time_for_edge *= (1 + congestion_tax)

                total_time += time_for_edge

            i += 1

        return total_time

    def calcula_pegada(self, caminho, vehicle,peso):
        # caminho is a list of nodes
        if vehicle == "mota":
            emission_rate = 60+0.05*peso  # Rates são de g/km
        elif vehicle == "carro":
            emission_rate = 122.3+0.03*peso
        else:
            emission_rate = 0

        total_emission = 0
        i = 0
        while i + 1 < len(caminho):
            node1 = caminho[i]
            node2 = caminho[i + 1]

            edge = None
            for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[node1]:
                if adjacente == node2:
                    edge = (node1, node2, status, peso, taxa_pegada, taxa_congestao)
                    break

            if edge:
                distance = edge[3]  # Distancia é o peso da aresta
                congestion_tax = edge[5]

                emission_for_edge = (distance * emission_rate) * (1 + congestion_tax)

                total_emission += emission_for_edge

            i += 1

        return total_emission

    ################################################################################
    # Procura DFS ( versão recursiva e não recursiva)
    ####################################################################################
    def procura_DFS(self, start, end, path=None, visited=None, expansion_order=None):
        if path is None:
            path = []
        if visited is None:
            visited = set()
        if expansion_order is None:
            expansion_order = []

        path.append(start)
        visited.add(start)
        expansion_order.append(start)

        if start == end:
            custoT = self.calcula_custo(path)
            return (path, custoT, expansion_order)

        for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[start]:
            if adjacente not in visited and status:
                resultado = self.procura_DFS(adjacente, end, path, visited, expansion_order)
                if resultado is not None:
                    return resultado

        path.pop()
        return None

    def procura_DFS2(self, start, end):
        visited = set()
        stack = [(start, [start], [start])]  # Stack stores (node, path, expansion_order)

        while stack:
            current, path, expansion_order = stack.pop()
            visited.add(current)

            if current == end:
                custoT = self.calcula_custo(path)
                return (path, custoT, expansion_order)

            for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[current]:
                if self.m_blocked.get(current) == adjacente:
                    continue
                if adjacente not in visited and status:
                    new_path = path + [adjacente]
                    new_expansion_order = expansion_order + [adjacente]
                    stack.append((adjacente, new_path, new_expansion_order))

        return None
    #####################################################
    # Procura Iterativa
    ######################################################
    def iterative_deepening_DFS(self, start, end):
        depth_limit = 0
        while True:
            result = self.DFS_with_limit(start, end, depth_limit)
            if result is not None:
                path, cost, expansion_order = result
                return path, cost, expansion_order, depth_limit

            depth_limit += 1
    def DFS_with_limit(self, start, end, depth_limit):
        visited = set()
        stack = [(start, [start], [start], 0)]  # (node, path, expansion_order, current_depth)

        while stack:
            current, path, expansion_order, depth = stack.pop()
            visited.add(current)

            if current == end:
                custoT = self.calcula_custo(path)
                return (path, custoT, expansion_order)

            if depth < depth_limit:
                for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[current]:
                    if self.m_blocked.get(current) == adjacente:
                        continue
                    if adjacente not in visited and status:
                        new_path = path + [adjacente]
                        new_expansion_order = expansion_order + [adjacente]
                        stack.append((adjacente, new_path, new_expansion_order, depth + 1))

        return None

    #####################################################
    # Procura BFS
    ######################################################
    def procura_BFS(self, start, end):
        # definir nodos visitados para evitar ciclos
        visited = set()
        fila = Queue()

        # adicionar o nodo inicial à fila e aos visitados
        fila.put(start)
        visited.add(start)

        # garantir que o start node nao tem pais...
        parent = dict()
        parent[start] = None

        expansion_order = []  # Track the expansion order
        path_found = False
        while not fila.empty() and path_found == False:
            nodo_atual = fila.get()

            if nodo_atual not in expansion_order:
                expansion_order.append(nodo_atual)

            if nodo_atual == end:
                path_found = True
            else:
                for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[nodo_atual]:
                    if self.m_blocked.get(nodo_atual) == adjacente:
                        continue
                    if adjacente not in visited and status:
                        fila.put(adjacente)
                        parent[adjacente] = nodo_atual
                        visited.add(adjacente)

        # Reconstruir o caminho
        path = []
        custo = 0
        if path_found:
            path.append(end)
            while parent[end] is not None:
                path.append(parent[end])
                end = parent[end]
            path.reverse()
            # função calcula custo caminho
            custo = self.calcula_custo(path)
        return (path, custo, expansion_order)

    ###################################################
    # Função   getneighbours, devolve vizinhos de um nó
    ####################################################

    def getNeighbours(self, nodo):
        lista = []
        for (adjacente,status,peso,taxa_pegada,taxa_congestao) in self.m_graph[nodo]:
            lista.append((adjacente,taxa_pegada*peso*0.65 + taxa_congestao*peso*0.35))
        return lista

    ###############################
    #  Procura Custo Uniforme
    ###############################
    def procura_custo_uniforme(self, start, goal):
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {}
        cost_so_far = {start: 0}
        expansion_order = []

        while not frontier.empty():
            current_cost, current_node = frontier.get()

            if current_node not in expansion_order:
                expansion_order.append(current_node)

            if current_node == goal:
                break

            for next_node,status,peso,taxa_pegada,taxa_congestao in self.m_graph[current_node]:
                if self.m_blocked.get(current_node) == next_node:
                    continue

                new_cost = cost_so_far[current_node] + taxa_pegada*peso*0.65 + taxa_congestao*peso*0.35

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost
                    frontier.put((priority, next_node))
                    came_from[next_node] = current_node

        # Reconstroi caminho

        path = [goal]
        while goal != start:
            goal = came_from[goal]
            path.append(goal)

        path.reverse()
        return path, self.calcula_custo(path), expansion_order

    ##########################################################################
    #  Desenha Grafo
    ##########################################################################
    def desenha(self):
        lista_v = self.m_nodes
        g = nx.Graph()

        for nodo in lista_v:
            n = nodo.getName()
            node_type = nodo.getType()
            if node_type == "Porta":
                node_color = "green"
                edge_color = "red"
            elif node_type == "Rua":
                node_color = "blue"
                edge_color = "yellow"
            else:
                node_color = "gray"
                edge_color = "black"

            node_coord = nodo.getCoordenada()
            g.add_node(n, color=node_color, pos=node_coord)

            for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[n]:
                g.add_edge(n, adjacente, weight=peso, color=edge_color)

        node_positions = {n: data['pos'] for n, data in g.nodes(data=True)}

        node_colors = [g.nodes[n]['color'] for n in g.nodes()]
        edge_colors = [g.edges[e]['color'] for e in g.edges()]

        plt.figure(figsize=(20, 20))  # Adjust the size here
        ax = plt.gca()
        ax.set_facecolor('gray')

        nx.draw(
            g, node_positions, with_labels=True, node_color=node_colors, node_size=20, edge_color=edge_colors,
            width=0.4, font_size=1  # Adjust node_size and font_size as needed
        )
        labels = nx.get_edge_attributes(g, 'weight')
        nx.draw_networkx_edge_labels(g, node_positions, edge_labels=labels, font_size=2)

        plt.show()

    ##########################################################################
    #  add_heuristica   -> define heuristica para cada nodo
    ##########################################################################

    def add_heuristica(self, n,tipo,coordenada, estima):
        n1 = Node(n,tipo,coordenada)
        if n1 in self.m_nodes:
            self.m_h[n] = estima

    def get_edges(self,nome):
        return self.m_graph[nome]

    #######################################################################
    #    heuristica   -> define heuristica para cada nodo 1 por defeito....
    #    apenas para teste de pesquisa informada
    #######################################################################

    def heuristica(self):
        nodos = self.m_graph.keys
        for n in nodos:
            self.m_h[n] = 1
        return (True)


    ##########################################

    def calcula_est(self, estima):
        l = list(estima.keys())
        min_estima = estima[l[0]]
        node = l[0]
        for k, v in estima.items():
            if v < min_estima:
                min_estima = v
                node = k
        return node

    ##########################################
    #    A*
    ##########################################

    def procura_aStar(self, start, end):
        open_list = {start}
        closed_list = set([])

        g = {}
        g[start] = 0

        parents = {}
        parents[start] = start

        expansion_order = []  # Manter a ordem de expansão

        while len(open_list) > 0:
            calc_heurist = {}
            flag = 0
            n = None

            for v in open_list:
                if n is None:
                    n = v
                else:
                    flag = 1
                    calc_heurist[v] = g[v] + self.getH(v)

            if flag == 1:
                min_estima = self.calcula_est(calc_heurist)
                n = min_estima

            if n is None:
                print('Path does not exist!')
                return None

            if n not in expansion_order:
                expansion_order.append(n)

            if n == end:
                reconst_path = []
                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start)
                reconst_path.reverse()

                return (reconst_path, self.calcula_custo(reconst_path), expansion_order)

            for (m, weight) in self.getNeighbours(n):
                if self.m_blocked.get(n) == m:
                    continue
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None

    ###################################3
    # devolve heuristica do nodo
    ####################################
    def getH(self, nodo):
        if nodo not in self.m_h.keys():
            return 1000
        else:
            return (self.m_h[nodo])

    def getFullH(self, nodo):
        if nodo not in self.m_h.keys():
            return 1000
        else:
            return (self.m_h[nodo])
    ##########################################
    #   Greedy
    ##########################################
    def greedy(self, start, end):
        open_list = set([start])
        closed_list = set([])

        parents = {}
        parents[start] = start

        expansion_order = []  # Manter a ordem de expansão

        while len(open_list) > 0:
            n = None

            for v in open_list:
                if n is None or self.m_h[v] < self.m_h[n]:
                    n = v

            if n is None:
                print('Path does not exist!')
                return None

            if n not in expansion_order:
                expansion_order.append(n)

            if n == end:
                reconst_path = []
                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start)
                reconst_path.reverse()

                return (reconst_path, self.calcula_custo(reconst_path), expansion_order)

            for (m, weight) in self.getNeighbours(n):
                if self.m_blocked.get(n) == m:
                    continue
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n

            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None
    ##########################################
    #   Gerar Grafo
    ##########################################
    def generate_graph(n, m, x, y): # gera um grafo random, não é muito bom tendo em conta a nossa abstração
        g = Grafo()

        nodes = [Node(f"Node_{i}", "Rua", (round(random.uniform(x, y), 1), i)) for i in range(n)]
        for node in nodes:
            g.add_node(node)

        created_edges = set()


        for node in nodes:
            other_node = random.choice([n for n in nodes if n != node])
            distance = abs(node.getCoordenada()[0] - other_node.getCoordenada()[0])
            weight = round(distance, 2)

            g.add_edge(
                node.getName(), node.getType(), node.getCoordenada(),
                other_node.getName(), other_node.getType(), other_node.getCoordenada(),
                weight, True, round(random.uniform(0.1, 0.5), 2), round(random.uniform(0.2, 0.7), 2)
            )
            created_edges.add((node.getName(), other_node.getName()))

        while len(created_edges) < m:
            node1 = random.choice(nodes)
            node2 = random.choice(nodes)


            if node1 != node2 and (node1.getName(), node2.getName()) not in created_edges and (
                    node2.getName(), node1.getName()) not in created_edges:
                distance = abs(node1.getCoordenada()[0] - node2.getCoordenada()[0])
                weight = round(distance, 2)

                g.add_edge(
                    node1.getName(), node1.getType(), node1.getCoordenada(),
                    node2.getName(), node2.getType(), node2.getCoordenada(),
                    weight, True, round(random.uniform(0.1, 0.5), 2), round(random.uniform(0.2, 0.7), 2)
                )
                created_edges.add((node1.getName(), node2.getName()))

        return g

    def generate_graphGridLike1(x, y):
        g = Grafo()

        distance = 0.3  # Define the distance between nodes

        nodes = [Node(f"Node_{i}_{j}", "Rua", (i * distance, j * distance)) for i in range(x) for
                 j in range(y)]
        for node in nodes:
            g.add_node(node)

        created_edges = set()

        for i in range(x):
            for j in range(y):
                # Verticais
                if j < y - 1:
                    node1 = f"Node_{i}_{j}"
                    node2 = f"Node_{i}_{(j + 1)}"
                    if (node1, node2) not in created_edges and (node2, node1) not in created_edges:
                        g.add_edge(
                            node1, "Rua", (i * distance, j * distance),
                            node2, "Rua", (i * distance, (j + 1) * distance),
                            distance, True, round(random.uniform(0.1, 0.5), 2), round(random.uniform(0.2, 0.7), 2)
                        )
                        created_edges.add((node1, node2))

                # Horizontais
                if i < x - 1:
                    node1 = f"Node_{i}_{j}"
                    node2 = f"Node_{(i + 1)}_{j}"
                    if (node1, node2) not in created_edges and (node2, node1) not in created_edges:
                        g.add_edge(
                            node1, "Rua", (i * distance, j * distance),
                            node2, "Rua", ((i + 1) * distance, j * distance),
                            distance, True, round(random.uniform(0.1, 0.5), 2), round(random.uniform(0.2, 0.7), 2)
                        )
                        created_edges.add((node1, node2))

        return g
    def has_edge(self, node1, node2):
        if node1 in self.m_graph and node2 in self.m_graph:
            edges_node1 = [edge[0] for edge in self.m_graph[node1]]
            edges_node2 = [edge[0] for edge in self.m_graph[node2]]

            return node2 in edges_node1 or node1 in edges_node2

        return False

    def generate_graphGridLike2(x, y):
        g = Grafo()
        fixed_distance = 0.3  # Define the fixed distance between nodes

        # Create the grid
        nodes = [
            Node(f"Node_{i}_{j}", "Rua", (i * fixed_distance + random.uniform(-0.115, 0.115), j * fixed_distance  + random.uniform(-0.115, 0.115)))
            for i in range(x) for j in range(y)
        ]
        for node in nodes:
            g.add_node(node)

        for i in range(x):
            for j in range(y):
                # Vertical edges
                if j < y - 1:
                    node1 = f"Node_{i}_{j}"
                    node2 = f"Node_{i}_{j + 1}"
                    n1 = g.get_node_by_name(node1)
                    n2 = g.get_node_by_name(node2)
                    d = g.distance(node1, node2)
                    g.add_edge(
                        node1, "Rua", n1.getCoordinates(),
                        node2, "Rua", n2.getCoordinates(),
                        d, True, round(random.uniform(0.1, 0.5), 2), round(random.uniform(0.2, 0.7), 2)
                    )

                # Horizontal edges
                if i < x - 1:
                    node1 = f"Node_{i}_{j}"
                    node2 = f"Node_{i + 1}_{j}"
                    n1 = g.get_node_by_name(node1)
                    n2 = g.get_node_by_name(node2)
                    d = g.distance(node1, node2)
                    g.add_edge(
                        node1, "Rua", n1.getCoordinates(),
                        node2, "Rua", n2.getCoordinates(),
                        d, True, round(random.uniform(0.1, 0.5), 2), round(random.uniform(0.2, 0.7), 2)
                    )

                # Diagonals random (top-left to bottom-right)
                if random.random() < 0.3 and i < x - 1 and j < y - 1:
                    node1 = f"Node_{i}_{j}"
                    node2 = f"Node_{i + 1}_{j + 1}"
                    n1 = g.get_node_by_name(node1)
                    n2 = g.get_node_by_name(node2)
                    d = g.distance(node1, node2)
                    g.add_edge(
                        node1, "Rua", n1.getCoordinates(),
                        node2, "Rua", n2.getCoordinates(),
                        d, True, round(random.uniform(0.1, 0.5), 2), round(random.uniform(0.2, 0.7), 2)
                    )

                # Diagonals Random (top-right to bottom-left)
                if random.random() < 0.3 and i > 0 and j < y - 1 and not g.has_edge(f"Node_{i - 1}_{j}",
                                                                                    f"Node_{i}_{j + 1}"):
                    node1 = f"Node_{i}_{j}"
                    node2 = f"Node_{i - 1}_{j + 1}"
                    n1 = g.get_node_by_name(node1)
                    n2 = g.get_node_by_name(node2)
                    d = g.distance(node1, node2)
                    g.add_edge(
                        node1, "Rua", n1.getCoordinates(),
                        node2, "Rua", n2.getCoordinates(),
                        d, True, round(random.uniform(0.1, 0.5), 2), round(random.uniform(0.2, 0.7), 2)
                    )

        return g

    ##########################################
    #   Simular Percurso
    ##########################################
    def search_token_sim(self,algc,start,end): # a heuristica é só posta uma vez ( se for baseada na distância)
        cam = None
        if algc == "a*":
            cam = self.procura_aStar(start,end)
        elif algc =="gu":
            self.all_heuristics(end)
            cam = self.greedy(start,end)
        elif algc =="cunif":
            cam = self.procura_custo_uniforme(start,end)
        elif algc == "bfs":
            cam = self.procura_BFS(start,end)
        elif algc =="dfs":
            cam = self.procura_DFS2(start,end)
        elif algc =="it":
            cam = self.iterative_deepening_DFS(start,end)

        return cam

    def simular_percurso(self, inicio, fim, estafeta_score,ac):
        g = copy.deepcopy(self)
        if ac =="gu" or ac =="a*":
            self.all_heuristics(fim)

        # Percurso inicial
        final_path, cost, expansion_order = g.search_token_sim(ac,inicio, fim)
        initial_path = final_path
        # Simula o ator a percorrer o percurso
        for i in range(len(final_path) - 1):
            current_node = final_path[i]
            next_node = final_path[i + 1]


            congestion = g.get_congestion(current_node, next_node)

            # Probablidade de conseguir ir
            probability = g.calculate_probability(congestion, estafeta_score)
            # Simula um novo caminho até ele conseguir
            while random.random() < probability:
                print("Deu asneira-> " + current_node + " " + next_node +" Motivo: " + g.generate_execuse())
                g.m_blocked[current_node] = next_node
                new_path, _, _ = g.search_token_sim(ac,current_node, fim)
                final_path = final_path[:i] + new_path  # Update ao caminho


        g.m_blocked = {}
        return initial_path,final_path,  # Retorna o planeado mais o real

    def desenhaWazeExp(self, path,step):
        lista_v = self.m_nodes
        g = nx.Graph()

        # Get the nodes up to the current step
        nodes_in_path = path

        for nodo in lista_v:
            n = nodo.getName()
            node_type = nodo.getType()
            node_color = "gray"
          #  if n not in nodes_in_path:
          #      continue
            if n in nodes_in_path:
                node_color = "orange"  # Highlight the nodes in the growing path in orange
            elif n == nodes_in_path[-1]:
                node_color = "red"  # Highlight the current node in red
            elif node_type == "Porta":
                node_color = "green"
            elif node_type == "Rua":
                node_color = "blue"
            else:
                node_color = "gray"

            node_coord = nodo.getCoordenada()
            g.add_node(n, color=node_color, pos=node_coord)
            for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[n]:
                g.add_edge(n, adjacente, weight=peso)

        node_positions = {n: data['pos'] for n, data in g.nodes(data=True)}

        node_colors = [g.nodes[n]['color'] for n in g.nodes()]
        edge_colors = ['black' for e in g.edges()]  # Default edge color

        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_facecolor('gray')

        nx.draw(
            g, node_positions, with_labels=True, node_color=node_colors, node_size=100, edge_color=edge_colors,
            width=0.4, font_size=4
        )
        labels = nx.get_edge_attributes(g, 'weight')
        nx.draw_networkx_edge_labels(g, node_positions, edge_labels=labels, font_size=8)

        plt.savefig(f"step_{step}_image.png")  # Save the image with a unique filename
        plt.close()

    def desenhaWazeExpMin(self, path, step):
        g = nx.Graph()

        # Get the nodes and edges in the path
        nodes_in_path = set(path)
        edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        for node in nodes_in_path:
            # Process nodes in the path
            node_color = "orange" if node != path[-1] else "red"  # Highlight path nodes in orange, last node in red
            node_coord = self.get_node_by_name(node).getCoordenada()
            g.add_node(node, color=node_color, pos=node_coord)

        for edge in edges_in_path:
            # Process edges in the path
            weight = self.get_arc_cost(edge[0],edge[1])
            g.add_edge(edge[0], edge[1], weight=weight)

        node_positions = {n: data['pos'] for n, data in g.nodes(data=True)}
        node_colors = [g.nodes[n]['color'] for n in g.nodes()]
        edge_colors = ['black' for e in g.edges()]

        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_facecolor('gray')

        nx.draw(
            g, node_positions, with_labels=True, node_color=node_colors, node_size=100, edge_color=edge_colors,
            width=0.4, font_size=4
        )
        labels = nx.get_edge_attributes(g, 'weight')
        nx.draw_networkx_edge_labels(g, node_positions, edge_labels=labels, font_size=8)

        plt.savefig(f"step_{step}_image.png")
        plt.close()

    def desenhaWazeCaminho(self, path, step):
        lista_v = self.m_nodes
        g = nx.Graph()

        # Get the nodes up to the current step
        nodes_in_path = path[:step]

        for nodo in lista_v:
            n = nodo.getName()
            node_type = nodo.getType()
            if n in nodes_in_path:
                node_color = "orange"  # Highlight the nodes in the growing path in orange
            elif n == nodes_in_path[-1]:
                node_color = "red"  # Highlight the current node in red
            elif node_type == "Porta":
                node_color = "green"
            elif node_type == "Rua":
                node_color = "blue"
            else:
                node_color = "gray"

            node_coord = nodo.getCoordenada()
            g.add_node(n, color=node_color, pos=node_coord)

            for (adjacente, status, peso, taxa_pegada, taxa_congestao) in self.m_graph[n]:
                g.add_edge(n, adjacente, weight=peso)

        node_positions = {n: data['pos'] for n, data in g.nodes(data=True)}

        node_colors = [g.nodes[n]['color'] for n in g.nodes()]
        edge_colors = ['black' for e in g.edges()]  # Default edge color

        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_facecolor('gray')

        nx.draw(
            g, node_positions, with_labels=True, node_color=node_colors, node_size=100, edge_color=edge_colors,
            width=0.4, font_size=0.5
        )
        labels = nx.get_edge_attributes(g, 'weight')
        nx.draw_networkx_edge_labels(g, node_positions, edge_labels=labels, font_size=0)

        plt.savefig(f"Caminho_{step}_image.png")  # Save the image with a unique filename
        plt.close()

    def simular_percursoWaze(self, inicio, fim, estafeta_score, vec, peso,ac):
        g = copy.deepcopy(self)
        lk = threading.Lock()
        caminho = [inicio]

        def update_edge_values(lock=lk):
            while True:
                with lock:
                    g.update_edge_values()
                time.sleep(2)

        edge_update_thread = threading.Thread(target=update_edge_values)
        edge_update_thread.daemon = True
        edge_update_thread.start()
        if ac =="gu" or ac =="a*":
            self.all_heuristics(fim)

        current_node = inicio
        step = 1
        while current_node != fim:
            with lk:
                path, cost, expansion_order = g.search_token_sim(ac,current_node, fim)

            next_node = path[1]
            congestion = g.get_congestion(current_node, next_node)
            probability = g.calculate_probability(congestion, estafeta_score)

            while random.random() < probability:
                print("Deu asneira-> " + current_node + " " + next_node + " Motivo: " + g.generate_execuse())
                g.m_blocked[current_node] = next_node
                with lk:
                    new_path, _, _ = g.search_token_sim(ac,current_node, fim)
                next_node = new_path[1]
                path = new_path

                # Simulate travel between nodes
            l = [current_node,next_node]
            demora_a_percorrer_aresta = g.calcula_tempo(l,vec,peso)
            time.sleep(demora_a_percorrer_aresta) # o tempo é dado em horas mas considerei segundos
            print(current_node + " -> " + next_node)
            caminho.append(next_node)  # Update ao caminho
            current_node = next_node
            g.desenhaWazeExpMin(path, step)  # Generate and save the image for this step
            step += 1  # Increment the step count

        g.m_blocked = {}
        self.edge_update_thread_running = False
        g.desenhaWazeCaminho(caminho, step)
        return caminho  # Retorna o caminho

    def get_congestion(self, node1, node2): # calcula a congestão
        custoT = math.inf
        a = self.m_graph[node1]  # lista de arestas para aquele nodo
        for (nodo, status, custo, taxa_pegada, taxa_congestao) in a:
            if nodo == node2:
                custoT = taxa_congestao

        return custoT
    def calculate_probability(self, congestion, estafeta_score): #calcula probablidade de erro
        t = congestion / 70
        te = (5 - estafeta_score) / 60
        return (t+te)

    def generate_execuse(self): # Escolhe uma desculpa
        c = random.random()
        desc = ""
        if 0 <= c <= 0.14:
            desc = "Estrada fechada para obras"
        elif 0.15 <= c <=0.20:
            desc = "Condutor em Segunda fila"
        elif 0.15 <= c <= 0.21:
            desc = "Acidente Grave"
        elif 0.21 <= c <= 0.90:
            desc = "Engano do Estafeta"
        elif 0.90 <= c <= 0.93:
            desc = "Parado pela policia"
        elif 0.93 <= c <= 0.935:
            desc = "Assalto em curso"
        elif 0.935 <= c <= 0.94:
            desc = "Derrocada na estrada"
        elif 0.94 <= c <= 0.95:
            desc = "Estrada com neve"
        elif 0.95 <= c <= 0.97:
            desc = "Adeptos do Vitoria na estrada"
        elif 0.97 <= c <= 0.9999:
            desc = "Desabamento da estrada"
        elif 0.9999 <= c <= 1:
            desc = "Metiorito na estrada"

        return desc

    def update_edge_values(self):
        for node in self.m_graph:
            for edge in self.m_graph[node]:
                congestion_index = random.uniform(0.2, 0.7)
                green_foot_index = random.uniform(0.2, 0.7)
                edge_index = self.m_graph[node].index(edge)
                self.m_graph[node][edge_index] = (
                    edge[0], edge[1], edge[2],
                    green_foot_index,
                    congestion_index
                )

    ##########################################
    #   Múltiplas entregas
    ########################################## (a*,gu,cunif,bfs,dfs,it)
    def search_token(self,algc,start,end):
        cam = None
        if algc == "a*":
            self.all_heuristics(end)
            cam = self.procura_aStar(start,end)
        elif algc =="gu":
            self.all_heuristics(end)
            cam = self.greedy(start,end)
        elif algc =="cunif":
            cam = self.procura_custo_uniforme(start,end)
        elif algc == "bfs":
            cam = self.procura_BFS(start,end)
        elif algc =="dfs":
            cam = self.procura_DFS2(start,end)
        elif algc =="it":
            cam = self.iterative_deepening_DFS(start,end)

        return cam[0]

    def distance(self,node1, node2):
        # Assuming nodes are tuples with (x, y) coordinates
        fimNode = self.get_node_by_name(node2)
        inicioNode = self.get_node_by_name(node1)
        return math.sqrt((inicioNode.getCoordenada()[0] - fimNode.getCoordenada()[0]) ** 2 + (inicioNode.getCoordenada()[1] - fimNode.getCoordenada()[1])  ** 2)


    def fazer_entrega(self,start,encomendas,algc,vec):

        first = True
        caminho = None
        custoT = 0
        custoP = 0
        l = []
        for pa in encomendas:
            l.append(int(pa[1]))

        totalPeso = sum(l)
        sorted_nodes = encomendas

        for (end,peso) in sorted_nodes:
            if first:
                caminho = self.search_token(algc,start,end)
                custoT  = self.calcula_tempo(caminho,vec,totalPeso)
                custoP = self.calcula_pegada(caminho,vec,totalPeso)
                totalPeso = totalPeso - peso
                first = False
            else:
                start = caminho.pop()
                novo_caminho = self.search_token(algc,start,end)
                custoT = custoT + self.calcula_tempo(caminho,vec,totalPeso)
                custoP = custoP + self.calcula_pegada(caminho, vec, totalPeso)
                totalPeso = totalPeso - peso
                caminho = caminho + novo_caminho

        custo = self.calcula_custo(caminho)

        return (caminho,custo,custoT,custoP)

    ##########################################
    #   Salvar/Carregar Grafo
    ##########################################
    def save_graph(graph, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(graph, file)

    def load_graph(file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)



    def generate_graph_osmnx(place_name):
        # Retrieve the street network for the defined location using OSMnx
        graph = ox.graph_from_place(place_name, network_type='drive')

        # Create a graph object to store the data based on your abstraction
        g = Grafo()  # Assuming you have a Grafo class defined

        # Add nodes from the OSMnx graph to your abstraction
        for node, data in graph.nodes(data=True):
            node_id = node
            node_type = "Rua"  # Assuming a default node type
            coordenada = (data['x'], data['y'])  # Latitude and longitude from OSMnx
            g.add_node(Node(str(node_id), node_type, coordenada))

        # Add edges from the OSMnx graph to your abstraction
        for u, v, data in graph.edges(data=True):
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]

            distance = data['length'] / 1000  # Assuming OSMnx calculates length as distance
            weight = round(distance, 2)  # Use distance as weight for the edge

            g.add_edge(
                str(u), "Rua", (u_data['x'], u_data['y']),
                str(v), "Rua", (v_data['x'], v_data['y']),
                weight, False, round(random.uniform(0.1, 0.5), 2), round(random.uniform(0.2, 0.7), 2)
            )

        return g

    def plot_graph(self):
        fig, ax = ox.plot_graph(self.m_graph, show=False, close=False)
        plt.tight_layout()
        plt.show()



