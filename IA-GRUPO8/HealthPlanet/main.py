from Graph import Grafo

import datetime

def main():
    #g = Grafo.generate_graph_osmnx("Paços de Ferreira,Portugal")
    g = ""

    posto = "não Atribuido"
    saida = -1
    while saida != 0:
        if (g == ""):


            print("1-Gerar Grafo")
            print("3-Carregar Grafo")
            print("0-Saír")
            saida = int(input("introduza a sua opcao-> "))
            if saida == 0:
                print("saindo.......")
            elif saida == 1:
                print("Aleatorio ou em Grelha?(a/g) ")
                t = input("->")
                if (t =="a"):
                    n = int (input("Número de Nós: "))
                    m = int (input("Número de Arestas: "))
                    x = int (input("Mínimo coordenada: "))
                    y = int (input("Máximo coordenada: "))
                    g = Grafo.generate_graph(n,m,x,y)
                else:
                    x = int(input("Coordenadas de lado: "))
                    print("Grelha Perfeita ou Imperfeita?(p/i) ")
                    t2 = input("->")
                    if (t2 == "p"):
                        g = Grafo.generate_graphGridLike1(x,x)
                    else:
                        g = Grafo.generate_graphGridLike2(x, x)

                l = input("prima enter para continuar")

            elif saida ==3:
                path = input("Insira o Path")
                g = Grafo.load_graph(path)
                l = input("prima enter para continuar")

            else:
                print("you didn't add anything")
                l = input("prima enter para continuar")
        else:
            print("1-Imprimir Grafo")
            print("2-Desenhar Grafo")
            print("3-Imprimir  nodos de Grafo")
            print("4-Imprimir arestas de Grafo")
            print("5-DFS")
            print("6-BFS")
            print("7-A*")
            print("8-Gulosa")
            print("9-Custo Uniforme")
            print("10-Iterativa")
            print("11-Simular (Contigência)")
            print("12-Simular (Dinâmico)")
            print("13-Múltiplas Entregas")
            print("14-Guardar Grafo")
            print("15-Carregar Grafo")
            print("0-Saír")

            saida = int(input("introduza a sua opcao-> "))
            if saida == 0:
                print("saindo.......")
            elif saida == 1:
                print(g.m_graph)
                l = input("prima enter para continuar")
            elif saida == 2:
                g.desenha()
            elif saida == 3:
                print(g.m_graph.keys())
                print(len(g.m_graph.keys()))
                l = input("prima enter para continuar")
            elif saida == 4:
                print(g.imprime_aresta())
                l = input("prima enter para continuar")
            elif saida == 5:
                inicio = input("Nodo inicial->")
                fim = input("Nodo final->")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)->")
                peso_enc = int(input("Peso da encomenda (kg)->"))
                start_time = datetime.datetime.now()  # Get current date and time before the operation
                temp = (g.procura_DFS2(inicio, fim))
                end_time = datetime.datetime.now()
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                execution_time = end_time - start_time  # Calculate the time difference
                print(f"Execution Time: {execution_time}")
                l = input("prima enter para continuar")
            elif saida == 6:
                inicio = input("Nodo inicial->")
                fim = input("Nodo final->")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)->")
                peso_enc = int (input("Peso da encomenda (kg)->"))
                start_time = datetime.datetime.now()  # Get current date and time before the operation
                temp = (g.procura_BFS(inicio, fim))
                end_time = datetime.datetime.now()
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                execution_time = end_time - start_time  # Calculate the time difference
                print(f"Execution Time: {execution_time}")
                l = input("prima enter para continuar")
                l = input("prima enter para continuar")
            elif saida == 7:
                inicio = input("Nodo inicial->")
                fim = input("Nodo final->")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)->")
                peso_enc = int (input("Peso da encomenda (kg)->"))
                # atribuir heuristica
                g.all_heuristics(fim)
                start_time = datetime.datetime.now()  # Get current date and time before the operation
                temp = (g.procura_aStar(inicio, fim))
                end_time = datetime.datetime.now()
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                execution_time = end_time - start_time  # Calculate the time difference
                print(f"Execution Time (sem contar com o calculo da heuristica): {execution_time}")
                l = input("prima enter para continuar")
            elif saida == 8:
                inicio = input("Nodo inicial->")
                fim = input("Nodo final->")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)->")
                peso_enc = int (input("Peso da encomenda (kg)->"))
                # atribuir heuristica
                g.all_heuristics(fim)
                start_time = datetime.datetime.now()  # Get current date and time before the operation
                temp = (g.greedy(inicio, fim))
                end_time = datetime.datetime.now()
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                execution_time = end_time - start_time  # Calculate the time difference
                print(f"Execution Time (sem contar com o calculo da heuristica): {execution_time}")
                l = input("prima enter para continuar")
            elif saida == 9:
                inicio = input("Nodo inicial->")
                fim = input("Nodo final->")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)->")
                peso_enc = int (input("Peso da encomenda (kg)->"))
                start_time = datetime.datetime.now()  # Get current date and time before the operation
                temp = (g.procura_custo_uniforme(inicio, fim))
                end_time = datetime.datetime.now()
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                execution_time = end_time - start_time  # Calculate the time difference
                print(f"Execution Time: {execution_time}")
                l = input("prima enter para continuar")
            elif saida == 10:
                inicio = input("Nodo inicial->")
                fim = input("Nodo final->")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)->")
                peso_enc = int(input("Peso da encomenda (kg)->"))
                start_time = datetime.datetime.now()  # Get current date and time before the operation
                temp = (g.iterative_deepening_DFS(inicio, fim))
                end_time = datetime.datetime.now()
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                execution_time = end_time - start_time  # Calculate the time difference
                print(f"Execution Time: {execution_time}")
                l = input("prima enter para continuar")
            elif saida == 11:
                inicio = input("Nodo inicial-> ")
                fim = input("Nodo final-> ")
                es = float (input("Score do estafeta (0 a 5)->  "))
                alg = input("Procura (a*,gu,cunif,bfs,dfs,it)-> ")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)-> ")
                peso_enc = int(input("Peso da encomenda (kg)-> "))
                temp = g.simular_percurso(inicio,fim,es,alg)
                r = (
                    temp[0],round(g.calcula_custo(temp[0])), round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))  # km, minutos e gramas
                print("Planeado-> " + str(r))

                r2 = (
                    temp[1],round(g.calcula_custo(temp[1])), round(g.calcula_tempo(temp[1], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[1], veiculo, peso_enc))  # km, minutos e gramas
                print("Real-> " + str(r2))
                l = input("prima enter para continuar")
            elif saida == 12:
                inicio = input("Nodo inicial-> ")
                fim = input("Nodo final-> ")
                es = float (input("Score do estafeta (0 a 5)->  "))
                alg = input("Procura (a*,gu,cunif,bfs,dfs,it)-> ")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)-> ")
                peso_enc = int(input("Peso da encomenda (kg)-> "))
                temp = g.simular_percursoWaze(inicio,fim,es,veiculo,peso_enc,alg)
                print(temp)
                l = input("prima enter para continuar")
            elif saida == 13:
                inicio = input("Nodo inicial-> ")
                num_e = int(input("Numero de encomendas-> "))
                encomendas = []
                for i in range(num_e):
                    fim = input("Destino final-> ")
                    peso_enc = int(input("Peso da encomenda (kg)-> "))
                    encomendas.append((fim,peso_enc))

                alg = input("Procura (a*,gu,cunif,bfs,dfs,it)-> ")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)-> ")
                caminho = g.fazer_entrega(inicio,encomendas,alg,veiculo)
                print(caminho)
                l = input("prima enter para continuar")
            elif saida == 14:
                path = input("Insira o Path ")
                Grafo.save_graph(g, path)
                l = input("prima enter para continuar")

            elif saida == 15:
                path = input("Insira o Path ")
                g = Grafo.load_graph(path)
                #g.add_edge("Node_24_22", "Rua", (24, 22), "Nodo_24_22_1","Porta", (24, 22.5), 0.5, True, 0.215, 0.18)
                #g.add_edge("Node_24_22_1","Porta", (24, 22.5), "Node_24_23", "Rua", (24, 23), 0.5, True, 0.215, 0.18)
                #g.add_edge("Node_22_18", "Rua", (22, 18), "Node_22_18_1", "Porta", (22, 18.25), 0.5, True, 0.05, 0.17)
                #g.add_edge("Node_22_18_1", "Porta", (22, 18.25), "Node_22_18_2", "Porta", (22, 18.5), 0.5, True, 0.05, 0.17)
                #g.add_edge("Node_22_18_2", "Porta", (22, 18.5), "Node_22_18_3", "Porta", (22, 18.75), 0.5, True, 0.05, 0.17)
                #g.add_edge("Node_22_18_3", "Porta", (22, 18.75), "Node_22_19", "Rua", (22, 19), 0.5, True, 0.05, 0.17)
                l = input("prima enter para continuar")

            elif saida == 16:
                inicio = input("Nodo inicial->")
                fim = input("Nodo final->")
                veiculo = input("Tipo de veiculo (mota,carro,bicicleta)->")
                peso_enc = int(input("Peso da encomenda (kg)->"))

                print("\n\n5\n\n")
                temp = (g.procura_DFS2(inicio, fim))
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                print(f"Nodos expandidos: {len(temp[2])}")
                print(f"Nodos objetivo: {len(temp[0])}")

                print("\n\n6\n\n")
                temp = (g.procura_BFS(inicio, fim))
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                print(f"Nodos expandidos: {len(temp[2])}")
                print(f"Nodos objetivo: {len(temp[0])}")

                print("\n\n7\n\n")
                g.all_heuristics(fim)
                temp = (g.procura_aStar(inicio, fim))
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                print(f"Nodos expandidos: {len(temp[2])}")
                print(f"Nodos objetivo: {len(temp[0])}")

                print("\n\n8\n\n")
                g.all_heuristics(fim)
                temp = (g.greedy(inicio, fim))
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                print(f"Nodos expandidos: {len(temp[2])}")
                print(f"Nodos objetivo: {len(temp[0])}")

                print("\n\n9\n\n")
                temp = (g.procura_custo_uniforme(inicio, fim))
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                print(f"Nodos expandidos: {len(temp[2])}")
                print(f"Nodos objetivo: {len(temp[0])}")

                print("\n\n10\n\n")
                temp = (g.iterative_deepening_DFS(inicio, fim))
                r = (
                    temp[0], temp[1], round(g.calcula_tempo(temp[0], veiculo, peso_enc), 2),
                    g.calcula_pegada(temp[0], veiculo, peso_enc))
                print(r)
                print("km,horas,gramas/km")
                print(temp[2])
                print(f"Nodos expandidos: {len(temp[2])}")
                print(f"Nodos objetivo: {len(temp[0])}")
                l = input("prima enter para continuar")

            else:
                print("you didn't add anything")
                l = input("prima enter para continuar")


if __name__ == "__main__":
    main()