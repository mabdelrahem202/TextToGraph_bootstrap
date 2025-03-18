import os
import spacy
import nltk
import numpy
import networkx as nx
import matplotlib.pyplot as plt
import scipy 
import numpy as np
import IPython
from pathlib import Path
from networkx.readwrite import json_graph
import json
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd
import time

start_timer = time.time()


nlp=spacy.load("en_core_web_sm")
doc=nlp("The Cat chased the Mouse")


nltk.Senna
import subprocess
myinput = open('in.txt',"r")
myoutput = open('out.txt', 'w')
p = subprocess.Popen('senna-win32.exe -srl', stdin=myinput, stdout=myoutput )

p.wait()
myoutput.flush()
srlout = []
with open('out.txt', 'r') as f2:
    data = f2.read()
    print(data)
    lines = data.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        tokens = line.split("\t")
        srlout.append([])
        for token in tokens:
            srlout[-1].append(token.strip())

myinput = open('in.txt',"r")
fulltext=myinput.read()
print(fulltext)
number_of_words=len(fulltext.split())

srltags = []
tokens = []
for token in srlout:
    tokens.append(token[0])

for layerid in range(2, len(srlout[0])):
    srltags.append([])
    sweeping = False
    item = ""
    for tokenid in range(0, len(srlout)):
        tag = srlout[tokenid][layerid]
        tags = tag.split("-")
        role = tags[-1]
        if role[0] == 'A' or role[0] == 'V':
            if sweeping:
                if len(tags) < 2:
                    print("Something is wrong in the analysis: " + tag + " is between an B and E")
                elif tags[-2] == 'I':
                    item += " " + tokens[tokenid]
                elif tags[-2] == 'E':
                    item += " " + tokens[tokenid]
                    sweeping = False
                    srltags[-1].append(item)
                    item = ""
            else:
                if len(tags) == 1 or tags[-2] == 'S':
                    srltags[-1].append(tags[-1] + ": " + tokens[tokenid])
                elif tags[-2] == 'B':
                    item = tags[-1] + ": " + tokens[tokenid]
                    sweeping = True

for verb in srltags:
    print(verb)

def connect_subgraphs(subgraphs):
    connected_subgraphs = []
    new_edges= []
    last_node=""
    first_node=""
    subgraphs = [list(subgraph) for subgraph in subgraphs]  # Convert sets to lists
    
    for i in range(len(subgraphs) - 1):
         current_subgraph = subgraphs[i]
         next_subgraph = subgraphs[i + 1]
         first_node=next_subgraph[0]
         last_node=current_subgraph[-1]
         new_directed_graph.add_edge(first_node,last_node,color='r')
         #for x in range(len(current_subgraph) - 1):
         connected_subgraphs.append((first_node ,last_node))
    print(connect_subgraphs)
    return connected_subgraphs         
  
   

def compute_graph_attributes(my_graph):
    # Compute centrality measures
    degree_centralities = nx.degree_centrality(my_graph)
    betweenness_centralities = nx.betweenness_centrality(my_graph)
    closeness_centralities = nx.closeness_centrality(my_graph)
    eigenvector_centralities = nx.eigenvector_centrality_numpy(my_graph)
    eccentricities = nx.eccentricity(new_directed_graph.to_undirected())
    radius = nx.radius(new_directed_graph.to_undirected())

    # Compute diameter and radius
    diameter = nx.diameter(new_directed_graph.to_undirected())
    my_nodes=[]

    #number_of_nodes=len(list(my_graph.nodes))
    #number_of_edges=len(list(my_graph.edges))
    # Calculate degree centrality
    centrality = nx.degree_centrality(my_graph)

# Print the degree centrality for each node
    for node, centrality_value in centrality.items():
        print(f"Node {node}: {centrality_value}")
        my_nodes.append(node)
        #df = pd.DataFrame({"Node {node}: {centrality_value}"})

    number_of_nodes=len(list(my_graph.nodes))
    number_of_edges=len(list(my_graph.edges))
    # Create a DataFrame
    data = {
    'Node': list(my_graph.nodes()),
    'Degree Centrality': list(degree_centralities.values()),
    'Betweenness Centrality': list(betweenness_centralities.values()),
    'Closeness Centrality': list(closeness_centralities.values()),
    'Eigenvector Centrality': list(eigenvector_centralities.values()),
    'Eccentricities': list(eccentricities.values())
    }
    df = pd.DataFrame(data)

    # Add diameter and radius to the DataFrame
    df.loc[len(df)] = ['Diameter','', '', '', '', diameter]
    df.loc[len(df)] = ['Radius', '', '', '', '', radius]
    df.loc[len(df)] = ['number of words','', '', '', '', number_of_words]
    df.loc[len(df)] = ['number of nodes','', '', '', '', number_of_nodes]
    df.loc[len(df)] = ['number of edges','', '', '', '', number_of_edges]

    #df = pd.DataFrame({'node':my_nodes, 'Degree Centralities': degree_centralities})
    #df = pd.DataFrame({'node':my_nodes,'Degree Centralities': degree_centralities, 'Betweenness Centralities': betweenness_centralities, 'Closeness Centralities' : closeness_centralities, 'Eigenvector Centralities' :eigenvector_centralities ,'Diameter' : diameter,'Radius' : radius,'eccentricities' :eccentricities,'number of nodes' :number_of_nodes,'number of edges' :number_of_edges })
    
    df.to_csv('SRL_Results.csv', sep=',', index=False, encoding='utf-8')
    print("Betweenness Centralities:", betweenness_centralities)
    print("Closeness Centralities:", closeness_centralities)
    print("Eigenvector Centralities:", eigenvector_centralities)
    print("Diameter:", diameter)
    print("Radius:", radius)
    print("eccentricities:", eccentricities)
    print("Number of words:", number_of_words)
    print("Number of nodes:", number_of_nodes)
    print("Number of edges:", number_of_edges)
    print(df)
    



# draw the SRL graph
    
g=nx.Graph()
tmp_verb=""
tmp=0
root_index=0
SRLtagsdict = {"Word":[],"arcs":[]};
i=0
roots= list()
mynodes= list()
for x in range(0, srltags.__len__(), 1):
    root=""
    tmp_root=0
    a0=""
    a1=""
    a2=""
    source=""
    destination=""
#{"start": 0, "end": 1, "label": "nsubj", "dir": "left"},
    for arg in srltags[x]:
        mynodes.append(str(arg))
        if(str(arg).startswith("V:")):
            root=str(arg)
            roots.append(root)
            case = {'text': str(arg).removeprefix("v:"), 'tag': "V"}
            source=str(i)
            print(str(root))
            tmp_root=1
        if(str(arg).startswith("A0:")):
            a0=str(arg)
            case = {'text': str(arg).removeprefix("A0:"), 'tag': "A0"}
            destination=str(i)
            print(str(a0))
        if(str(arg).startswith("A1:")):
            a1=str(arg)
            case = {'text': str(arg).removeprefix("a1:"), 'tag': "A1"}
            destination=str(i)
            print(str(a1))
        if(str(arg).startswith("A2:")):
            a2=str(arg)
            case = {'text': str(arg).removeprefix("a2:"), 'tag': "A2"}
            destination=str(i)
            print(str(a2))
        
        SRLtagsdict["Word"].append(case)
        if(root==0):
            case1={"end": destination, "dir": "left"}
        
        #case1.update({'start': source})
        if(tmp_root==0):
            case1={"end": destination, "dir": "left"}
        
        i=i+1
        try: 
            SRLtagsdict["arcs"].append(case1)
        except ValueError:
            print(f"No edges found")
        
    # draw the edges

    if a0 !="" :
        token1=nlp(root)
        token2= nlp(a0)
        #g.add_edge(root,a0,weight=round(token1.similarity(token2),2),label="subject")
        g.add_edge(root,a0)
        print("weight of edge between :"+token1.text + " and "+ token2.text + " = "+ str(token1.similarity(token2)) )
        SRLtagsdict["Word"].append(case)
    if a1 !="" :   
        token1=nlp(root)
        token2= nlp(a1)
        #g.add_edge(root,a1,weight=round(token1.similarity(token2),2),label="object")
        g.add_edge(root,a1)
        print("weight of edge between :"+token1.text + " and "+ token2.text + " = "+ str(token1.similarity(token2)) )
    if a2 !="" :  
        token1=nlp(root)
        token2= nlp(a2)
        #g.add_edge(root,a2,weight=round(token1.similarity(token2),2))
        g.add_edge(root,a2)
        print("weight of edge between :"+token1.text + " and "+ token2.text + " = "+ str(token1.similarity(token2)) )
    
   
print(SRLtagsdict)     

pos=nx.spring_layout( g )
labels = nx.get_edge_attributes(g,'weight')
node_labels=nx.get_node_attributes(g,'node_link_data')
#nx.draw_kamada_kawai(g,with_labels=True)
nx.draw_spring(g,with_labels=True)
#nx.draw_networkx_labels(g, pos, font_size=12, font_family="sans-serif")
#nx.draw_networkx_edge_labels(g,pos)

end_timer_SRL = time.time()
print("Elapsed Time for drawing SRL "+str(end_timer_SRL - start_timer))


# Identify connected components
connected_components = list(nx.weakly_connected_components(g.to_directed()))
    # Create a layout for each subgraph
pos = {}
for i, component in enumerate(connected_components):
    subgraph = g.subgraph(component)
    sub_pos = nx.spring_layout(subgraph, seed=42)
    
    # Offset each subgraph to avoid overlap
    offset_x = i * 2  # Change the multiplier to adjust spacing between subgraphs
    for node in sub_pos:
        sub_pos[node][0] += offset_x
    
    pos.update(sub_pos)

# Check if positions are correctly assigned
print("Node positions:", pos)

# Draw the graph with separated subgraphs
plt.figure(figsize=(12, 8))
try:
    nx.draw(g, pos, with_labels=True, node_size=700, font_size=10, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(g, 'label')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)
except Exception as e:
    print("An error occurred while drawing the graph:", e)




if g.is_multigraph:
    print("the graph is multigraph")
plt.show()
start_timer=time.time()
new_directed_graph=g
new_directed_graph=new_directed_graph.to_directed()





#list of all weakly connected subgraphs, each one is a set. 
is_weakly_connected =nx.is_weakly_connected(new_directed_graph)
print("the graph weakly connected is :"+str(is_weakly_connected))
list_of_subgraphs = list(nx.weakly_connected_components(new_directed_graph))

# list and print the subgraphs
list_of_digraphs = []
for subgraph in list_of_subgraphs:
    list_of_digraphs.append(nx.subgraph(new_directed_graph, subgraph)) #this is the relevant part.    
    print(subgraph)

# Bootstrap the subgraphs to have 1 graph
connected_subgraphs=connect_subgraphs(list_of_subgraphs)

# Draw the graph
pos=nx.kamada_kawai_layout( new_directed_graph )
labels = nx.get_edge_attributes(new_directed_graph,'weight')
nx.draw_kamada_kawai(new_directed_graph,with_labels=True)
#nx.draw_networkx_edge_labels(new_directed_graph, pos)

end_timer_SRL_bootstrapped = time.time()
print("Elapsed Time for bootstraping SRL "+str(end_timer_SRL_bootstrapped- start_timer ))

# show the graph
plt.show()
print(nx.is_weakly_connected(new_directed_graph))

for c in sorted(nx.weakly_connected_components(new_directed_graph), key=len, reverse=True):
    print(c)

# calculate centrailty , eccentricity and diameter
compute_graph_attributes(new_directed_graph)

## creating interactive graph 
# Create a network visualization
nt = Network(notebook=True)

# Add nodes and edges to the visualization
for node in new_directed_graph.nodes:
    if(node.startswith('V')):
        #nt.add_node(node,color='red')
        nt.add_node(node)
    else:
        nt.add_node(node)

for edge in new_directed_graph.edges:
    nt.add_edge(edge[0], edge[1])

# Display the interactive graph
nt.show('graph_SRL.html')

# drawing the spanning tree
G = nx.Graph()
G.add_edges_from(g.edges)
'''
G.add_edges_from(
    [
        ("V: chased", "A0: the cat", {"weight": 0.48}),
        ("V: chased", "A1: the mouse", {"weight": 0.49}),
        ("V: ate", "A0: the cat", {"weight": 0.60}),
        ("V: ate", "A1: him", {"weight": 0.60})        
    ]
)
'''
# Find the minimum spanning tree
T = nx.minimum_spanning_tree(G)

# Visualize the graph and the minimum spanning tree
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
nx.draw_networkx_edges(G, pos, edge_color="grey")
nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
#nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)})
nx.draw_networkx_edges(T, pos, edge_color="green", width=2)
plt.axis("off")
plt.show()




















