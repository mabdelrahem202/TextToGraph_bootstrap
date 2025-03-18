import spacy
from spacy.symbols import nsubj, VERB, dobj
import spacy
import nltk
#from rdflib import Graph, URIRef, Literal, RDF
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd
import time

start_timer = time.time()




nlp = spacy.load('en_core_web_sm')
list_of_Nodes = []
nx_graph=nx.DiGraph()
connected_subgraphs=nx.DiGraph()
def extract_rdf_constituents(text):
    doc = nlp(text)
    for sent in doc.sents:
        # Reset the subject and verb for each sentence
        subject = None
        verb = None

        for token in sent:
            if token.dep == nsubj:  # Identify the subject
                subject = token.text
            elif token.pos == VERB:  # Identify the verb
                verb = token.lemma_
            elif token.dep == dobj:  # Identify the direct object
                if subject and verb:
                    print(f"Subject: {subject}, Verb: {verb}, Object: {token.text}")
                    
                    nx_graph.add_edge(subject,token.text,label=verb)
                    
                    #graph.add((subject, URIRef('has_noun'), Literal(np.text.lower())))                    

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
    
    def connect(self, other_node):
        self.next = other_node

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
        nx_graph.add_edge(first_node,last_node)
        #for x in range(len(current_subgraph) - 1):
        connected_subgraphs.append((first_node ,last_node))
   print(connect_subgraphs)
   return connected_subgraphs 
    
def compute_graph_attributes(my_graph):
    # Compute centrality measures
    degree_centralities = nx.degree_centrality(my_graph)
    betweenness_centralities = nx.betweenness_centrality(my_graph)
    closeness_centralities = nx.closeness_centrality(my_graph)
    #eigenvector_centralities = nx.eigenvector_centrality(my_graph)
    eigenvector_centralities=nx.betweenness_centrality(my_graph)
    eccentricities = nx.eccentricity(nx_graph.to_undirected())
    radius = nx.radius(nx_graph.to_undirected())
    

    # Compute diameter and radius
    diameter = nx.diameter(nx_graph.to_undirected())
    my_nodes=[]
    
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
    'Eccentricities': list(eccentricities.values()),
    'Diameter' : diameter,
    'Radius' : radius
    }
    df = pd.DataFrame(data)

    # Add diameter and radius to the DataFrame
    '''
    df.loc[len(df)] = ['Diameter', diameter]
    
    df.loc[len(df)] = ['Radius',  radius]
    df.loc[len(df)] = ['Numer_of_words', number_of_words]
    df.loc[len(df)] = ['Numer_of_nodes',number_of_nodes]
    df.loc[len(df)] = ['Numer_of_edges',number_of_edges]
    '''
    

    #df = pd.DataFrame({'node':my_nodes, 'Degree Centralities': degree_centralities})
    #df = pd.DataFrame({'Degree Centralities': degree_centralities, 'Betweenness Centralities': betweenness_centralities, 'Closeness Centralities' : closeness_centralities, 'Eigenvector Centralities' :eigenvector_centralities ,'Diameter' : diameter,'Radius' : radius,'eccentricities' :eccentricities })
    df.to_csv('RDF_Results.csv', sep=',', index=False, encoding='utf-8')
    print("Betweenness Centralities:", betweenness_centralities)
    print("Closeness Centralities:", closeness_centralities)
    print("Eigenvector Centralities:", eigenvector_centralities)
    print("Diameter:", diameter)
    print("Radius:", radius)
    print("Number of words:", number_of_words)
    print("Number of nodes:", number_of_nodes)
    print("Number of edges:", number_of_edges)
    print("eccentricities:", eccentricities)
    print(df)
# Example usage
with open('in.txt') as f:
    contents = f.read()
    print(contents)


input_text = contents
number_of_words=len(input_text.split())


#"john plays football with ahmed , and ahmed ate the sandwitch alone"
#"Mohamed plays football with ahmed, ahmed is his friend who visited him "
#input_text = "Mohamed plays football with Ahmed ,and his friends visited him ,and the football consider boring sport."
extract_rdf_constituents(input_text)

#write attempt
'''
file = open("RDF_output.txt", mode="w")
file.write(nx_graph.serialize(format='turtle'))
print(nx_graph.serialize(format='turtle'))'''

# Draw the NetworkX graph
pos = nx.circular_layout(nx_graph)
nx.draw(nx_graph, pos, with_labels=True, node_size=1500, font_size=9, edge_color='black', arrows=True)
nx.draw_networkx_edge_labels(nx_graph, pos)

end_timer_SRL = time.time()
print("Elapsed Time for drawing RDF "+str(end_timer_SRL - start_timer))


plt.show()
#edge_labels = {(n1, n2): d['label'] for n1, n2, d in nx_graph.edges(data=True)}
start_timer=time.time()

#list of all weakly connected subgraphs, each one is a set. 
is_weakly_connected =nx.is_weakly_connected(nx_graph)
print("the graph strongly connected is :"+str(is_weakly_connected))
list_of_subgraphs = list(nx.weakly_connected_components(nx_graph))


# list and print the subgraphs
list_of_digraphs = []
for subgraph in list_of_subgraphs:
    list_of_digraphs.append(nx.subgraph(nx_graph, subgraph)) #this is the relevant part.    
    print(subgraph)

# Bootstrap the subgraphs to have 1 graph
connected_subgraphs=connect_subgraphs(list_of_subgraphs)

# Draw the graph
nx.draw_circular(nx_graph)
nx.draw_networkx_edge_labels(nx_graph, pos)

end_timer_rdf_bootstrapped = time.time()
print("Elapsed Time for bootstraping RDF "+str(end_timer_rdf_bootstrapped- start_timer ))
# show the graph
plt.show()

## creating interactive graph 
# Create a network visualization
nt = Network(notebook=True)

# Add nodes and edges to the visualization
for node in nx_graph.nodes:
    if(node.startswith('V')):
        nt.add_node(node,color='red')
    else:
        nt.add_node(node)

for edge in nx_graph.edges:
    nt.add_edge(edge[0], edge[1])

compute_graph_attributes(nx_graph)
# Display the interactive graph
nt.show('graph_RDF.html')
