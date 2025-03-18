# TextToGraph_bootstrap
This repository contains the implementation of a novel two-phase unlearnable approach designed to convert text into its semantic graph representation. The method effectively addresses the challenges posed by existing text-to-graph conversion tools, particularly when handling long or complex texts, which often result in fragmented representations.
## Overview  
This repository provides implementations of **text-to-graph conversion methods** using:  
- **Semantic Role Labeling (SRL)** via the **Senna framework**  
- **Resource Description Framework (RDF)** via **FRED**  
- **Bootstrapping techniques** applied in both **SRL** and **RDF**  

The project utilizes two datasets:  
- **Automated Student Assessment Prize (ASAP) dataset** (for essay scoring and assessment)  
- **COVID-19 Fake News dataset** (for misinformation detection)  

## Features  
✅ **SRL-Based Graph Construction** using the Senna framework  
✅ **RDF-Based Graph Construction** using FRED for semantic knowledge representation  

## File Structure  
📂 text-to-graph
├──📂srl_text_to_graph_conversion_bootstrpe
  ├── srl_text_to_graph.py # SRL-based graph conversion using Senna
  ├── in.txt # to add the testing sentences
  ├── out.txt # the SRL tree will be saved in this file
├── 📂srl_text_to_graph_conversion_bootstrpe
  ├── rdf_text_to_graph.py # RDF-based graph conversion using FRED
  ├── in.txt # to add the testing sentences
  ├── out.txt # the RDF triples will be saved in this file
├── 📂datasets/ # Folder containing ASAP and COVID-19 datasets
├── README.md # Project documentation


## Installation & Dependencies  
### 
1. Install Python Dependencies**  
Ensure you have Python 3.8+ installed. Then, install the required libraries:  
```bash
pip install requests rdflib spacy networkx pandas

2. Set Up Senna for SRL
Download the Senna toolkit and configure it in your project directory.
https://ronan.collobert.com/senna/

3. Access FRED API for RDF Conversion
To use FRED, obtain an API key from FRED.
http://wit.istc.cnr.it/stlab-tools/fred/demo/

Usage
Run the SRL-based conversion + bootstrapping :
in.txt <-- "The study found that online learning improved student performance."
python srl_text_to_graph.py 

Run the RDF-based conversion + bootstrapping:
in.txt <-- "The study found that online learning improved student performance."
python rdf_text_to_graph.py

for testing sentences fill file in.txt with the required sentences


ASAP dataset
dataset datasets/ASAP.csv

COVID19_FakeNews dataset:
datasets/COVID19_FakeNews.csv

