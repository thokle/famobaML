# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:36:36 2024

@author: hp
"""
import sys
import pandas as pd
from py2neo import Graph
from graphdatascience import GraphDataScience

""" Establish connection to the remote Neo4j database """
uri = "neo4j://65.108.80.255:7687"

graph = Graph(uri)
gds = GraphDataScience(uri)

""" Create the pipeline """
# Create Link Prediction Pipeline (driver)
pipe, _ = gds.beta.pipeline.linkPrediction.create("pipe2")

# Add Fast RP Embeddings (driver)
pipe.addNodeProperty("fastRP",
                     mutateProperty="embedding",
                     embeddingDimension=256,
                     iterationWeights=[0.8, 1, 1, 1],
                     normalizationStrength=0.5,
                     randomSeed=42)

for algo in ['pageRank', 'betweenness']:
    pipe.addNodeProperty(algo, mutateProperty=algo)

# Add features to pipeline (driver)
pipe.addFeature("hadamard", nodeProperties=['embedding', 'pageRank', 'betweenness'])

# Split Train/Test (driver)
pipe.configureSplit(trainFraction=0.3, testFraction=0.3, validationFolds=7)

# Create graph projection (driver)
node_projection = ["User", "Child", "Groups", "Tags"]
relationship_projection = {
    "UserIsInGroup": {"orientation": "UNDIRECTED"},
    "ChildBelongToParent": {"orientation": "UNDIRECTED"},
    "UserMatches": {"orientation": "UNDIRECTED"},
    "user_has_tags": {"orientation": "UNDIRECTED"}
}
# drop any existing graph with the same name before creating a new one
try:
    G.drop()
except Exception:
    pass

'''
the gds.graph.project() function is called to create the projected graph based on 
the specified node and relationship projections. The projected graph is assigned to G, 
and since we're not using the relationship types returned by gds.graph.project(), 
we ignore the second return value with _.
'''
G, _ = gds.graph.project("famoba", node_projection, relationship_projection)

# Train link prediction model (driver)
pipe.   addLogisticRegression()
model_name = "Famoba-pipeline-model"
trained_pipe_model, res = pipe.train(G, targetRelationshipType="UserIsInGroup", modelName=model_name)

# Stream Results (driver)
results = trained_pipe_model.predict_stream(G, topN=1000)
results_df = pd.DataFrame(results)


def get_id_prediction(user_id='', group_id='', result=results_df):
    ''' Predicts if a user is in a group using user Id and group Id. '''
    if user_id and group_id:
        final_table = result.loc[(result.node1 == user_id) & (result.node2 == group_id)]
        if final_table.size < 1:
            print("Relationship Not Significant!!!")
            return None
        else:
            table = final_table.reset_index(drop=True)
            return table

    elif user_id:
        final_table = result.loc[result.node1 == user_id]
        if final_table.size < 1:
            print("Relationship Not Significant!!!")
            return None
        else:
            table = final_table.reset_index(drop=True)
            return table

    elif group_id:
        final_table = result.loc[result.node2 == group_id]
        if final_table.size < 1:
            print("Relationship Not Significant!!!")
            return None
        else:
            table = final_table.reset_index(drop=True)
            return table
    else:
        print("User node ID attribute missing")


def get_username_prediction(user_email='', group_name='', result=results_df):
    '''
        Predicts probability of user being in a group using the username and group name.
    '''
    if user_email and group_name:
        query1 = f""" Match (n:User {{ email: '{user_email}'}}) 
            RETURN id(n)
            """
        query2 = f""" Match (n:Groups {{ name: '{group_name}'}}) 
            RETURN id(n)
            """
        Node1 = graph.run(query1).to_series()
        Node2 = graph.run(query2).to_series()
        final_table = result.loc[(result.node1.isin(Node1)) & (result.node2.isin(Node2))]
        if final_table.size < 1:
            print("Relationship Not Significant!!!")
            return None
        else:
            table = final_table.reset_index(drop=True)
            return table

    elif user_email and (not group_name):
        query = f""" Match (n:User {{ _email: '{user_email}'}}) 
            RETURN id(n)
            """
        Node1 = graph.run(query).to_series()
        final_table = result.loc[result.node1.isin(Node1)]
        if final_table.size < 1:
            print("Relationship Not Significant!!!")
            return None
        else:
            table = final_table.reset_index(drop=True)
            return table

    elif (not user_email) and group_name:
        query = f""" Match (n:Groups {{ name: '{group_name}'}}) 
            RETURN id(n) 
            """
        Node2 = graph.run(query).to_series()
        final_table = result.loc[result.node2.isin(Node2)]
        if final_table.size < 1:
            print("Relationship Not Significant!!!")
            return None
        else:
            table = final_table.reset_index(drop=True)
            return table
    else:
        print("User attribute missing")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
              Please provide user_email and group_name as shown below:\n\t
              Usage: python3 pipeline.py <user_email> <group_name>
              """)
    else:
        email = sys.argv[1]
        group = sys.argv[2]
        get_username_prediction(email, group)
    ### Close the database connection.
    gds.close()
