# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:42:03 2024

@author: hp
"""

import sys
import pandas as pd
from py2neo import Graph
from graphdatascience import GraphDataScience


class Pipe(object):
    def __init__(self, uri="neo4j://65.108.80.255:7687", graph_name='famoba', pipe_name='pipe2'):
        """ Establish connection to the remote Neo4j database """
        self.uri = uri
        self.graph = Graph(uri)
        self.gds = GraphDataScience(uri)
        self.pipe_name = pipe_name
        self.graph_name = graph_name
        self.G = None
        self.pipe = None
        self.model = None
        self.result = None

    def create_pipeline(self):
        """ Create the pipeline """
        # Create Link Prediction Pipeline (driver)
        pipeline_exists = self.gds.pipeline.exists('pipe2')[2]
        if pipeline_exists:
            pipe = self.gds.pipeline.get('pipe2')

            self.gds.beta.pipeline.drop(pipe)

        self.pipe, _ = self.gds.beta.pipeline.linkPrediction.create(self.pipe_name)

        self.pipe.addNodeProperty("fastRP",
                                  mutateProperty="embedding",
                                  embeddingDimension=256,
                                  iterationWeights=[0.8, 1, 1, 1],
                                  normalizationStrength=0.5,
                                  randomSeed=42)

        for algo in ['pageRank', 'betweenness']:
            self.pipe.addNodeProperty(algo, mutateProperty=algo)

        # Add features to pipeline (driver)
        self.pipe.addFeature("hadamard", nodeProperties=['embedding', 'pageRank', 'betweenness'])
        # Split Train/Test (driver)
        self.pipe.configureSplit(trainFraction=0.3, testFraction=0.3, validationFolds=7)

    def create_model(self):
        # Create graph projection (driver)
        node_projection = ["User", "Child", "Groups"]
        relationship_projection = {
            "UserIsInGroup": {"orientation": "UNDIRECTED"},
            "ChildBelongToParent": {"orientation": "UNDIRECTED"},
            "UserMatches": {"orientation": "UNDIRECTED"}
        }
        # drop any existing graph with the same name before creating a new one
        try:
            if self.gds.graph.exists(self.graph_name)[1]:
                self.gds.graph.drop(self.graph_name)
        except Exception:
            pass

        '''
        the gds.graph.project() function is called to create the projected graph based on 
        the specified node and relationship projections. The projected graph is assigned to G, 
        and since we're not using the relationship types returned by gds.graph.project(), 
        we ignore the second return value with _.
        '''
        self.G, _ = self.gds.graph.project(self.graph_name, node_projection, relationship_projection)

        # Train link prediction model (driver)
        self.pipe.addLogisticRegression()
        model_name = "Famoba-pipeline-model"

        if self.gds.model.exists(model_name)[2]:
            model = self.gds.model.get(model_name)
            self.gds.model.drop(model)

        self.model, res = self.pipe.train(self.G, targetRelationshipType="UserIsInGroup", modelName=model_name)

        return self.model

    def get_result(self):
        # Stream Results (driver)
        self.result = pd.DataFrame(self.model.predict_stream(self.G, topN=1000))

    def get_id_prediction(self, user_id='', group_id=''):
        ''' Predicts if a user is in a group using user Id and group Id. '''
        if user_id and group_id:
            final_table = self.result.loc[(self.result.node1 == user_id) & (self.result.node2 == group_id)]
            if final_table.size < 1:
                print("Relationship Not Significant!!!")
                return None
            else:
                table = final_table.reset_index(drop=True)
                return table

        elif user_id:
            final_table = self.result.loc[self.result.node1 == user_id]
            if final_table.size < 1:
                print("Relationship Not Significant!!!")
                return None
            else:
                table = final_table.reset_index(drop=True)
                return table

        elif group_id:
            final_table = self.result.loc[self.result.node2 == group_id]
            if final_table.size < 1:
                print("Relationship Not Significant!!!")
                return None
            else:
                table = final_table.reset_index(drop=True)
                return table
        else:
            print("User node ID attribute missing")

    def get_username_prediction(self, user_email='', group_name=''):
        '''
            Predicts probability of user being in a group using the username and group name.
        '''
        if user_email and group_name:
            query1 = f""" Match (n:User {{ _email: '{user_email}'}}) 
                RETURN id(n)
                """
            query2 = f""" Match (n:Groups {{ name: '{group_name}'}}) 
                RETURN id(n)
                """
            Node1 = self.graph.run(query1).to_series()
            Node2 = self.graph.run(query2).to_series()
            final_table = self.result.loc[(self.result.node1.isin(Node1)) & (self.result.node2.isin(Node2))]
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
            Node1 = self.graph.run(query).to_series()
            final_table = self.result.loc[self.result.node1.isin(Node1)]
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
            Node2 = self.graph.run(query).to_series()
            final_table = self.result.loc[self.result.node2.isin(Node2)]
            if final_table.size < 1:
                print("Relationship Not Significant!!!")
                return None
            else:
                table = final_table.reset_index(drop=True)
                return table
        else:
            print("User attribute missing")

    def close_connection(self):
        self.gds.close()


if __name__ == "__main__":
    famoba_pipe = Pipe()
    famoba_pipe.create_pipeline()
    famoba_pipe.create_model()
    # famoba_pipe.get_username_prediction(email, group)
    ### Close the database connection.
    famoba_pipe.close_connection()
