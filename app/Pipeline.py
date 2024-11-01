# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:42:03 2024
@author: Thomas Kleist
"""

import logging
import pandas as pd
from graphdatascience.pipeline.training_pipeline import TrainingPipeline
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

logging.basicConfig(level=logging.INFO)

class Pipe:
    def __init__(self, uri="bolt://65.108.80.255:7687", username="neo4j", password="famoba2024",
                 graph_name='famoba', pipe_name='pipe2'):
        """Initialize Neo4j and Graph Data Science connections with authentication."""
        self.uri = uri
        self.graph_name = graph_name
        self.pipe_name = pipe_name
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.gds = GraphDataScience(uri, auth=(username, password))

        self.G = None
        self.pipe = None
        self.model = None
        self.result = None

    def create_pipeline(self):
        """Create or reset the link prediction pipeline."""
        try:
            # Check if the pipeline already exists and drop it if it does
            if self.gds.pipeline.exists(self.pipe_name)[2]:

                pipe  = self.gds.pipeline.get(self.pipe_name)
                self.gds.pipeline.drop(pipe)
            else:
            # Create a new pipeline
                self.pipe, _ = self.gds.beta.pipeline.linkPrediction.create(self.pipe_name)

            if self.pipe is None:
                logging.error("Failed to create pipeline; `self.pipe` is None.")
                return

            # Add node properties and features if pipeline creation was successful
            self.pipe.addNodeProperty("fastRP", mutateProperty="embedding", embeddingDimension=256,
                                      iterationWeights=[0.8, 1, 1, 1], normalizationStrength=0.5, randomSeed=42)
            for algo in ['pageRank', 'betweenness']:
                self.pipe.addNodeProperty(algo, mutateProperty=algo)
            self.pipe.addFeature("hadamard", nodeProperties=['embedding', 'pageRank', 'betweenness'])

            # Configure train/test split
            self.pipe.configureSplit(trainFraction=0.3, testFraction=0.3, validationFolds=7)
            logging.info("Pipeline created and configured.")
        except Exception as e:
            logging.error(f"Error creating pipeline: {e}")

    def _configure_pipeline(self):
        """Add properties and features to the pipeline."""
        self.pipe.addNodeProperty("fastRP", mutateProperty="embedding", embeddingDimension=256,
                                  iterationWeights=[0.8, 1, 1, 1], normalizationStrength=0.5, randomSeed=42)
        for algo in ['pageRank', 'betweenness']:
            self.pipe.addNodeProperty(algo, mutateProperty=algo)
        self.pipe.addFeature("hadamard", nodeProperties=['embedding', 'pageRank', 'betweenness'])
        self.pipe.configureSplit(trainFraction=0.3, testFraction=0.3, validationFolds=7)

    def create_model(self):
        """Project the graph, create, and train the model."""
        if self.pipe is None:
            logging.error("Pipeline creation failed; cannot proceed with model creation.")
            return None

        node_projection = ["User", "Child", "Groups", "Tags"]
        relationship_projection = {
            "UserIsInGroup": {"orientation": "UNDIRECTED"},
            "ChildBelongToParent": {"orientation": "UNDIRECTED"},
            "UserMatches": {"orientation": "UNDIRECTED"},
            "userhastags": {"orientation": "UNDIRECTED"}
        }

        try:
            # Check if the graph already exists and drop it if it does
            if self.gds.graph.exists(self.graph_name)[1]:
                self.gds.graph.drop(self.graph_name)

            # Project the graph and add model configurations
            self.G, _ = self.gds.graph.project(self.graph_name, node_projection, relationship_projection)
            self.pipe.addLogisticRegression(penalty=(0.1, 2))

            # Train the model and check if model creation succeeds
            model_name = "Famoba-pipeline-model"
            if self.gds.model.exists(model_name)[2]:
                self.gds.model.drop(model_name)

            self.model, _ = self.pipe.train(self.G, targetRelationshipType="UserIsInGroup", modelName=model_name)
            logging.info("Model created and trained.")
            return self.model
        except Exception as e:
            logging.error(f"Error creating model: {e}")
            return None

    def _train_model(self):
        """Train the logistic regression model on the projected graph."""
        model_name = "Famoba-pipeline-model"
        if self.gds.model.exists(model_name)[2]:
            self.gds.model.drop(model_name)

        self.pipe.addLogisticRegression(penalty=(0.1, 2))
        self.model, _ = self.pipe.train(self.G, targetRelationshipType="UserIsInGroup", modelName=model_name)

    def get_result(self):
        """Stream and retrieve prediction results."""
        if self.model:
            self.result = pd.DataFrame(self.model.predict_stream(self.G, topN=1000))
            logging.info("Result fetched and converted to DataFrame.")
        else:
            logging.warning("Model is not available; cannot fetch results.")

    def get_prediction(self, user_id=None, group_id=None):
        """Predict if a user is in a group based on provided IDs."""
        if not self.result.empty:
            # Create an empty filter condition
            filter_condition = pd.Series([True] * len(self.result))

            # Apply conditions based on user_id and group_id if provided
            if user_id is not None:
                filter_condition &= (self.result.node1 == user_id)
            if group_id is not None:
                filter_condition &= (self.result.node2 == group_id)

            final_table = self.result[filter_condition]
            return final_table.reset_index(drop=True) if not final_table.empty else None
        else:
            logging.warning("Result DataFrame is empty.")
            return None

    def get_entity_id(self, label, property_name, value):
        """Retrieve Neo4j node ID by label and property."""
        query = f"MATCH (n:{label} {{{property_name}: '{value}'}}) RETURN id(n)"
        try:
            with self.driver.session() as session:
                return session.run(query).single().value()
        except Exception as e:
            logging.error(f"Error fetching entity ID: {e}")
            return None

    def get_username_prediction(self, user_email=None, group_name=None):
        """Predict probability of user being in a group using username and group name."""
        if not self.result.empty:
            user_id = self.get_entity_id("User", "_email", user_email) if user_email else None
            group_id = self.get_entity_id("Groups", "name", group_name) if group_name else None
            return self.get_prediction(user_id=user_id, group_id=group_id)
        else:
            logging.warning("Result DataFrame is empty.")
            return None

    def close_connection(self):
        """Close Neo4j and GDS connections."""
        try:
            self.gds.close()
            self.driver.close()
            logging.info("Connections closed successfully.")
        except Exception as e:
            logging.error(f"Error closing connection: {e}")

if __name__ == "__main__":
    famoba_pipe = Pipe()
    famoba_pipe.create_pipeline()
    famoba_pipe.create_model()
    famoba_pipe.get_result()
    famoba_pipe.close_connection()
