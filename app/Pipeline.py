# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:42:03 2024
@author: Thomas Kleist
"""

import logging
import pandas as pd
from graphdatascience.model.model import Model
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
            # Check if pipeline already exists
            exists_response = self.gds.pipeline.exists(self.pipe_name)
            logging.debug(f"Pipeline exists check response: {exists_response}")
            exists_flag = exists_response.get("exists", [False])[0] if isinstance(
                exists_response.get("exists", [False]), list) else exists_response.get("exists", False)

            if exists_flag:
                logging.info(f"Pipeline '{self.pipe_name}' exists. Dropping it...")
                existing_pipeline = self.gds.pipeline.get(self.pipe_name)
                self.gds.pipeline.drop(existing_pipeline)
            else:
                logging.info(f"No pipeline exists for '{self.pipe_name}'. Creating a new one...")

            # Create the pipeline
            try:
                self.pipe, _ = self.gds.beta.pipeline.linkPrediction.create(name=self.pipe_name)
                logging.info("Pipeline created successfully.")
            except Exception as e:
                logging.error(f"Pipeline creation failed: {e}")
                self.pipe = None
                return

            if not isinstance(self.pipe, TrainingPipeline):
                logging.error("Pipeline creation failed; `self.pipe` is invalid.")
                return

            # Add node properties and features
            logging.info("Adding node properties to the pipeline...")
            self.pipe.addNodeProperty(
                "fastRP",
                mutateProperty="embedding",
                embeddingDimension=256,
                iterationWeights=[0.8, 1, 1, 1],
                normalizationStrength=0.5,
                randomSeed=42
            )
            logging.info("Added fastRP property.")

            for algo in ["pageRank", "betweenness"]:
                self.pipe.addNodeProperty(algo, mutateProperty=algo)
                logging.info(f"Added {algo} property.")

            # Add features
            logging.info("Adding features to the pipeline...")
            self.pipe.addFeature(
                "hadamard",
                nodeProperties=["embedding", "pageRank", "betweenness"]
            )
            logging.info("Added Hadamard features.")

            # Configure train/test split
            logging.info("Configuring train/test split...")
            self.pipe.configureSplit(
                trainFraction=0.3,
                testFraction=0.3,
                validationFolds=7
            )
            logging.info("Pipeline features added and split configured.")

        except Exception as e:
            logging.error(f"Error during pipeline creation: {e}")

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
            if self.gds.graph.exists(self.graph_name).iloc[1]:
                self.gds.graph.drop(self.graph_name)

            # Project the graph
            self.G, _ = self.gds.graph.project(self.graph_name, node_projection, relationship_projection)
            logging.info(f"Graph '{self.graph_name}' projected successfully.")

            # Add Logistic Regression
            self.pipe.addLogisticRegression(penalty=(0.1, 2))
            logging.info("Added Logistic Regression to pipeline.")

            # Train the model
            model_name = "Famoba-pipeline-model"
            model_exists = self.gds.model.exists(model_name).iloc[2]
            models = self.gds.model.list()
            print(models)
            if model_exists:
                self.gds.model.drop(models[0])

            self.model, _ = self.pipe.train(self.G, targetRelationshipType="UserIsInGroup", modelName=model_name)
            logging.info("Model created and trained successfully.")

            if not isinstance(self.model, Model):
                logging.error(f"Expected a 'Model' type, got {type(self.model)} instead.")
                return None

            return self.model

        except Exception as e:
            logging.error(f"Error during model creation: {e}")
            return None


        except AttributeError as attr_err:
            logging.error(f"Attribute error during model training: {attr_err}")
        except Exception as e:
            logging.error(f"Error creating model: {e}")
            return None

    def get_result(self):
        """Stream and retrieve prediction results."""
        if self.model:
            try:
                self.result = pd.DataFrame(self.model.predict_stream(self.G, topN=1000))
                logging.info("Result fetched and converted to DataFrame.")
            except Exception as e:
                logging.error(f"Error fetching results: {e}")
        else:
            logging.warning("Model is not available; cannot fetch results.")

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
