# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:37:54 2024
@author: Thomas Kleist
"""
import json

import pandas as pd
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import logging

logging.basicConfig(level=logging.INFO)

class Neo4jRecommendationSystem:
    def __init__(self, uri, username, password):
        """Initialize connection to the Neo4j database with authentication."""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            self.gds = GraphDataScience(uri, auth=(username, password))
            self.result = None
            logging.info("Neo4j and GDS connections initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Neo4j connections: {e}")

    def establish_connection(self):
        """Project the graph and generate similarity scores."""
        try:
            graph_name = "pipe2"
            node_projection = ["User", "Child", "Groups", "Tags"]
            relationship_projection = {
                "UserIsInGroup": {"orientation": "UNDIRECTED"},
                "ChildBelongToParent": {"orientation": "UNDIRECTED"},
                "UserMatches": {"orientation": "UNDIRECTED"},
                "userhastags": {"orientation": "UNDIRECTED"},
                 }

            if self.gds.graph.exists(graph_name).iloc[1]:
                # Drop the existing graph if it exists
                self.gds.graph.drop(graph_name)

            # Project the graph and create it if it doesn't exist
            self.G, _ = self.gds.graph.project(graph_name, node_projection, relationship_projection)

            self._generate_similarity_scores()
            logging.info("Graph projection and similarity computation complete.")
        except Exception as e:
            logging.error(f"Error establishing graph connection: {e}")

    def _generate_similarity_scores(self):
        """Generate similarity scores for users based on embeddings."""
        embed_rule = {
            'embeddingDimension': 4,
            'iterationWeights': [0.8, 1, 1, 1],
            'randomSeed': 42,
            'mutateProperty': 'embedding'
        }
        self.gds.fastRP.mutate(self.G, **embed_rule)

        knn_config = {
            'nodeProperties': ['embedding'],
            'topK': 2,
            'sampleRate': 1.0,
            'deltaThreshold': 0.0,
            'randomSeed': 42,
            'concurrency': 1,
            'writeProperty': 'score',
            'writeRelationshipType': 'SIMILAR'
        }
        self.gds.knn.write(self.G, **knn_config)

        similar_query = """
            MATCH (n:User)-[r:SIMILAR]->(m:User)
            RETURN n.firstName AS person1, m.firstName AS person2, r.score AS similarity
            ORDER BY similarity DESCENDING, person1, person2
        """
        query_result = self.driver.session().run(similar_query)
        self.result = pd.DataFrame(query_result.data())
        logging.info("Similarity scores generated and stored in DataFrame.")

    def recommender(self, user, similar_user):
        """Generate group recommendations for a user based on similar user memberships."""
        try:
            recommend_query = f"""
                MATCH (:User {{firstName: '{user}'}})-->(g1:Groups)
                WITH collect(g1) AS groups
                MATCH (:User {{firstName: '{similar_user}'}})-->(g2:Groups)
                WHERE NOT g2 IN groups
                RETURN DISTINCT g2 AS Recommended_Group
            """
            # Run the query and convert the result to a DataFrame
            query_result = self.driver.session().run(recommend_query)
            recommendation_df = pd.DataFrame([record["Recommended_Group"] for record in query_result],
                                             columns=["Recommended_Group"])

            # Extract the recommended groups as a set
            recommendation = set(recommendation_df["Recommended_Group"])
            return recommendation
        except Exception as e:
            logging.error(f"Error in recommender method: {e}")
            return set()

    def get_username(self, email="tobias.steffensen@gmail.com"):
        """Retrieve the first name of a user based on their email."""
        try:
            query = f"MATCH (u:User {{email: '{email}'}}) RETURN u.firstName AS firstName"
            result = self.driver.session().run(query).single()
            return result['firstName'] if result else None
        except Exception as e:
            logging.error(f"Error fetching username: {e}")
            return None

    def get_similarities(self, username, show_result=False):
        """Retrieve similar users based on similarity scores."""
        try:
            filtered_df = self.result[self.result['person1'] == username]
            if show_result:
                logging.info(filtered_df)
            return set(filtered_df['person2'])
        except Exception as e:
            logging.error(f"Error in get_similarities method: {e}")
            return set()

    def get_recommendation(self, email):
        """Generate recommendations for a user based on similar user memberships."""
        try:
            username = self.get_username(email)
            if not username:
                logging.warning(f"No user found with email: {email}")
                return None

            similar_users = self.get_similarities(username)
            recommended_groups = set()
            for similar_user in similar_users:
                recommended_groups.update(self.recommender(username, similar_user))

            if recommended_groups:
                return recommended_groups
            else:
                logging.info(f"No recommendations found for user: {username} with email: {email}")
                return None
        except Exception as e:
            logging.error(f"Error in get_recommendation method: {e}")
            return None

    def close_connection(self):
        """Close connections to Neo4j and GDS."""
        try:
            self.gds.close()
            self.driver.close()
            logging.info("Neo4j and GDS connections closed successfully.")
        except Exception as e:
            logging.error(f"Error closing connections: {e}")


# Usage example
if __name__ == "__main__":
    uri = "neo4j://65.108.80.255:7687"
    recommendation_system = Neo4jRecommendationSystem(uri, "neo4j", "famoba2024")
    recommendation_system.establish_connection()
    recommendation = recommendation_system.get_recommendation("posttilnicolai@gmail.com")
    print(recommendation)
    recommendation_system.close_connection()
