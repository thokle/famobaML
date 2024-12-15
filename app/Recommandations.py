# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:37:54 2024

@author: Thomas Kleist
"""
import json
import pandas as pd
from py2neo import Graph
from graphdatascience import GraphDataScience


class Neo4jRecommendationSystem:
    def __init__(self, uri, username, password):
        self.graph = Graph(uri, auth=(username, password))
        self.gds = GraphDataScience(uri, auth=(username, password))
        self.result = None

    def establish_connection(self):
        try:
            graph_name = "famoba"
            node_projection = ["User", "Child", "Groups"]
            relationship_projection = {
                "UserIsInGroup": {"orientation": "UNDIRECTED"},
                "ChildBelongToParent": {"orientation": "UNDIRECTED"},
                "UserMatches": {"orientation": "UNDIRECTED"}

            }

            if self.gds.graph.exists(graph_name).iloc[1]:
                self.gds.graph.drop(graph_name)

            self.G, _ = self.gds.graph.project(graph_name, node_projection, relationship_projection)

            embed_rule = {'embeddingDimension': 4, 'iterationWeights': [0.8, 1, 1, 1],
                          'randomSeed': 42, 'mutateProperty': 'embedding'}
            self.gds.fastRP.mutate(self.G, **embed_rule)

            knn_config = {'nodeProperties': ['embedding'], 'topK': 2, 'sampleRate': 1.0,
                          'deltaThreshold': 0.0, 'randomSeed': 42, 'concurrency': 1,
                          'writeProperty': 'score', 'writeRelationshipType': 'SIMILAR'}
            self.gds.knn.write(self.G, **knn_config)

            similar_query = """ MATCH (n:User)-[r:SIMILAR]->(m:User)  where n.gender = m.gender
                                RETURN n.firstname AS person1, m.firstname AS person2, r.score AS similarity
                                ORDER BY similarity DESCENDING, person1, person2 """
            self.result = self.graph.run(similar_query).to_data_frame()
        except Exception as e:
            print(f"An error occurred while establishing connection: {str(e)}")

    def recommender(self, user, similar_user):
        try:
            recommend_query = f"""MATCH (:User {{firstname: '{user}'}})-->(g1:Groups)                  
                                  WITH collect(g1) AS groups
                                  MATCH (:User {{firstname: '{similar_user}'}})-->(g2:Groups)
                                  WHERE NOT g2 IN groups
                                  RETURN DISTINCT g2 AS Recommended_Group"""
            recommendation = set(self.graph.run(recommend_query).to_series(dtype='object'))
            return recommendation
        except Exception as e:
            print(f"An error occurred in recommender: {str(e)}")

    def get_username(self, email="tobias.steffensen@gmail.com"):
        try:
            email_to_name_query = f"""MATCH (u:User {{email: '{email}'}}) RETURN u.`firstname`"""
            name_result = self.graph.run(email_to_name_query)
            for record in name_result:
                firstName = record['u.`firstname`']
            return firstName
        except Exception as e:
            print(f"An error occurred in get_username: {str(e)}")

    def get_similarities(self, username, show_result=False):
        try:
            filtered_df = self.result[self.result['person1'] == username]
            if show_result:
                print(filtered_df)
            similar = set(filtered_df['person2'])
            return similar
        except Exception as e:
            print(f"An error occurred in get_similarities: {str(e)}")

    def get_recommendation(self, email):
        try:
            username = self.get_username(email)
            similar_users = self.get_similarities(username)
            recommended_groups = set()
            for similar_user in similar_users:
                recommendation = self.recommender(username, similar_user)
                recommended_groups.update(recommendation)
            if recommended_groups:
                return recommended_groups
            else:
                print(f"There is no suitable recommendation for user: {username} with email: {email}")
        except Exception as e:
            print(f"An error occurred in get_recommendation: {str(e)}")

    def close_connection(self):
        try:
            self.gds.close()
        except Exception as e:
            print(f"An error occurred while closing connection: {str(e)}")


# Usage example
if __name__ == "__main__":
    uri = "neo4j://65.108.80.255:7687"
    username = "neo4j"  # replace with your Neo4j username
    password = "famoba2024"  # replace with your Neo4j password

    recommendation_system = Neo4jRecommendationSystem(uri, username, password)
    recommendation_system.establish_connection()
    recommendation = recommendation_system.get_recommendation("posttilnicolai@gmail.com")
    print(recommendation)
    recommendation_system.close_connection()
