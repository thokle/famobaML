import pandas as pd
from py2neo import Graph
from graphdatascience import GraphDataScience
class PipelineClass(object):
    uri = "neo4j://65.108.80.255:7687"
    graph = Graph(uri)
    gds = GraphDataScience(uri)

    def createPipline(self, pipeline):
        return self.gds.model.get(pipeline)





