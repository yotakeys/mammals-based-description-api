import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr


class MammalsSearcher():
    index = faiss.read_index('resources/mammals.index')
    data = pd.read_csv('resources/data.csv')
    model = SentenceTransformer('resources/model/')

    query = ""
    top_k = 0
    results = list()

    def fetch_mammals(self, dataframe_idx):
        info = self.data.iloc[dataframe_idx]
        meta = dict()
        meta['organism_name'] = info['organism_name']
        return meta

    def search(self):
        query_vector = self.model.encode([self.query])
        self.top_k = self.index.search(query_vector, self.top_k)
        result_id = self.top_k[1].tolist()[0]
        result_id = list(np.unique(result_id))
        results = [self.fetch_mammals(idx) for idx in result_id]
        return results

    def recommend(self, query, top_k=5):
        self.top_k = top_k
        self.query = query

        self.results = self.search()

        return self.results


def giveRecommend(Description, Amount):
    results = Mammals.recommend(Description, int(Amount))

    out = ""
    for result in results:
        out += str(result['organism_name']) + ",\n"

    return out


Mammals = MammalsSearcher()

# Main program
app = gr.Interface(fn=giveRecommend,
                   inputs=["textbox", "number"],
                   outputs="textbox",
                   organism_name="Mammals Searcher",
                   description="Search for mammals by description."
                   )
app.launch()
