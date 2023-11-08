import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from llama_hub.file.pdf.base import PDFReader
from llama_index.response.notebook_utils import display_source_node
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import VectorStoreIndex, ServiceContext, get_response_synthesizer, SimpleDirectoryReader
from llama_index.llms import OpenAI
import json

from llama_index import Document

from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode

from llama_index.embeddings import resolve_embed_model

import os
import openai
from flask import Flask, request, render_template


class DCPRAssistant:
    def __init__(self):
        self.openai_api_key = "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"
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self.llm = OpenAI(temperature="0.1", model="gpt-3.5-turbo", system_prompt="""\
            You are DCPR Assistant who is knowledgeable about development regulations for the buildings in Mumbai. Use the following pieces of retrieved context to answer the question. Always answer like a human and do not mention that you are just reading from context documents. If you are very sure about the answer, give a very straightforward answer.
            If you are not completely sure about the answer. Present your answer in the format
            Format Example:
            Small para of introduction and your interpretation
            1. Relevant summary chunk from document 1
            2. Relevant summary chunk from document 2
            3. Relevant summary chunk from document 3
            considerations and follow up if necessary.
            Mention that you are a learning assistant that will get better with the more questions as you ask it
            If you feel the user did not frame the question properly, you should ask for a follow up question from the user once you have presented with your interpretation of the answer
            """)

        self.ctx = ServiceContext.from_defaults(llm=self.llm, chunk_size=512)

        self.response_synthesizer = get_response_synthesizer(service_context=self.ctx, response_mode="tree_summarize")

        self.loader = PDFReader()

        self.docs = SimpleDirectoryReader("/content/sample_data/data").load_data()

        self.node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)

        self.base_nodes = self.node_parser.get_nodes_from_documents(self.docs)
        # set node ids to be a constant
        for idx, node in enumerate(self.base_nodes):
            node.id_ = f"node-{idx}"

        self.llm = OpenAI(model="gpt-3.5-turbo")
        self.service_context = ServiceContext.from_defaults(llm=self.llm)

        self.sub_chunk_sizes = [1024, 2048, 4096]
        self.sub_node_parsers = [
            SimpleNodeParser.from_defaults(chunk_size=c) for c in self.sub_chunk_sizes
        ]

        self.all_nodes = []
        for base_node in self.base_nodes:
            for n in self.sub_node_parsers:
                sub_nodes = n.get_nodes_from_documents([base_node])
                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                self.all_nodes.extend(sub_inodes)

            # also add the original node to node
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            self.all_nodes.append(original_node)

        self.all_nodes_dict = {n.node_id: n for n in self.all_nodes}

        self.vector_index_chunk = VectorStoreIndex(self.all_nodes, service_context=self.service_context)

        self.vector_retriever_chunk = self.vector_index_chunk.as_retriever(similarity_top_k=5)

        self.retriever_chunk = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": self.vector_retriever_chunk},
            node_dict=self.all_nodes_dict,
            verbose=True,
        )

    def answer_question(self, question):
        nodes = self.retriever_chunk.retrieve(question)
        answer_text = ""
        for node in nodes:
            answer_text += node.node.text + "\n"
        return answer_text



app = Flask(__name)

dcpr_assistant = DCPRAssistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = dcpr_assistant.answer_question(question)
    return render_template('answer.html', question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
