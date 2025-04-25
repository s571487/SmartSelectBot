from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough


try:
  import google.colab
  from google.colab import output
  output.enable_custom_widget_manager()
except:
  pass

os.environ["OPENAI_API_KEY"] = "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" # replace your OPEN_API key
os.environ["NEO4J_URI"] = "neo4j+s://5b1684a3.databases.neo4j.io" # NEO4J_URI
os.environ["NEO4J_USERNAME"] = "neo4j" # NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = "y1guz43OpzBbI5_SydXAfUB0yiCWunao4yCbb3s6sjM" # NEO4J_PASSWORD

_search_query = RunnableLambda(lambda x : x["question"])


graph = Neo4jGraph()

llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125") # gpt-4-0125-preview occasionally has issues
llm_transformer = LLMGraphTransformer(llm=llm)

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)


graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the courses, requirements, or business entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting product names, categories, features, and brand entities from the user's input.",
        ),
        (
            "human",
            "Use the given format to extract information from the following input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f""" Structured data: {structured_data} Unstructured data: {"#Document ". join(unstructured_data)}  """
    return final_data


def answerquery(question: str):
  template = """Answer the question based only on the following context:
  {context}

  Question: {question}
  Use natural language and be concise.
  Answer:"""
  prompt = ChatPromptTemplate.from_template(template)

  chain = (
      RunnableParallel(
          {
              "context": _search_query | retriever,
              "question": RunnablePassthrough(),
          }
      )
      | prompt
      | llm
      | StrOutputParser()
  )

  return chain.invoke(({"question": question}))
