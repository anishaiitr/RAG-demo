from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chat_models import ChatOpenAI
import os
from langchain.prompts import ChatPromptTemplate
import random
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import LLMChain
from operator import itemgetter

# Function to create dataset for RAGAS, which includes question, answer, context and groundtruth which are generated from GPT-3.5
def create_ragas_dataset(rag_pipeline, eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    answer = rag_pipeline.invoke({"question" : row["question"]})
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer["response"],
         "contexts" : [context.page_content for context in answer["context"]],
         "ground_truths" : [row["ground_truth"]]
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

# Defining RAGAS metrics
def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
  )
  return result

# Helper function to define rag chain to split documents
def split_documents():
    loader = PyPDFDirectoryLoader("data/rag")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(docs)
    return chunked_documents

# Function to define rag chain to be evaluated 
def find_context(index_path='index/faiss_index'):
    # Loading FAISS index
    print('Loading index')
    db = FAISS.load_local(index_path, embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'), allow_dangerous_deserialization=True)
    print('Loaded index')

    # Creating sparse embeddings retrievers
    chunked_documents = split_documents()
    bm25_retriever = BM25Retriever.from_documents(chunked_documents)
    bm25_retriever.k=5

    # Creating ensemble retriever
    faiss_retriever = db.as_retriever(search_kwargs={"k":5})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,faiss_retriever],
                                       weights=[0.2,0.8])
    

    # Defining prompt
    prompt_template = """
    ### [INST] Instruction: Answer the question based on the given context only:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    # Create endpoint to make inference call on Mistral
    HUGGINGFACEHUB_API_TOKEN = "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"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN
    )

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain 
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    rag_chain = ( 
    {"context": itemgetter("question") | ensemble_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | llm, "context": itemgetter("context")}
    )
    return rag_chain

if __name__ == "__main__":
    loader = PyPDFDirectoryLoader("data/")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                chunk_overlap = 16,
                                                length_function=len)
    docs = text_splitter.split_documents(pages)


    question_schema = ResponseSchema(
        name="question",
        description="a question about the context."
    )
    question_response_schemas = [
        question_schema,
    ]

    question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
    format_instructions = question_output_parser.get_format_instructions()

    os.environ["OPENAI_API_KEY"] = "sk-proj-MgkJJfg21ZCoCrziAi1XT3BlbkFJEciB52YuH3ThYC1uTcdP"
    question_generation_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    bare_prompt_template = "{content}"
    bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)

    qa_template = """\
    You are a user who wants to know about insurance products. For each context, create a question that is specific to the context. Avoid creating generic or general questions.
    question: a question about the context.
    Format the output as JSON with the following keys:
    question
    context: {context}
    """
    prompt_template = ChatPromptTemplate.from_template(template=qa_template)
    messages = prompt_template.format_messages(
        context=docs[0],
        format_instructions=format_instructions
    )
    question_generation_chain = bare_template | question_generation_llm
    response = question_generation_chain.invoke({"content" : messages})
    output_dict = question_output_parser.parse(response.content)

    random.seed(42)
    qac_triples = []
    # randomly select 100 chunks from the ~1300 chunks
    for text in tqdm(random.sample(docs, 100)):
        messages = prompt_template.format_messages(
            context=text,
            format_instructions=format_instructions
        )
        response = question_generation_chain.invoke({"content" : messages})
        try:
            output_dict = question_output_parser.parse(response.content)
        except Exception as e:
            continue
        output_dict["context"] = text
        qac_triples.append(output_dict)

    answer_generation_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    answer_schema = ResponseSchema(
        name="answer",
        description="an answer to the question"
    )
    answer_response_schemas = [
        answer_schema,
    ]
    answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
    format_instructions = answer_output_parser.get_format_instructions()

    qa_template = """\
    You are an expert creating a test for advanced students. For each question and context, create an answer.
    answer: a answer about the context.
    Format the output as JSON with the following keys:
    answer
    question: {question}
    context: {context}
    """
    prompt_template = ChatPromptTemplate.from_template(template=qa_template)
    messages = prompt_template.format_messages(
        context=qac_triples[0]["context"],
        question=qac_triples[0]["question"],
        format_instructions=format_instructions
    )
    answer_generation_chain = bare_template | answer_generation_llm
    response = answer_generation_chain.invoke({"content" : messages})
    output_dict = answer_output_parser.parse(response.content)

    for triple in tqdm(qac_triples):
        messages = prompt_template.format_messages(
            context=triple["context"],
            question=triple["question"],
            format_instructions=format_instructions
        )
        response = answer_generation_chain.invoke({"content" : messages})
        try:
            output_dict = answer_output_parser.parse(response.content)
        except Exception as e:
            continue
        triple["answer"] = output_dict["answer"]


    ground_truth_qac_set = pd.DataFrame(qac_triples)
    ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))
    ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer" : "ground_truth"})
    eval_dataset = Dataset.from_pandas(ground_truth_qac_set)

    rag_chain = find_context()
    basic_qa_ragas_dataset = create_ragas_dataset(rag_chain, eval_dataset)

    # Below is a sample of the above script run output, this script takes a very long time to run
    ## {'context_precision': 0.4486, 'faithfulness': 0.6667, 'answer_relevancy': 0.9930, 'context_recall': 0.5000, 'context_relevancy': 0.0208, 'answer_correctness': 0.2953, 'answer_similarity': 0.8813}






