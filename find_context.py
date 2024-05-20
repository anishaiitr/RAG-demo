from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from trulens_eval import TruChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from build_index import split_documents
import os
from trulens_eval.feedback.provider import OpenAI
from trulens_eval.feedback import Feedback
import numpy as np
from trulens_eval import TruChain
import numpy as np
from trulens_eval import Feedback, Select
from datetime import datetime
from trulens_eval import Tru


def find_context(question: str, index_path: str, openai: str):
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
    {"context": faiss_retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

    # Setting up tru lens for evaluation of rag chain
    os.environ["OPENAI_API_KEY"] = openai
    tru = Tru()
    provider = OpenAI("gpt-3.5-turbo-16k")
    context = TruChain.select_context(rag_chain)
    f_context_relevance = (
        Feedback(provider.context_relevance)
        .on_input()
        .on(context)
        .aggregate(np.mean)
        )
    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()

    f_groundedness = Feedback(
        provider.groundedness_measure_with_cot_reasons, 
        name = "Groundedness").on(context).on_output()

    tru_recorder = TruChain(rag_chain, app_id="App_3",
        feedbacks=[f_context_relevance,
                f_qa_relevance,
                f_groundedness
        ])
    
    # Generating response
    with tru_recorder as recording:
        response = rag_chain.invoke(question)

    eval_df = tru.get_leaderboard(app_ids=[])
    print(eval_df)
    date = datetime.now()
    eval_df.to_csv(f'results/rag-eval-{date}.csv')

    return response['text']

