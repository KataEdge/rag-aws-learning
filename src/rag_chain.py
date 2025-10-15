from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(retriever, llm):
    """
    Get the RAG chain.
    """
    template = """
    You are a helpful assistant.
    Answer the question based only on the following context.
    
    Context:
    {context}
    
    Question:
    {question}
    """
    
    prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
