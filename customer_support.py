import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def load_document(file_path):
    print("Loading the document...")
    return TextLoader(file_path).load()

def chunk_data(data, chunk_size=1000, chunk_overlap=20):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    ).split_documents(data)

def create_embeddings(chunks, persist_directory='./chroma_db'):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if os.path.exists(persist_directory):
        print("Loading existing vectorstore...")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    print("Creating new vectorstore...")
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    
    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )

def process_unknown_query(user_query):
    folder_path = 'unknown_queries'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, 'queries.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(file_path, 'a') as file:
        file.write(f"{timestamp}: {user_query}\n")
    
    return "Our team will get back to you shortly regarding this query."

def setup_qa_chain(vector_store):
    system_message = (
        "You are an empathetic and helpful customer support agent. But If the information is not in the context or previous queries, only respond '[UNKNOWN_QUERY]'. Respond to user queries based on the provided context but you can paraphrase it if required depending on conversation. Keep responses concise and in a single paragraph without special formatting.")
    
    prompt_template = PromptTemplate(
        template=f"{system_message}\n\nContext: {{context}}\n\nChat History: {{chat_history}}\n\nHuman: {{question}}\n\nAssistant:",
        input_variables=["context", "chat_history", "question"]
    )
    
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
    
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5})
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

def main():
    file_path = "bosch.txt"
    data = load_document(file_path)
    
    if data:
        chunks = chunk_data(data)
        vector_store = create_embeddings(chunks)
        print('System ready')
        
        qa_chain = setup_qa_chain(vector_store)
        chat_history = []
        
        while True:
            q = input("Ask a question (or type 'quit' to exit): ")
            if q.lower() == 'quit':
                break
            
            result = qa_chain.invoke({"question": q, "chat_history": chat_history})
            answer = result['answer']
            
            if '[UNKNOWN_QUERY]' in answer:
                answer = process_unknown_query(q)
            else:
                chat_history.append((q, answer))
            
            print("\nAnswer:")
            print(answer)
            print("-" * 50)

if __name__ == "__main__":
    main()