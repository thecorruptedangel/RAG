import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# Secure retrieval of API key with error handling
def get_groq_api_key():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment variables.")
    return api_key

# Load and cache PDF content
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    document_pages = loader.load()
    return "\n".join([page.page_content for page in document_pages])

# Initialize the LLM model
def init_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model_name='gemma2-9b-it',
        temperature=0,
    )

# Define RAG prompt template
rag_template = """You will STRICTLY ADHERE TO THE FOLLOWING INSTRUCTIONS: Your name is Lucy from Tech support. 
Your role is to Help with user's queries only. If any user's question is out of context or unrelated to what is already in your context,
only respond '[UNKNOWN_QUERY]' without any further explanation.
Try to calm users if they become frustrated. You are allowed to welcome users and be polite when user thanks or acknowledge or ending the call.
Your only job is to respond to used based on the information in context provided and nothing else.
Do not acknowledge your other skills such as coding, math or other skills you have.

Context: {context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

# Process user question with LLM
def process_question(user_question, llm, full_text):
    start_time = time.monotonic()
    final_prompt = rag_prompt.format(context=full_text, question=user_question.strip())
    response = llm.invoke(final_prompt)
    end_time = time.monotonic()
    
    response_time = f"Response time: {end_time - start_time:.2f} seconds."
    return response_time, response.content

# Main function for user interaction
def main():
    try:
        print("Loading document...")
        full_text = load_pdf("resume.pdf")
        groq_api_key = get_groq_api_key()

        # Initialize the LLM once, when first question is asked
        llm = init_llm(groq_api_key)

        print("Ask any question related to the document 'resume.pdf'. Type 'exit' to quit.")
        while True:
            user_question = input("\nYour question: ")
            if user_question.lower() == 'exit':
                print("Exiting...")
                break

            response_time, response = process_question(user_question, llm, full_text)
            print(response)
            print(response_time)

    except KeyboardInterrupt:
        print("\nGracefully exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
