from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


prompt_templete = """
<prompt>
  <role>Metaviz Information Assistant</role>
  <description>
    RAG-based chatbot that provides details about Metaviz by retrieving data and generating natural, informative responses.
  </description>
  <goals>
    <primary>Retrieve raw information and manipulate it into useful answers.</primary>
    <secondary>Maintain a casual, human-like tone and provide relevant, accurate information.</secondary>
  </goals>
  <instructions>
    <step>Analyze the user's message to understand intent.</step>
    <step>Retrieve relevant data from the vector database based on the query.</step>
    <step>Manipulate the retrieved data into a natural, easy-to-read format.</step>
    <step>Always provide accurate, fact-based information.</step>
    <step>If no relevant data is found, apologize and let the user know politely.</step>
  </instructions>
  <Style>
    <tone>Conversational</tone>
    <formality>Casual yet professional</formality>
    <output>Clear, informative, and aligned with Metaviz’s branding</output>
  </Style>
  <Context>{context}</Context>
  <ChatHistory>{chat_history}</ChatHistory>
</prompt>
"""

class TourBot:
    def __init__(self, prompt):
        self.prompt_templete = prompt
        self.embedds = self.return_embeddind()
        self.prompt = self.return_prompt()
        self.llm = self.return_llm()
        self.retriver = self.load_data_from_pinecone()
        self.doc_chain = self.return_doc_chain()
        self.retrival_chain = self.return_retrival_chain()
        self.history = self.return_memory()
        self.runnable = self.return_runnabale_with_history()


    def return_prompt(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_templete),
            ("human", "{input}")
        ])
        return prompt
    def return_embeddind(self):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    
    def return_llm(self):
        llm = ChatGroq(model="gemma2-9b-it", temperature=0.3)
        return llm

    def load_data_from_pinecone(self):
        pc = Pinecone()
        index_name = "metaviz-final-base"
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=self.embedds)
        retriver = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriver
    
    def return_doc_chain(self):
        return create_stuff_documents_chain(llm=self.llm, prompt=self.prompt)
    
    def return_retrival_chain(self):
        return create_retrieval_chain(self.retriver, self.doc_chain)
    
    def return_memory(self):
        return StreamlitChatMessageHistory()
    
    def return_runnabale_with_history(self):
        return RunnableWithMessageHistory(
            self.retrival_chain,
            lambda: self.history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
    
    def return_response(self, query):
        resp =  self.runnable.invoke({"input": query}, config={"configurable": {"session_id": "user123"}})
        return resp['answer']
    
        



if "bot" not in st.session_state:
    st.session_state.bot = TourBot(prompt=prompt_templete)

st.set_page_config(page_title="Metaviz ChatBot", page_icon="🤖")
st.title("📚 Ask Anything about Metaviz")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Placeholder for bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.bot.return_response(prompt)

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
