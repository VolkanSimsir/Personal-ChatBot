from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage


chat_history = [
    SystemMessage("sen bir kişisel sohbet botusun.İsmin loracan")

] 

class ChatBot:
    def __init__(self,model_name):
        self.model_name = model_name
        self.load_gpt = None
        if self.load_gpt is None:
            self.load_gpt = self.load_model()


    def load_model(self):
        load_gpt= ChatOpenAI(temperature=0.7, model_name=self.model_name)

        return load_gpt


    def generate_response(self,user_query):
        chat_history.append(HumanMessage(content=user_query))
        response = self.load_gpt.invoke(chat_history)
        chat_history.append(response.content)

        return response.content

            
if __name__ == "__main__":
    chatbot = ChatBot(model_name="gpt-3.5-turbo")
    
