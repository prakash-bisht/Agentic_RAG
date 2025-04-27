from langchain_openai import ChatOpenAI
import os
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import tool
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY ')

llm = LLM(model="groq/deepseek-r1-distill-qwen-32b")

rag_tool = PDFSearchTool(pdf='attenstion_is_all_you_need.pdf',
    config=dict(
        llm=dict(
            provider="groq", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="deepseek-r1-distill-qwen-32b",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="huggingface", # or openai, ollama, ...
            config=dict(
                model="BAAI/bge-small-en-v1.5",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)

@tool("web_search_tool")
def web_search_tool(question: str) -> str:
    """This tool is useful when we want web search for current events."""
    websearch = TavilySearchResults()
    response = websearch.invoke({"query":question})
    return response

@tool("router tool")
def router_tool(question:str) -> str:
    """Router Function"""
    prompt = f"""Based on the Question provided below, determine whether it is eligible for a vectorstore search or a web search.
                Return 'vectorstore' if it is eligible for vectorstore search, otherwise return 'websearch'.
                Question: {question}
                """
    response = llm.invoke(prompt).content.strip()  # Ensure the response is clean and matches expected output
    if response.lower() == "vectorstore":
        return 'vectorstore'
    else:
        return 'websearch'


@tool("retriver tool")
def retriver_tool(router_response:str, question:str) -> str:
    """Retriever Function"""
    if router_response == 'vectorstore':
        return rag_tool(question)  # Perform vectorstore search using rag_tool
    elif router_response == 'websearch':
        return web_search_tool(question)  # Perform web search using web_search_tool
    else:
        return "Invalid response from router"  # Handle unexpected cases

  
Router_Agent = Agent(
  role='Router',
  goal='Route user question to a vectorstore or web search',
  backstory=(
    "You are an expert at routing a user question to a vectorstore or web search."
    "Use the vectorstore for questions on concept related to Retrieval-Augmented Generation."
    "You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
  tools=[router_tool],
)

Retriever_Agent = Agent(
role="Retriever",
goal="Use the information retrieved from the vectorstore to answer the question",
backstory=(
    "You are an assistant for question-answering tasks."
    "Use the information present in the retrieved context to answer the question."
    "You have to provide a clear concise answer."
),
verbose=True,
allow_delegation=False,
llm=llm,
tools=[retriver_tool,web_search_tool],
)

router_task = Task(
    description=("Analyse the keywords in the question {question}"
    "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
    "Return a single word 'vectorstore' if it is eligible for vectorstore search."
    "Return a single word 'websearch' if it is eligible for web search."
    "Do not provide any other premable or explaination."
    ),
    expected_output=("Give a binary choice 'websearch' or 'vectorstore' based on the question"
    "Do not provide any other premable or explaination."),
    agent=Router_Agent,
)

retriever_task = Task(
    description=("Based on the response from the router task extract information for the question {question} with the help of the respective tool."
    "Use the web_search_tool to retrieve information from the web in case the router task output is 'websearch'."
    "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
    ),
    expected_output=("You should analyse the output of the 'router_task'"
    "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
    "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
    "Return a claer and consise text as response."),
    agent=Retriever_Agent,
    context=[router_task],
)

rag_crew = Crew(
    agents=[Router_Agent, Retriever_Agent],
    tasks=[router_task, retriever_task],
    verbose=True,

)

# inputs ={"question":"What is self attention formula?"}
inputs = {"question":"Who won cricket t20 world cup 2024?"}
result = rag_crew.kickoff(inputs=inputs)
result 