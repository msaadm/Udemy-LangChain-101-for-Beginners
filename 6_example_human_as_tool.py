from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# import pprint
# from langchain.agents import get_all_tool_names
# pp = pprint.PrettyPrinter(indent=4)
# pprint.pprint(get_all_tool_names())

from langchain_openai import OpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

llm = OpenAI(temperature=0)
tools = load_tools(["human"])

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "What's my friend Waheed's surname?"}))
