from langchain_community.llms.ollama import Ollama
from langchain_community.tools import DuckDuckGoSearchResults
from crewai import Crew, Agent, Task, Process
from crewai_tools import tool

llm = Ollama(model="gemma2")


@tool("Search the internet using duckduckgo")
def search(query: str):
    """Search the internet using duckduckgo"""
    search_results = DuckDuckGoSearchResults().run(query)
    return search_results


# Define agents with specific roles and tools
researcher = Agent(
    role="Senior Research Analyst",
    goal="Discover key points on India's budget news for the year 2024-25",
    backstory="""You're a senior research analyst at a large company.
        You're responsible for analyzing data and providing insights
        to the business.
        You're currently working on a project to find key points on India's budget news for the year 2024-25.""",
    tools=[search],
    llm=llm,
)

writer = Agent(
    role="Content Writer",
    goal="Write Key points on India's budget news for the year 2024-25",
    backstory="""You're a senior writer at a large company.
        You're responsible for creating content to the business.
        You're currently working on a project to write key points on India's budget news for the year 2024-25.""",
    verbose=True,
    llm=llm,
)

# Create tasks for the agents
research_task = Task(
    description="Search the internet for key points on India's budget news for the year 2024-25",
    agent=researcher,
    expected_output="A string containing key points on India's budget news for the year 2024-25",
)
write_article_task = Task(
    description="Write an article on India's budget news for the year 2024-25",
    agent=writer,
    expected_output="Article on India's budget news for the year 2024-25 containing key points and insights",
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_article_task],
    processes=Process.sequential,
    full_output=True,
    verbose=True,
    manager_llm=llm,
)


result = crew.kickoff()

print(result)
