from langchain_community.llms.ollama import Ollama
from langchain_community.tools import DuckDuckGoSearchResults
from crewai import Crew, Agent, Task, Process
from crewai_tools import tool

llm = Ollama(model="openhermes")


@tool("Search the internet using duckduckgo")
def search(query: str):
    """Search the internet using duckduckgo"""
    search_results = DuckDuckGoSearchResults().run(query)
    return search_results


# Define agents with specific roles and tools
researcher = Agent(
    role="Senior Research Analyst",
    goal="Discover innovative AI technologies",
    backstory="""You're a senior research analyst at a large company.
        You're responsible for analyzing data and providing insights
        to the business.
        You're currently working on a project to analyze the
        trends and innovations in the space of artificial intelligence.""",
    tools=[search],
    llm=llm,
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging articles on AI discoveries",
    backstory="""You're a senior writer at a large company.
        You're responsible for creating content to the business.
        You're currently working on a project to write about trends
        and innovations in the space of AI for your next meeting.""",
    verbose=True,
    llm=llm,
)

# Create tasks for the agents
research_task = Task(
    description="Identify breakthrough AI technologies",
    agent=researcher,
    expected_output="A bullet list summary of the top 5 most important AI news",
)
write_article_task = Task(
    description="Draft an article on the latest AI technologies",
    agent=writer,
    expected_output="3 paragraph blog post on the latest AI technologies",
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
