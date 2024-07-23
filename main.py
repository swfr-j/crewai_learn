from dotenv import load_dotenv

load_dotenv()

from crewai import Crew, Process
from langchain_community.llms.ollama import Ollama
from agents import AINewsLetterAgents
from tasks import AINewsLetterTasks
from file_io import save_markdown

agents = AINewsLetterAgents()
tasks = AINewsLetterTasks()

llm = Ollama(
    model="openhermes:latest",
)

# setting up agents
editor = agents.editor_agent()
news_fetcher = agents.news_fetcher_agent()
news_analyzer = agents.news_analyzer_agent()
newsletter_compiler = agents.newsletter_compiler_agent()

# setting up tasks
fetch_news_task = tasks.fetch_news_task(news_fetcher)
analyzed_news_task = tasks.analyze_news_task(news_analyzer, [fetch_news_task])
compiled_newsletter_task = tasks.compile_newsletter_task(
    newsletter_compiler, [analyzed_news_task], callback_function=save_markdown
)

# setting up crew
crew = Crew(
    agents=[editor, news_fetcher, news_analyzer, newsletter_compiler],
    tasks=[fetch_news_task, analyzed_news_task, compiled_newsletter_task],
    process=Process.hierarchical,
    verbose=2,
    manager_llm=llm,
)

# running the crew
results = crew.kickoff()

print("Crew Work Results:")
print(results)
