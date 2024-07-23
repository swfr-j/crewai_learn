from crewai import Crew, Process
from agents import AINewsLetterAgents
from tasks import AINewsLetterTasks

agents = AINewsLetterAgents()
tasks = AINewsLetterTasks()

# setting up agents
editor = agents.editor_agent()
news_fetcher = agents.news_fetcher_agent()
news_analyzer = agents.news_analyzer_agent()
newsletter_compiler = agents.newsletter_compiler_agent()

# setting up tasks
fetch_news_task = tasks.fetch_news_task(news_fetcher)
analyzed_news_task = tasks.analyze_news_task(news_analyzer, [fetch_news_task])
# TODO: Add callback function
compiled_newsletter_task = tasks.compile_newsletter_task(
    newsletter_compiler, [analyzed_news_task], callback_function=None
)

# setting up tools
crew = Crew(
    agents=[editor, news_fetcher, news_analyzer, newsletter_compiler],
    tasks=[fetch_news_task, analyzed_news_task, compiled_newsletter_task],
    process=Process.hierarchical,
    verbose=2,
    # manager_llm,
)
