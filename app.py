import os
from crewai import Agent, Task, Crew
from crewai_tools import tool, SerperDevTool

os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["SERPER_API_KEY"] = ""
llm = "gpt-4o-mini"

# Define the tools
@tool("read_text_file")
def read_text_file(filename: str):
    """
    Reads a text file from the 'data' directory and returns its content.

    Args:
    filename (str): The name of the file to read (with or without .txt extension).

    Returns:
    str: Content of the file read, or None if an error occurs.
    """
    directory = 'data'
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    file_path = os.path.join(directory, filename)
    
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' does not exist.")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}")
        return None
    
# Define search tool for internet searches
search_tool = SerperDevTool()
    
# Define the agents
article_summary_agent = Agent(
    role='Summary Agent',
    goal='Read article content and provide a summary which highlights the main points of the document',
    backstory='You are well read agent who excels at writing concise summaries of techncial articles',
    llm=llm,
    tools=[read_text_file]
)

keyword_agent = Agent(
    role='Keyword Agent',
    goal='Extract the semantic meaning from the article and search keywords to create web searches for similar content',
    backstory='You are an expert literary analyst who specialises in extracting semantic meaning ad keywords from writing',
    llm=llm,
    tools=[read_text_file]
    
)

research_agent = Agent(
    role='Research Agent',
    goal='Use keywords and semantic meaning to search the internet for similar content, return a list of URLs & webpage titles',
    backstory='You are an expert researcher who can compose excellent search queries and find web pages which match a set of keywords',
    llm=llm,
    tools=[search_tool]
)

summary_agent = Agent(
    role='Summary Agent',
    goal='Combines multiple sources of information into an informative short article which could be placed in a newsletter',
    backstory='You are an expert writer who can combine multiple related sources of information into an engaging summary',
    llm=llm
)

# Define the tasks
read_article = Task(
    description='Reads the article.txt content and produces an engaging summary',
    expected_output='A summary of no more than 300 words which captures the key points of the article and will entice readers to the original article',
    agent=article_summary_agent
)

extract_meaning = Task(
    description='Reads the article and extracts semantic meaning & keywords from it',
    expected_output='A short description, 10 words maximum, of the articles meaning and a list of 5 keywords which a user might search for to find the article',
    agent=keyword_agent
)

search_web = Task(
    description='Searches the web and returns a set of web page titles & links to content which matches the search',
    expected_output='A list of results which match the provided keywords. The links should be formatted in markdown',
    context=[extract_meaning],
    agent=research_agent
)

write_summary = Task(
    description='Combines the results of other agents into an engaging format',
    expected_output='The output should include the summary, semantic meaning, keywords & search results formatted in markdown',
    context=[read_article, extract_meaning, search_web],
    agent=summary_agent
)

crew = Crew(
    agents=[article_summary_agent, keyword_agent, research_agent, summary_agent],
    tasks=[read_article, extract_meaning, search_web, write_summary],
    verbose=True,
    planning=True
)

crew.kickoff()
task_output = write_summary.output

print(f"Task Description: {task_output.description}")
print(f"Task Summary: {task_output.summary}")
print(f"Raw Output: {task_output.raw}")