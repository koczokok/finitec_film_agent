from dataclasses import dataclass
from typing import Any
# from devtools import debug
from httpx import AsyncClient
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic_ai.exceptions import ModelRetry
# Add this import
import wptools
from wikipediaapi import Wikipedia
from pydantic import BaseModel, Field
from difflib import SequenceMatcher
from rich.prompt import Prompt
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import logfire
import asyncio
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
wiki = Wikipedia(user_agent="Film_Agent (p.kurylowicz@outlook.com)", language="en")
llm = os.getenv("LLM_MODEL")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
movie_folder = Path("./movies")
files = list(movie_folder.glob("*.md"))

file_texts = [f.read_text(encoding="utf-8") for f in files]
embeddings = sentence_model.encode(file_texts)

client = AsyncOpenAI(
    api_key=os.getenv("OPEN_API_KEY")
)
model = OpenAIModel(
    model_name="o4-mini", 
    provider=OpenAIProvider(
        api_key=os.getenv("OPEN_API_KEY")
    )
)

# client = AsyncOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key='ollama'
# )

# model = OpenAIModel(
#     model_name=llm, 
#     provider=OpenAIProvider(
#         base_url="http://localhost:11434/v1", 
#         api_key='ollama'
#     )
# )
logfire_token = os.getenv("LOGFIRE_TOKEN", "")
logfire.configure(token=logfire_token)
logfire.info('Hello, {place}!', place='World')
logfire.instrument_openai()

class Film(BaseModel):
    name: str
    budget: int = Field(default=0, description="Budget in millions USD just an integer")
    profit: int = Field(default=0, description="Profit in millions USD just an integer")
    tool_name: str = Field(default="", description="Name of the tool used")
    

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None


web_search_agent = Agent(
    model,
    system_prompt="""You are a precise financial data extractor for films.
    TOOL USAGE RULES (MANDATORY):
    1. NEVER attempt to guess, fabricate, or hallucinate financial data.
    2. ALWAYS note the tool that returned the final usable data by setting the `tool_name` field to one of: `film_file_search`, `web_search`, `wikipedia_search`.
    

    REQUIRED OUTPUT FORMAT:
    
    {
    "name": "Film Name",
    "budget": 150,
    "profit": 100,
    "tool_name": "film_file_search"
    }

    STRICT GUIDELINES:
- Do not describe or explain what you have done, return the final output as final_result tool output.
- From provided context extract the film financial data budget and profit.
- Always return numeric values in millions of USD (e.g., 150, not "150 million" or "$150M").
- If a value is unknown, default to `0` (do not guess).
- Do not include textual explanations, analysis, or summaries. Only return structured data.

You are a reliable, methodical assistant. Precision and consistency matter more than speed. Follow the tool order strictly and never skip fallback logic.""",
    output_type=Film,
    deps_type=Deps,
    retries=3,
)

@web_search_agent.tool
async def wikipedia_search(ctx: RunContext[Deps], film_name: str) -> str:
    """
    Use this tool to get film financial data from Wikipedia.
    As a query use film name.
    You will need to extract budget and box office data from the Wikipedia page.
    Budget will be budget and profit will be box office - budget.
    If Budget is a range take upper bound.
    """
    page = wptools.page(film_name)
    page.get_parse() 
    infobox = page.data.get("infobox")
    try:
        budget = infobox.get("budget")
        gross = infobox.get("gross")
    except:
        raise ModelRetry("Fallback to web search")
    return budget, gross



@web_search_agent.tool
async def internal_search(ctx: RunContext[Deps], film_name: str) -> str:
    """
    That is the most trustwhorhty source of information for film financial data.
    Always use this tool first.
    Searches for a film in the local markdown files. Falls back to web if film name mismatch."""
    query_embedding = sentence_model.encode([film_name])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    best_idx = int(np.argmax(scores))
    best_file = files[best_idx]
    best_content = file_texts[best_idx]

    
    for line in best_content.splitlines():
        if line.lower().startswith("name:"):
            extracted_title = line.split(":", 1)[1].strip()
            break
    else:
        extracted_title = best_file.stem  

    ratio = SequenceMatcher(None, film_name.lower(), extracted_title.lower()).ratio()
    if ratio < 0.75:
      
        raise ModelRetry("Fallback to wikipedia search")
      

    return f"From file: {best_file.name}\n\n{best_content}"



@web_search_agent.tool
async def web_search(ctx: RunContext[Deps], web_query: str) -> str:

    """ 
    Create a web_query to search for film financial information
    This is your last resort tool. Use it only if you cannot find the information in the files. 
    If you cannot find the information try to rerun the tool with a different query.
    In your web_query try to be as specific as possible and include the year of the film.
    Profit should be box office - budget.

    """
    if not ctx.deps.brave_api_key:
        return "Brave API key not provided."

    headers = {
        "X-Subscription-Token": ctx.deps.brave_api_key,
        "Accept": "application/json",
    }
    response = await ctx.deps.client.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers=headers,
        params={
            "q": web_query + " financial details budget profit",
            "count": 5,
            "text_decorations": "true",
            "search_lang": "en",
        }
    )
    response.raise_for_status()
    data = response.json()
    
    web_results = data.get("web", {}).get("results", [])
    search_text = "\n".join([
        f"{result.get('title', '')} {result.get('description', '')}" 
        for result in web_results
    ])
    
    return search_text


    

async def main():
    async with AsyncClient() as client:
        brave_api_key = os.getenv("BRAVE_API_KEY", "")
        if not brave_api_key:
            print("Error: BRAVE_API_KEY is not set in .env")
            return
        
        deps = Deps(client=client, brave_api_key=brave_api_key)

        
       
        try:
            results = await web_search_agent.run(f"Find me financial summary of film Hacksaw Ridge", deps=deps)
            print("Search Results:")
            print(results.output)
          
        
        except Exception as e:
            print(f"Error in web search: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())