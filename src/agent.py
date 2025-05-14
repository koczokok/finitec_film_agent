from dataclasses import dataclass
from typing import Any
# from devtools import debug
from httpx import AsyncClient
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
# Add this import
from pydantic import BaseModel, Field

from rich.prompt import Prompt
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import logfire
import asyncio
import os

load_dotenv()
llm = os.getenv("LLM_MODEL", "gpt-4o")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
movie_folder = Path("./movies")
files = list(movie_folder.glob("*.md"))

file_texts = [f.read_text(encoding="utf-8") for f in files]
embeddings = sentence_model.encode(file_texts)

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key='ollama'
)

model = OpenAIModel(
    model_name=llm, 
    provider=OpenAIProvider(
        base_url="http://localhost:11434/v1", 
        api_key='ollama'
    )
)
logfire_token = os.getenv("LOGFIRE_TOKEN", "")
logfire.configure(token=logfire_token)
logfire.info('Hello, {place}!', place='World')
logfire.instrument_openai()

class Film(BaseModel):
    name: str
    budget: int = Field(default=0, description="Budget in million USD")
    domestic_box_office: int = Field(default=0, description="Domestic Box Office in million USD")
    international_box_office: int = Field(default=0, description="International Box Office in million USD")
    worldwide_box_office: int = Field(default=0, description="Worldwide Box Office in million USD")
    dvd_sales: int = Field(default=0, description="DVD Sales in million USD")
    total_earnings: int = Field(default=0, description="Total Earnings in million USD")
    profit: int = Field(default=0, description="Profit in million USD")
    

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None


web_search_agent = Agent(
    model,
    system_prompt="""You are a precise financial data extractor for films.
    REQUIRED OUTPUT FORMAT:
    ```
    Name: [Film Name]
    Budget: [Integer amount in USD]
    Domestic Box Office: [Integer amount in USD]
    International Box Office: [Integer amount in USD]
    Total Earnings: [Integer amount in USD]
    Profit: [Integer amount in USD]
    ```
    Guidelines:
    1. Always return numeric values
    2. If no exact data found, use best available estimate
    3. Total Earnings = Domestic + International Box Office
    4. Profit = Total Earnings - Budget
    5. If no data available, return zeros""",
    # output_type=Film,
    deps_type=Deps,
    retries=2,
)

@web_search_agent.tool
async def film_file_search(ctx: RunContext[Deps], film_name: str) -> str:
    """Finds the most relevant movie file for a given title and returns its content."""
    query_embedding = sentence_model.encode([film_name])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    best_idx = int(np.argmax(scores))
    return f"From file: {files[best_idx].name}\n\n{file_texts[best_idx]}"


# @web_search_agent.tool
# async def web_search(ctx: RunContext[Deps], web_query: str) -> str:
#     """ Search the web for film financial information. """
    
#     if not ctx.deps.brave_api_key:
#         return "Brave API key not provided."

#     headers = {
#         "X-Subscription-Token": ctx.deps.brave_api_key,
#         "Accept": "application/json",
#     }
#     response = await ctx.deps.client.get(
#         "https://api.search.brave.com/res/v1/web/search",
#         headers=headers,
#         params={
#             "q": web_query + " financial details box office budget",
#             "count": 5,
#             "text_decorations": "true",
#             "search_lang": "en",
#         }
#     )
#     response.raise_for_status()
#     data = response.json()
    
#     web_results = data.get("web", {}).get("results", [])
#     search_text = "\n".join([
#         f"{result.get('title', '')} {result.get('description', '')}" 
#         for result in web_results
#     ])
    
#     return search_text


    

async def main():
    async with AsyncClient() as client:
        brave_api_key = os.getenv("BRAVE_API_KEY", "")
        if not brave_api_key:
            print("Error: BRAVE_API_KEY is not set in .env")
            return
        
        deps = Deps(client=client, brave_api_key=brave_api_key)
        film_name = Prompt.ask("What film would you like to search for?")
        
       
        try:
            results = await web_search_agent.run(f"Find me financial summary of film {film_name}", deps=deps)
            print("Search Results:")
            print(results.output)
          
        
        except Exception as e:
            print(f"Error in web search: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())