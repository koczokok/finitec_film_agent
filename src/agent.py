from dataclasses import dataclass
from typing import Any
from httpx import AsyncClient
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
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
import wptools
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

wiki = Wikipedia(user_agent="Film_Agent (p.kurylowicz@outlook.com)", language="en")
llm = os.getenv("LLM_MODEL")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
movie_folder = Path("./movies")
files = list(movie_folder.glob("*.md"))

file_texts = [f.read_text(encoding="utf-8") for f in files]
embeddings = sentence_model.encode(file_texts)

# Ollama configuration
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

# OpenAI/ChatGPT-4 configuration
# client = AsyncOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# model = OpenAIModel(
#     model_name="gpt-4", 
#     provider=OpenAIProvider(
#         api_key=os.getenv("OPENAI_API_KEY")
#     )
# )
logfire_token = os.getenv("LOGFIRE_TOKEN", "")
logfire.configure(token=logfire_token)
logfire.info('Hello, {place}!', place='World')
logfire.instrument_openai()

class Film(BaseModel):
    name: str
    budget: int = Field(default=0, description="Budget in million USD")
    box_office: int = Field(default=0, description="Box Office in million USD")
    

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
    Box Office: [Integer amount in USD]
    ```
    Guidelines:
    1. Use web search to find the film financial data
    2. Always return numeric values
    3. Look only for Budget, Box Office.
    4. If no exact data found, use best available estimate
    5. If no data available, return zeros""",
    output_type=Film,
    deps_type=Deps,
    retries=2,
)


# @web_search_agent.tool
# async def get_wikidata(ctx: RunContext[Deps], film_name: str) -> str:
#     """
#     This is your secondary tool use it if film_file_search fails and there is no file corresponding to the film name.

#     """
#     page = wptools.page(film_name)
#     page.get_parse()  
#     infobox = page.data.get("infobox")
    
#     return infobox




# @web_search_agent.tool
# async def film_file_search(ctx: RunContext[Deps], film_name: str) -> str:
#     """
#     This is your main tool use it first
#     IMPORTANT: ONLY output the following fields with numeric values:
# Name, Budget, Domestic Box Office, International Box Office, Worldwide Box Office, DVD Sales, Total Earnings, Profit, Tool Name.

# Do NOT include any plot summaries, cast info, or other text.

# If you cannot find a value, return 0 for that field.
#     """
#     query_embedding = sentence_model.encode([film_name])
#     scores = cosine_similarity(query_embedding, embeddings)[0]
#     best_idx = int(np.argmax(scores))
#     best_file = files[best_idx]
#     best_content = file_texts[best_idx]

#     for line in best_content.splitlines():
#         if line.lower().startswith("name:"):
#             extracted_title = line.split(":", 1)[1].strip()
#             break
#     else:
#         extracted_title = best_file.stem

#     ratio = SequenceMatcher(None, film_name.lower(), extracted_title.lower()).ratio()
#     if ratio < 0.75:
#         # Attempt Wikipedia fallback
#         wiki_result = await get_wikidata(ctx, film_name=film_name)
#         # if wiki_result.strip().startswith("Wikipedia page not found"):
#         #     web_result = await web_search(ctx, web_query=film_name)
#         #     return f"(Fallback to web - closest local file was '{extracted_title}', which doesn't match)\n\n{web_result}"
#         # return f"(Fallback to Wikipedia - closest local file was '{extracted_title}', which doesn't match)\n\n{wiki_result}"

#     return  {best_content}



@web_search_agent.tool
async def web_search(ctx: RunContext[Deps], web_query: str) -> str:
    """ Using this tool extract financial data of a film from the web."""
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
            "q": web_query + " financial details box office budget",
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
        # film_name = Prompt.ask("What film would you like to search for?")
        
       
        try:
            results = await web_search_agent.run(f"Find me financial summary of film Inception", deps=deps)
            print("Search Results:")
            print(results.output)
          
        
        except Exception as e:
            print(f"Error in web search: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())