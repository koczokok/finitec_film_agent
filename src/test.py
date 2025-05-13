import os
from dotenv import load_dotenv
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load environment variables from .env file
load_dotenv()


class CityLocation(BaseModel):
    city: str
    country: str


ollama_model = OpenAIModel(
    model_name='llama3.2', 
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'), 
    system_prompt_role=(
        'You are a helpful assistant that provides precise, '
        'structured information about Olympic host cities.'
    ),
)
agent = Agent(ollama_model, output_type=CityLocation)

result = agent.run_sync('Where were the 2024 Summer Olympics held? Provide the city, country, and whether it is a capital city.')
print(result.output)
