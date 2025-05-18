
from dotenv import load_dotenv
from typing import List
import asyncio
import logfire
import httpx
import os

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from agent import web_search_agent, Deps


load_dotenv()

logfire.configure(send_to_logfire='never')

class CLI:
    def __init__(self):
        self.messages: List[ModelMessage] = []
        self.deps = Deps(client=httpx.AsyncClient(), brave_api_key=os.getenv("BRAVE_API_KEY"))

    async def chat(self):
        print("Film Financial Data Agent CLI (type 'quit' to exit)")
        print("Enter name of the film to receive its financial data:")

        try:
            while True:
                user_input = input("> ").strip()
                if user_input.lower() == 'quit':
                    break

                result = await web_search_agent.run(
                    f"Find me financial summary of film {user_input}",
                    deps=self.deps,
                    message_history=self.messages
                )


                self.messages.append(
                    ModelRequest(parts=[UserPromptPart(content=user_input)])
                )

              
                filtered_messages = [msg for msg in result.new_messages() 
                                if not (hasattr(msg, 'parts') and 
                                        any(part.part_kind == 'user-prompt' or part.part_kind == 'text' for part in msg.parts))]
                self.messages.extend(filtered_messages)

                print(result.output)

                self.messages.append(
                    ModelResponse(parts=[TextPart(content=result.output.model_dump_json())])
                )
        finally:
            await self.deps.client.aclose()

async def main():
    cli = CLI()
    await cli.chat()

if __name__ == "__main__":
    asyncio.run(main())