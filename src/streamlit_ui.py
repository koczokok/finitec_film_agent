from __future__ import annotations
import streamlit_patch
import asyncio
import os
from typing import Literal, TypedDict

import streamlit as st
from dotenv import load_dotenv
from httpx import AsyncClient
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
os.environ["STREAMLIT_WATCHDOG_USE_POLLING"] = "true"
os.environ["STREAMLIT_WATCH_DIRECTORIES"] = "D:/WORK/ML/Finitec/Film/src"

from agent import web_search_agent, Deps

load_dotenv()

model = OpenAIModel(
    model_name="o4-mini",
    provider=OpenAIProvider(api_key=os.getenv("OPEN_API_KEY"))
)

class ChatMessage(TypedDict):
    role: Literal["user", "model"]
    timestamp: str
    content: str

def display_message(message):
    role = "user" if isinstance(message, ModelRequest) else "assistant"
    for part in message.parts:
        if isinstance(part, (UserPromptPart, TextPart)):
            with st.chat_message(role):
                st.markdown(part.content)

async def run_agent(user_input: str):
    async with AsyncClient() as client:
        deps = Deps(client=client, brave_api_key=os.getenv("BRAVE_API_KEY"))
        result = await web_search_agent.run(
            user_input,
            deps=deps,
            message_history=[msg for msg in st.session_state.messages if isinstance(msg, ModelRequest)]
        )
        film = result.output
        return (
            f"**Film:** {film.name}\n\n"
            f"**Budget (mln USD):** {film.budget}\n\n"
            f"**Profit (mln USD):** {film.profit}\n\n"
            f"**Data source:** {film.tool_name}"
        )

async def main():
    st.set_page_config(page_title="Film Financial Extractor")
    st.title("Film Financial Extractor")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        display_message(msg)

    user_input = st.chat_input("Enter a film title to receive its financial data")

    if user_input:
        user_msg = ModelRequest(parts=[UserPromptPart(content=user_input)])
        st.session_state.messages.append(user_msg)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            assistant_reply = await run_agent(user_input)
            st.markdown(assistant_reply)

        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=assistant_reply)])
        )

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
