from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, CreateMessageResult, CreateMessageRequestParams
import json
import uuid
import asyncio
from typing import List, Dict, Any
import re
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import refine_script
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mcp = FastMCP("Golden Rule Server")

# Pre-load patterns and embeddings from refine_script.
patterns = refine_script.load_patterns()
embeddings = refine_script.compute_category_embeddings(patterns)

@mcp.tool()
async def refine_text(text: str) -> str:
    top_categories = refine_script.find_top_patterns(text, embeddings)
    prompt = refine_script.generate_prompt(text, top_categories, patterns)
    return refine_script.refine_text_with_gpt4(prompt)

async def handle_sampling(message: CreateMessageRequestParams) -> CreateMessageResult:
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text="サンプリング結果"),
        model="gpt-4",
        stopReason="endTurn"
    )

mcp.sampling_callback = handle_sampling

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MCP server on http://192.168.11.3:8000/sse")
    uvicorn.run(mcp.sse_app(), host="192.168.11.3", port=8000)
