"""
Equity Research AI Agent powered by Claude.

Uses an agentic loop with tool use to research stocks and generate insights.
"""

import json
import os
from typing import Iterator

import anthropic
import httpx

from .tools import TOOL_DEFINITIONS, execute_tool


def _make_client(api_key: str | None = None) -> anthropic.Anthropic:
    """
    Create an Anthropic client, using Bearer auth with the Claude Code
    session token when running inside a Claude Code remote environment.
    """
    # Prefer explicit key
    if api_key:
        return anthropic.Anthropic(api_key=api_key)

    # Standard env var
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()

    # Claude Code remote environment: use session ingress token as Bearer
    session_token_path = "/home/claude/.claude/remote/.session_ingress_token"
    if os.path.exists(session_token_path):
        with open(session_token_path) as f:
            token = f.read().strip()

        def _swap_auth(request: httpx.Request) -> None:
            request.headers["Authorization"] = f"Bearer {token}"
            # Remove x-api-key so the Bearer header is used instead
            if "x-api-key" in request.headers:
                del request.headers["x-api-key"]

        http_client = httpx.Client(event_hooks={"request": [_swap_auth]})
        return anthropic.Anthropic(api_key="placeholder", http_client=http_client)

    raise ValueError(
        "No API key found. Set ANTHROPIC_API_KEY or run inside Claude Code."
    )

SYSTEM_PROMPT = """You are an expert equity research analyst with deep knowledge of financial markets,
valuation methodologies, and investment analysis. Your job is to help users research stocks and make
informed investment decisions.

You have access to real-time market data tools that let you:
- Fetch stock prices and price history
- Pull financial statements (income statement, balance sheet, cash flow)
- Get key valuation and fundamental metrics (P/E, P/B, EV/EBITDA, margins, growth rates)
- Access analyst estimates and upgrade/downgrade history
- Read recent company news
- Compare multiple stocks side by side
- Get company overview and sector info

When analyzing stocks, follow this methodology:
1. **Business Understanding**: What does the company do? What sector/industry?
2. **Financial Performance**: Revenue trends, profitability, margins, cash generation
3. **Valuation**: How does the stock trade vs. peers and historical norms?
4. **Growth**: What are the growth drivers? What do analysts expect?
5. **Risk Factors**: Debt levels, competitive threats, macro risks
6. **Sentiment**: Recent news, analyst ratings, price targets

Always:
- Use actual data from the tools before drawing conclusions
- Cite specific numbers and ratios in your analysis
- Provide balanced views (bull and bear case)
- Note limitations and risks
- Format numbers clearly (e.g., "$1.2B revenue", "25.3x P/E", "+12.5% YoY growth")

Important disclaimer: This is for educational and research purposes only.
Nothing here constitutes personalized financial advice."""


class EquityResearchAgent:
    """
    An AI-powered equity research agent that uses Claude with tool use
    to fetch real market data and provide comprehensive stock analysis.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-opus-4-6"):
        self.client = _make_client(api_key)
        self.model = model
        self.conversation_history: list[dict] = []

    def reset(self) -> None:
        """Reset conversation history to start a new research session."""
        self.conversation_history = []

    def research(self, query: str) -> Iterator[str]:
        """
        Process a research query and stream the response.

        Args:
            query: The research question or stock to analyze

        Yields:
            Text chunks of the streamed response
        """
        self.conversation_history.append({"role": "user", "content": query})

        while True:
            # Stream the response
            text_buffer = []
            response_content = []
            stop_reason = None

            with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=self.conversation_history,
                thinking={"type": "adaptive"},
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            chunk = event.delta.text
                            text_buffer.append(chunk)
                            yield chunk

                final_msg = stream.get_final_message()
                response_content = final_msg.content
                stop_reason = final_msg.stop_reason

            # Append assistant response to history
            self.conversation_history.append(
                {"role": "assistant", "content": response_content}
            )

            # If done with tool calls, break
            if stop_reason == "end_turn":
                break

            # Handle tool calls
            if stop_reason == "tool_use":
                tool_use_blocks = [b for b in response_content if b.type == "tool_use"]

                if not tool_use_blocks:
                    break

                # Execute each tool and collect results
                yield "\n"
                tool_results = []
                for tool_call in tool_use_blocks:
                    yield f"\n*[Fetching {tool_call.name}({json.dumps(tool_call.input)})...]*\n"

                    result = execute_tool(tool_call.name, tool_call.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": result,
                    })

                # Add tool results to conversation
                self.conversation_history.append(
                    {"role": "user", "content": tool_results}
                )
                # Continue the loop so Claude can process the tool results
                continue

            # Any other stop reason — exit
            break

    def research_sync(self, query: str) -> str:
        """
        Non-streaming version: run a research query and return the complete response.

        Args:
            query: The research question or stock to analyze

        Returns:
            Complete response text
        """
        return "".join(self.research(query))
