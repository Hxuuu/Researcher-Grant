#!/usr/bin/env python3
"""
Equity Research AI Agent - Interactive CLI

Usage:
    python main.py                    # Interactive chat mode
    python main.py --ticker AAPL      # Quick analysis of a specific stock
    python main.py --compare AAPL MSFT GOOGL  # Compare multiple stocks
    python main.py --query "..."      # Run a single research query
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

load_dotenv()

console = Console()


def print_banner():
    console.print(
        Panel.fit(
            "[bold blue]Equity Research AI Agent[/bold blue]\n"
            "[dim]Powered by Claude claude-opus-4-6 + Real-Time Market Data[/dim]\n\n"
            "[yellow]Commands:[/yellow]\n"
            "  [green]analyze <TICKER>[/green]   - Full stock analysis\n"
            "  [green]compare <T1> <T2> ...[/green] - Compare stocks\n"
            "  [green]news <TICKER>[/green]       - Latest company news\n"
            "  [green]reset[/green]               - Start new research session\n"
            "  [green]quit[/green] / [green]exit[/green]         - Exit\n\n"
            "[dim]Or ask any research question directly![/dim]",
            title="[bold]Welcome[/bold]",
            border_style="blue",
        )
    )


def stream_response(agent, query: str) -> None:
    """Stream agent response to console with rich formatting."""
    console.print(Rule("[dim]Research Analysis[/dim]", style="dim"))

    response_text = []

    # Stream output character by character using a live display
    with console.status("[bold green]Researching...[/bold green]", spinner="dots") as status:
        buffer = ""
        first_chunk = True

        for chunk in agent.research(query):
            if first_chunk:
                status.stop()
                first_chunk = False

            # Check if this is a status message (italic)
            if chunk.startswith("\n*[") or chunk.startswith("*["):
                # Flush buffer if any
                if buffer:
                    console.print(buffer, end="", highlight=False)
                    response_text.append(buffer)
                    buffer = ""
                # Print status in dim style
                console.print(chunk, style="dim yellow", end="")
                response_text.append(chunk)
            else:
                buffer += chunk
                response_text.append(chunk)

    # Print any remaining buffer
    if buffer:
        # Try to render as markdown
        try:
            console.print(Markdown(buffer))
        except Exception:
            console.print(buffer)

    console.print()


def run_interactive(agent) -> None:
    """Run interactive chat mode."""
    print_banner()
    console.print("\n[dim]Type your question or command. Use 'quit' to exit.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye! Happy investing![/yellow]")
            break

        if lower == "reset":
            agent.reset()
            console.print("[green]Conversation reset. Starting fresh session.[/green]")
            continue

        # Parse shortcut commands
        if lower.startswith("analyze "):
            ticker = user_input.split(None, 1)[1].strip().upper()
            query = f"Please provide a comprehensive equity research analysis of {ticker}. Cover the business overview, recent financial performance, key metrics, valuation, analyst estimates, recent news, and your overall assessment."
        elif lower.startswith("compare "):
            tickers = user_input.split()[1:]
            ticker_str = ", ".join(t.upper() for t in tickers)
            query = f"Please compare these stocks side by side: {ticker_str}. Cover valuation, growth, profitability, and which looks most attractive."
        elif lower.startswith("news "):
            ticker = user_input.split(None, 1)[1].strip().upper()
            query = f"What is the latest news about {ticker}? Summarize the key developments and their potential impact on the stock."
        else:
            query = user_input

        try:
            stream_response(agent, query)
        except anthropic.AuthenticationError:
            console.print("[red]Error: Invalid API key. Please set ANTHROPIC_API_KEY.[/red]")
            break
        except anthropic.RateLimitError:
            console.print("[red]Rate limited. Please wait a moment and try again.[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def run_single_query(agent, query: str) -> None:
    """Run a single query and print the result."""
    console.print(f"\n[dim]Query:[/dim] {query}\n")

    buffer = ""
    for chunk in agent.research(query):
        if chunk.startswith("\n*[") or chunk.startswith("*["):
            if buffer:
                print(buffer, end="", flush=True)
                buffer = ""
            print(chunk, end="", flush=True)
        else:
            buffer += chunk

    if buffer:
        print(buffer, flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Equity Research AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker", "-t", help="Analyze a specific stock ticker")
    parser.add_argument(
        "--compare",
        "-c",
        nargs="+",
        metavar="TICKER",
        help="Compare multiple stock tickers",
    )
    parser.add_argument("--query", "-q", help="Run a single research query")
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Claude model to use (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (print complete response at end)",
    )

    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[red]Error: ANTHROPIC_API_KEY environment variable not set.[/red]\n"
            "Set it with: [yellow]export ANTHROPIC_API_KEY=your_key_here[/yellow]\n"
            "Or create a [yellow].env[/yellow] file with: [yellow]ANTHROPIC_API_KEY=your_key_here[/yellow]"
        )
        sys.exit(1)

    from equity_agent.agent import EquityResearchAgent

    agent = EquityResearchAgent(api_key=api_key, model=args.model)

    if args.ticker:
        ticker = args.ticker.upper()
        query = (
            f"Please provide a comprehensive equity research analysis of {ticker}. "
            "Cover the business overview, recent financial performance, key metrics, "
            "valuation, analyst estimates, recent news, and your overall assessment with bull/bear case."
        )
        run_single_query(agent, query)
    elif args.compare:
        tickers = [t.upper() for t in args.compare]
        ticker_str = ", ".join(tickers)
        query = (
            f"Please compare these stocks side by side: {ticker_str}. "
            "Cover valuation multiples, growth rates, profitability, financial health, "
            "and provide a relative attractiveness ranking."
        )
        run_single_query(agent, query)
    elif args.query:
        run_single_query(agent, args.query)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    import anthropic
    main()
