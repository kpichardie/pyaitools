import os
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

# --- CONFIGURATION ---
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# Keywords to search for in the model list
SEARCH_TERMS = ["gemini", "deepseek", "kimi", "moonshot", ":free"]

console = Console()

def get_openrouter_credits():
    """Fetches the wallet balance."""
    if not OPENROUTER_KEY:
        return None
    try:
        res = requests.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {OPENROUTER_KEY}"}, timeout=5)
        if res.status_code == 200:
            data = res.json().get("data", {})
            return data.get("total_credits", 0.0), data.get("total_usage", 0.0)
    except:
        return None
    return 0.0, 0.0

def get_filtered_models():
    """Fetches ALL models and filters them dynamically."""
    try:
        console.print("[dim]Fetching live model list from OpenRouter...[/dim]")
        res = requests.get("https://openrouter.ai/api/v1/models", timeout=10)

        if res.status_code != 200:
            return []

        all_models = res.json().get("data", [])
        filtered_models = []

        for m in all_models:
            mid = m.get("id", "").lower()

            # Check if it matches our search terms
            if any(term in mid for term in SEARCH_TERMS):
                pricing = m.get("pricing", {})

                # Calculate costs (OpenRouter returns strings usually)
                try:
                    in_cost = float(pricing.get("prompt", 0)) * 1_000_000
                    out_cost = float(pricing.get("completion", 0)) * 1_000_000
                except (ValueError, TypeError):
                    in_cost, out_cost = -1, -1

                # Determine status
                is_free_id = ":free" in mid
                is_zero_cost = (in_cost == 0 and out_cost == 0)

                # Clean up the ID for display (remove provider prefix if it's too long)
                display_name = mid.split("/")[-1] if "/" in mid else mid

                filtered_models.append({
                    "id": mid,
                    "name": display_name,
                    "in_cost": in_cost,
                    "out_cost": out_cost,
                    "context": m.get("context_length", 0),
                    "is_free": is_free_id or is_zero_cost
                })

        # Sort: Free models first, then by name
        return sorted(filtered_models, key=lambda x: (not x["is_free"], x["name"]))

    except Exception as e:
        console.print(f"[red]Error fetching models: {e}[/red]")
        return []

def generate_dashboard():
    # 1. Fetch Data
    credits_data = get_openrouter_credits()
    models = get_filtered_models()

    # 2. Header Section
    if credits_data:
        balance, usage = credits_data
        # If API key is present but balance is 0, it might be a valid 0 balance.
        header_text = f"💰 Balance: ${balance:.4f}  |  📉 Total Used: ${usage:.4f}"
        border_style = "green"
    else:
        header_text = "⚠️  OPENROUTER_API_KEY missing or invalid"
        border_style = "red"

    console.print(Panel(Text(header_text, justify="center", style="bold white"), style=border_style))
    console.print("") # spacer

    # 3. Table Section
    table = Table(box=box.SIMPLE, expand=True, show_lines=False)

    # Compact Columns
    table.add_column("Model ID", style="cyan", ratio=3, overflow="fold")
    table.add_column("Input\n($/1M)", justify="right", style="dim", ratio=1)
    table.add_column("Output\n($/1M)", justify="right", style="dim", ratio=1)
    table.add_column("Ctx", justify="right", ratio=1) # Context
    table.add_column("Status", justify="center", ratio=1)

    for m in models:
        # Format costs
        if m["in_cost"] < 0:
            c_in, c_out = "?", "?"
        else:
            c_in = f"{m['in_cost']:.2f}" if m['in_cost'] > 0 else "0"
            c_out = f"{m['out_cost']:.2f}" if m['out_cost'] > 0 else "0"

        # Format Context (e.g., 100000 -> 100k)
        ctx = m['context']
        if ctx >= 1_000_000:
            ctx_str = f"{ctx/1_000_000:.1f}M"
        elif ctx >= 1000:
            ctx_str = f"{ctx/1000:.0f}k"
        else:
            ctx_str = str(ctx)

        # Status Icon
        if m["is_free"]:
            status = "[bold green]FREE[/]"
        elif "moonshot" in m["id"] or "kimi" in m["id"]:
             # Kimi is rarely free via API, usually paid
            status = "[yellow]PAY[/]"
        else:
            status = "[dim]PAY[/]"

        table.add_row(
            m["id"],
            c_in,
            c_out,
            ctx_str,
            status
        )

    console.print(table)

if __name__ == "__main__":
    generate_dashboard()
