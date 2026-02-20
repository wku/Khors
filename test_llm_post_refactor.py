import json
from khors.llm import fetch_openrouter_pricing

print("Fetching pricing...")
pricing = fetch_openrouter_pricing()
print(f"Prices loaded: {len(pricing)}")
if pricing:
    model, cost = list(pricing.items())[0]
    print(f"Example: {model} -> {cost}")

