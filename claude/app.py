"""
AI Query Tool - Supporting Multiple Free & Paid APIs
Includes Claude (paid), OpenRouter (free tier), and Groq (free)
"""

import os
import requests
from anthropic import Anthropic

class AIQueryTool:
    def __init__(self, provider="claude", api_key=None):
        """
        Initialize the AI Query Tool

        Args:
            provider (str): API provider - "claude", "openrouter", or "groq"
            api_key (str, optional): API key for the provider
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.conversation_history = []

        if self.provider == "claude":
            if not self.api_key:
                raise ValueError("Claude requires ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=self.api_key)

    def _get_api_key(self):
        """Get API key from environment based on provider"""
        env_vars = {
            "claude": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY"
        }
        return os.environ.get(env_vars.get(self.provider))

    def query(self, prompt, model=None, max_tokens=1024, temperature=1.0,
              system_prompt=None):
        """
        Send a query to the AI provider

        Args:
            prompt (str): The user's question or prompt
            model (str, optional): Model to use (provider-specific)
            max_tokens (int): Maximum tokens in response
            temperature (float): Response randomness (0-1)
            system_prompt (str, optional): System instructions

        Returns:
            str: AI response
        """
        if self.provider == "claude":
            return self._query_claude(prompt, model, max_tokens, temperature, system_prompt)
        elif self.provider == "openrouter":
            return self._query_openrouter(prompt, model, max_tokens, temperature, system_prompt)
        elif self.provider == "groq":
            return self._query_groq(prompt, model, max_tokens, temperature, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _query_claude(self, prompt, model, max_tokens, temperature, system_prompt):
        """Query Claude API (PAID)"""
        model = model or "claude-sonnet-4-20250514"

        params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system_prompt:
            params["system"] = system_prompt

        response = self.client.messages.create(**params)
        return response.content[0].text

    def _query_openrouter(self, prompt, model, max_tokens, temperature, system_prompt):
        """Query OpenRouter API (Has FREE models)"""
        # Default to a free model
        model = model or "meta-llama/llama-3.2-3b-instruct:free"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _query_groq(self, prompt, model, max_tokens, temperature, system_prompt):
        """Query Groq API (FREE with limits)"""
        # Default to a free fast model
        model = model or "llama-3.3-70b-versatile"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def chat(self, user_message, model=None, max_tokens=1024,
             temperature=1.0, system_prompt=None):
        """
        Have a conversation (maintains history)
        Note: History tracking works best with Claude
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # For simplicity, use query method
        response = self.query(user_message, model, max_tokens, temperature, system_prompt)

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_history(self):
        """Get conversation history"""
        return self.conversation_history

# Example usage
if __name__ == "__main__":
    print("=== AI Query Tool Demo ===\n")

    # Example 1: Groq (FREE)
    print("1. Using Groq (FREE - https://console.groq.com)")
    try:
        groq = AIQueryTool(provider="groq")
        response = groq.query("What is Python? Answer in one sentence.")
        print(f"Groq: {response}\n")
    except Exception as e:
        print(f"Groq Error: {e}")
        print("Get free API key at: https://console.groq.com\n")

    # Example 2: OpenRouter (FREE tier available)
    print("2. Using OpenRouter FREE model (https://openrouter.ai)")
    try:
        openrouter = AIQueryTool(provider="openrouter")
        response = openrouter.query("What is Docker? Answer in one sentence.")
        print(f"OpenRouter: {response}\n")
    except Exception as e:
        print(f"OpenRouter Error: {e}")
        print("Get free API key at: https://openrouter.ai/keys\n")

    # Example 3: Claude (PAID but has free credits)
    print("3. Using Claude (PAID - but $5 free credits)")
    try:
        claude = AIQueryTool(provider="claude")
        response = claude.query("What is Kubernetes? Answer in one sentence.")
        print(f"Claude: {response}\n")
    except Exception as e:
        print(f"Claude Error: {e}")
        print("Get API key at: https://console.anthropic.com\n")

    print("\nSetup Instructions:")
    print("- Groq (FREE): export GROQ_API_KEY='your-key' - https://console.groq.com")
    print("- OpenRouter (FREE tier): export OPENROUTER_API_KEY='your-key' - https://openrouter.ai")
    print("- Claude ($5 free): export ANTHROPIC_API_KEY='your-key' - https://console.anthropic.com")
