import os
import sys
import argparse
import logging
from google import genai
from google.genai import errors, types

# Configure logging to stderr so it doesn't interfere with the text output
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Query Google Gemini AI")
    parser.add_argument("prompt", nargs="*", help="The prompt to send to the AI")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model to use")
    parser.add_argument("--system", type=str, help="System instruction/persona")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature (0.0 to 2.0)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    return parser.parse_args()

def get_api_key() -> str:
    """Retrieve API key or raise an error."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Environment variable 'GEMINI_API_KEY' is not set.")
    return api_key

def generate_content(client: genai.Client, model: str, prompt: str, system: str, temp: float,stream: bool = True):
    """Generate content from Gemini."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temp,
    )

    try:
        if stream:
            response = client.models.generate_content_stream(model=model, contents=prompt, config=config)
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print() # New line at end
        else:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            if response.text:
                print(response.text)

    except errors.ClientError as e:
        logger.error(f"Google API Client Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

def main():
    args = get_args()

    # Check if input is being piped in
    input_data = ""
    if not sys.stdin.isatty():
        input_data = sys.stdin.read().strip()

    arg_prompt = " ".join(args.prompt).strip()

    # Combine them (useful for: cat code.py | gemini "refactor this")
    full_prompt = f"{input_data}\n\n{arg_prompt}".strip()

    if not full_prompt:
        logger.error("No prompt provided via arguments or stdin.")
        sys.exit(1)

    try:
        client = genai.Client(api_key=get_api_key())

        # Display model info only if not piping output to another tool
        if sys.stdout.isatty():
            logger.info(f"Model: {args.model} | Temp: {args.temp}")
        generate_content(
            client=client,
            model=args.model,
            prompt=full_prompt,
            system=args.system,
            temp=args.temp,
            stream=not args.no_stream
        )

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except ValueError as ve:
        logger.error(ve)
        sys.exit(1)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()

