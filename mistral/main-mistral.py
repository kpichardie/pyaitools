import sys
import os
import argparse
from mistralai import Mistral
from dotenv import load_dotenv
from typing import List, Optional, Any

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("Error: The variable MISTRAL_API_KEY is missing.")
    sys.exit(1)

DEFAULT_MODEL = "mistral-tiny"
DEFAULT_TEMPERATURE = 0.7

class Args:
    def __init__(
        self,
        prompt: List[str],
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        output_file: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.output_file = output_file
        self.max_length = max_length

def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt", nargs="*", help="The prompt to send to Mistral"
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, help="Mistral model to use"
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature (0.0 to 1.0)"
    )
    parser.add_argument("-f", "--output-file", help="Save the response to a file")
    parser.add_argument(
        "-l", "--max-length", type=int, help="Maximum number of tokens to generate"
    )
    args = parser.parse_args()
    return Args(
        prompt=args.prompt,
        model=args.model,
        temperature=args.temperature,
        output_file=args.output_file,
        max_length=args.max_length,
    )

def get_input_from_stdin() -> str:
    return sys.stdin.read() if not sys.stdin.isatty() else ""

def main() -> Any:
    args = parse_args()
    if not args.prompt:
        print("Error: No input provided.")
        sys.exit(0)

    client = Mistral(api_key=MISTRAL_API_KEY)
    input_from_stdin = get_input_from_stdin()
    user_prompt = " ".join(args.prompt)
    full_content = f"{input_from_stdin}\n\n{user_prompt}"

    try:
        chat_response = client.chat.complete(
            model=args.model,
            messages=[{"role": "user", "content": full_content}],
            temperature=args.temperature,
            max_tokens=args.max_length,
        )
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(chat_response.choices[0].message.content)
        else:
            print(chat_response.choices[0].message.content)
    except Exception as e:
        print(f"Error when calling Mistral: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
