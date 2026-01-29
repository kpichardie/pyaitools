---

# Gemini CLI Tool

A lightweight Python command-line interface for interacting with Google's Gemini AI models. This tool supports streaming responses, system instructions, and piping input from other command-line utilities.

## Features

- **Streaming Support**: Real-time text output by default.
- **Piping & Redirection**: Seamlessly integrates with Unix pipes (e.g., `cat file.txt | askgemini "summarize"`).
- **Customizable**: Easily switch models, adjust temperature, and set system personas.
- **Clean Logging**: Status messages are sent to `stderr`, ensuring that redirected output contains only the AI's response.

## Prerequisites

- Python 3.9 or higher
- A Google Gemini API Key (Get one at [Google AI Studio](https://aistudio.google.com/))

## Installation

### Option 1: Local Setup
1. **Install the required library**:
   ```bash
   pip install google-genai
   ```

2. **Set your API Key**:
   **macOS/Linux:** `export GEMINI_API_KEY='your_key'`
   **Windows:** `$env:GEMINI_API_KEY='your_key'`

3. **(Optional) Create an alias**:
   To use `askgemini` instead of `python script.py`, add this to your `.bashrc` or `.zshrc`:
   ```bash
   alias askgemini='python /path/to/your/script.py'
   ```

### Option 2: Docker
1. **Build the image**:
   ```bash
   docker build -t gemini-tool .
   ```
2. **Run via Docker**:
   ```bash
   cat myapp.py | docker run -i -e GEMINI_API_KEY=$GEMINI_API_KEY gemini-tool "how can i improve this?"
   ```

## Usage

### Basic Query
```bash
askgemini "What are the three laws of robotics?"
```

### Using System Instructions
Set a persona or specific constraints for the AI:
```bash
askgemini "Explain quantum physics" --system "You are a pirate who loves science."
```

### Piping Input (The Power User Way)
You can pipe file contents or output from other commands directly into the script:
```bash
cat main.py | askgemini "Refactor this code for better readability"
```

### Advanced Options
```bash
# Disable streaming
askgemini "Explain Kubernetes" --no-stream

# Use a specific model and temperature
askgemini "Write a creative story" --model "gemini-1.5-pro" --temp 0.9

# Save output to a file
askgemini "Generate a README.md for a Python project" > README.md
```

## Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `prompt` | The text prompt to send to the AI. | Required (or stdin) |
| `--model` | The Gemini model ID to use. | `gemini-2.0-flash` |
| `--system` | System instruction or persona for the AI. | None |
| `--temp` | Controls randomness (0.0 to 2.0). | `0.7` |
| `--no-stream` | Disable real-time streaming. | `False` |

## Error Handling

- **API Key Missing**: The script will exit with an error if `GEMINI_API_KEY` is not found in your environment.
- **Network Errors**: Google API errors are caught and logged to `stderr` so they don't corrupt your piped data.
- **Interruption**: Use `Ctrl+C` to safely stop a streaming response.

## License
[MIT](https://opensource.org/licenses/MIT)

