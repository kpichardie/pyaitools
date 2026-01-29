#!/bin/bash

# AI Query Tool - Bash Functions
# Add these to your ~/.bashrc or ~/.zshrc

# Simple query function (uses Groq by default)
ai() {
    local prompt=""

    # Check if input is piped
    if [ -p /dev/stdin ]; then
        local piped_input=$(cat)
        prompt="$* $piped_input"
    else
        prompt="$*"
    fi

    if [ -z "$prompt" ]; then
        echo "Usage: ai <your question>"
        echo "       echo 'data' | ai <your question>"
        echo "Example: ai what is docker"
        echo "Example: cat error.log | ai explain this error"
        return 1
    fi

    # Escape quotes in prompt for JSON
    prompt=$(echo "$prompt" | sed 's/"/\\"/g')

    curl -s -X POST http://localhost:5000/query \
        -H "Content-Type: application/json" \
        -d "{\"provider\": \"groq\", \"prompt\": \"$prompt\"}" \
        | jq -r '.response'
}

# Query with specific provider
ai_provider() {
    local provider="$1"
    shift
    local prompt=""

    # Check if input is piped
    if [ -p /dev/stdin ]; then
        local piped_input=$(cat)
        prompt="$* $piped_input"
    else
        prompt="$*"
    fi

    if [ -z "$provider" ] || [ -z "$prompt" ]; then
        echo "Usage: ai_provider <provider> <your question>"
        echo "       echo 'data' | ai_provider <provider> <your question>"
        echo "Providers: groq, openrouter, claude"
        echo "Example: ai_provider claude explain kubernetes"
        echo "Example: cat logs.txt | ai_provider groq analyze these logs"
        return 1
    fi

    # Escape quotes in prompt for JSON
    prompt=$(echo "$prompt" | sed 's/"/\\"/g')

    curl -s -X POST http://localhost:5000/query \
        -H "Content-Type: application/json" \
        -d "{\"provider\": \"$provider\", \"prompt\": \"$prompt\"}" \
        | jq -r '.response'
}

# Query with system prompt
ai_expert() {
    local expert="$1"
    shift
    local prompt=""

    # Check if input is piped
    if [ -p /dev/stdin ]; then
        local piped_input=$(cat)
        prompt="$* $piped_input"
    else
        prompt="$*"
    fi

    if [ -z "$expert" ] || [ -z "$prompt" ]; then
        echo "Usage: ai_expert <expert_type> <your question>"
        echo "       echo 'data' | ai_expert <expert_type> <your question>"
        echo "Example: ai_expert python how do I read a file"
        echo "Example: cat code.py | ai_expert python review this code"
        return 1
    fi

    local system_prompt="You are an expert in ${expert}. Provide clear, concise answers."

    # Escape quotes in prompt for JSON
    prompt=$(echo "$prompt" | sed 's/"/\\"/g')

    curl -s -X POST http://localhost:5000/query \
        -H "Content-Type: application/json" \
        -d "{
            \"provider\": \"groq\",
            \"prompt\": \"$prompt\",
            \"system_prompt\": \"$system_prompt\"
        }" \
        | jq -r '.response'
}

# List available providers
ai_providers() {
    curl -s http://localhost:5000/providers | jq '.'
}

# Health check
ai_health() {
    curl -s http://localhost:5000/health | jq '.'
}

# Advanced query with all options
ai_advanced() {
    local json_payload="$1"

    if [ -z "$json_payload" ]; then
        echo "Usage: ai_advanced '<JSON payload>'"
        echo "Example: ai_advanced '{\"provider\":\"groq\",\"prompt\":\"test\",\"max_tokens\":500}'"
        return 1
    fi

    curl -s -X POST http://localhost:5000/query \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        | jq -r '.response'
}

# Quick code helper
aicode() {
    local language="$1"
    shift
    local question=""

    # Check if input is piped
    if [ -p /dev/stdin ]; then
        local piped_input=$(cat)
        question="$* $piped_input"
    else
        question="$*"
    fi

    if [ -z "$language" ] || [ -z "$question" ]; then
        echo "Usage: aicode <language> <question>"
        echo "       echo 'code' | aicode <language> <question>"
        echo "Example: aicode python how to read json file"
        echo "Example: cat script.py | aicode python review and optimize this"
        return 1
    fi

    local system_prompt="You are an expert ${language} programmer. Provide code examples and clear explanations."

    # Escape quotes in question for JSON
    question=$(echo "$question" | sed 's/"/\\"/g')

    curl -s -X POST http://localhost:5000/query \
        -H "Content-Type: application/json" \
        -d "{
            \"provider\": \"groq\",
            \"prompt\": \"$question\",
            \"system_prompt\": \"$system_prompt\"
        }" \
        | jq -r '.response'
}

# DevOps helper
aidevops() {
    local prompt=""

    # Check if input is piped
    if [ -p /dev/stdin ]; then
        local piped_input=$(cat)
        prompt="$* $piped_input"
    else
        prompt="$*"
    fi

    if [ -z "$prompt" ]; then
        echo "Usage: aidevops <question>"
        echo "       echo 'data' | aidevops <question>"
        echo "Example: aidevops how to debug kubernetes pod"
        echo "Example: kubectl logs mypod | aidevops analyze these errors"
        echo "Example: cat nginx.conf | aidevops review this config"
        return 1
    fi

    local system_prompt="You are a DevOps and infrastructure expert. Provide practical solutions with commands and best practices."

    # Escape quotes in prompt for JSON
    prompt=$(echo "$prompt" | sed 's/"/\\"/g')

    curl -s -X POST http://localhost:5000/query \
        -H "Content-Type: application/json" \
        -d "{
            \"provider\": \"groq\",
            \"prompt\": \"$prompt\",
            \"system_prompt\": \"$system_prompt\"
        }" \
        | jq -r '.response'
}

# Help function
ai_help() {
    cat << 'EOF'
AI Query Tool - Bash Functions

Basic Commands:
  ai <question>                    - Ask a question (uses Groq)
  ai what is docker
  ai explain kubernetes pods
  echo "data" | ai analyze this     - Pipe input to AI

Provider-Specific:
  ai_provider <provider> <question> - Use specific provider
  ai_provider claude explain rust
  cat logs.txt | ai_provider groq analyze these logs

Expert Mode:
  ai_expert <expert_type> <question> - Query with expert system prompt
  ai_expert python how to use decorators
  cat code.py | ai_expert python review this code

Code Helper:
  aicode <language> <question>      - Programming questions
  aicode javascript async await examples
  cat app.js | aicode javascript optimize this

DevOps Helper:
  aidevops <question>               - DevOps/infrastructure questions
  aidevops debug kubernetes crashloopbackoff
  kubectl logs mypod | aidevops why is this failing
  cat nginx.conf | aidevops review this config
  docker ps | aidevops explain these containers

Utility:
  ai_providers                      - List available providers
  ai_health                         - Check API health
  ai_help                          - Show this help

Advanced:
  ai_advanced '<json>'              - Raw JSON query
  ai_advanced '{"provider":"groq","prompt":"test","max_tokens":100}'

Piping Examples:
  cat error.log | aidevops explain these errors
  kubectl describe pod mypod | aidevops troubleshoot
  docker inspect container | aidevops optimize
  cat Dockerfile | aidevops improve this
  git diff | aicode python explain these changes
  tail -100 /var/log/syslog | aidevops analyze
  kubectl get events | aidevops summarize issues

Requirements:
  - Docker container running on localhost:5000
  - jq installed (sudo apt install jq)

Examples:
  ai how do I create a dockerfile
  ai_provider claude explain monads
  ai_expert rust borrowing and lifetimes
  aicode python read csv with pandas
  aidevops troubleshoot pod not starting

Piping Examples:
  cat error.log | aidevops explain
  kubectl logs pod | aidevops analyze
  docker ps | aidevops what are these containers
  cat script.py | aicode python review
EOF
}

# Print instructions on source
echo "AI Query functions loaded!"
echo "Type 'ai_help' for usage information"
echo "Quick start: ai what is kubernetes"
