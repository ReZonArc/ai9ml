# ChatGPT-4o Integration Guide

## Overview

The AIML chatbot now includes integration with OpenAI's ChatGPT-4o-latest model, providing enhanced responses when traditional AIML patterns and OpenCog reasoning are insufficient.

## Usage Modes

### 1. ChatGPT-4o Only Mode
```bash
./chatmachine9 chatgpt4o
```
- Uses basic AIML patterns with ChatGPT-4o fallback
- OpenCog cognitive features disabled for simplicity
- Best for testing ChatGPT-4o integration

### 2. Full AI Mode
```bash
./chatmachine9 full
```
- Combines AIML + OpenCog + ChatGPT-4o
- Provides maximum AI capabilities
- Response hierarchy: AIML → OpenCog → ChatGPT-4o

### 3. Traditional Modes
```bash
./chatmachine9 basic      # AIML only
./chatmachine9 opencog    # AIML + OpenCog
./chatmachine9 noopencog  # AIML only (explicitly disabled OpenCog)
```

## Configuration

### API Key Setup
Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
./chatmachine9 chatgpt4o
```

Without an API key, the system runs in simulation mode with intelligent mock responses.

### Runtime Commands
- `gpt4o` - Show ChatGPT-4o configuration
- `stats` - Show OpenCog knowledge statistics (if enabled)  
- `quit` - Exit the program

## Response Flow

1. **AIML Pattern Matching**: Traditional rule-based responses
2. **OpenCog Enhancement**: Cognitive reasoning and learning (if enabled)
3. **ChatGPT-4o Fallback**: Advanced AI responses when others fail

Responses from ChatGPT-4o are prefixed with `[GPT-4o]` for identification.

## Implementation Notes

- **Simulation Mode**: Provides mock responses when no API key is configured
- **Conversation History**: Maintains context for ChatGPT-4o interactions
- **Minimal Integration**: Preserves existing AIML/OpenCog functionality
- **Error Handling**: Graceful fallbacks when ChatGPT-4o is unavailable

## Example Interaction

```
USER> hello
CHATMACHINE> Hello there!

USER> explain quantum entanglement
[Consulting ChatGPT-4o...]
CHATMACHINE> [GPT-4o] Quantum entanglement is a phenomenon where two or more particles become interconnected in such a way that the quantum state of each particle cannot be described independently...

USER> gpt4o
CHATMACHINE> 
=== ChatGPT-4o Configuration ===
Model: gpt-4o-latest
Temperature: 0.7
Max Tokens: 1000
API Key: Configured
==============================
```

## Development Notes

For production deployment with real OpenAI API access, add HTTPS/TLS support using libcurl or similar HTTP client library. The current implementation provides a complete integration framework with simulation capabilities.