# chatmachine

A chatbot in C++ with **OpenCog cognitive architecture integration**. 

## Features

- **Traditional AIML** pattern matching and response generation
- **OpenCog AtomSpace** hypergraph knowledge representation
- **Cognitive reasoning** with concept hierarchies and semantic similarity
- **Knowledge-based learning** from user interactions
- **Context awareness** and conversation memory

## OpenCog Integration

This implementation includes:

- **AtomSpace**: Hypergraph database for knowledge representation
- **Atoms**: ConceptNode, WordNode, SentenceNode for knowledge storage
- **Links**: InheritanceLink, ImplicationLink for relationships
- **Semantic matching**: Enhanced pattern matching using concept similarity
- **Learning**: Dynamic knowledge acquisition from conversations

## Usage

At root of project, make the directories obj/, database/Alice/ (download aiml files from ALICE project : https://github.com/Urthen/aiml-en-us-foundation-alice.v1-4)

**NEW: ChatGPT-4o Integration Available!** 

Enter commands:

```bash
make clean
make

# Traditional modes
./chatmachine9 basic       # Basic AIML only  
./chatmachine9 opencog     # Enable OpenCog cognitive mode  
./chatmachine9 noopencog   # Traditional AIML only

# NEW: ChatGPT-4o Integration
./chatmachine9 chatgpt4o   # AIML + ChatGPT-4o fallback
./chatmachine9 full        # AIML + OpenCog + ChatGPT-4o (full AI mode)
```

### ChatGPT-4o Setup
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
./chatmachine9 chatgpt4o
```

Without an API key, runs in simulation mode with intelligent mock responses.

### Commands in Chat:
- `gpt4o` - Show ChatGPT-4o configuration and status
- `stats` - Show OpenCog knowledge statistics
- `quit` - Exit the program

## Architecture

The system combines traditional AIML with OpenCog's cognitive architecture and ChatGPT-4o integration:

```
User Input → AIML Parser → OpenCog AtomSpace → ChatGPT-4o Fallback → Enhanced Response
                ↓              ↓                    ↓
         Pattern Match   Semantic Analysis    AI Generation
                ↓              ↓                    ↓
         Traditional     Knowledge-based      Advanced AI
          Response        Generation           Response
```

**Response Hierarchy:**
1. **AIML Pattern Matching** - Traditional rule-based responses
2. **OpenCog Enhancement** - Cognitive reasoning and learning (if enabled)  
3. **ChatGPT-4o Fallback** - Advanced AI responses when others are insufficient

Responses from ChatGPT-4o are prefixed with `[GPT-4o]` for identification.

## Implementation Notes

- **Memory Management**: Uses smart pointers for safe memory handling
- **C++11 Compatible**: Builds with standard C++11 compiler
- **Modular Design**: OpenCog and ChatGPT-4o can be enabled/disabled at runtime
- **1,352+ lines** of OpenCog integration code
- **500+ lines** of ChatGPT-4o integration code
- **Intelligent Fallbacks**: Graceful degradation when AI services are unavailable
- **Simulation Mode**: Runs without external API dependencies for testing

## Contributing


