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

Enter commands:

```bash
make clean
make
./chatmachine9 opencog     # Enable OpenCog cognitive mode  
./chatmachine9 noopencog   # Traditional AIML only
./chatmachine9 basic       # Use basic test files
```

### Commands in Chat:
- `stats` - Show OpenCog knowledge statistics
- `quit` - Exit the program

## Architecture

The system combines traditional AIML with OpenCog's cognitive architecture:

```
User Input -> AIML Parser -> OpenCog AtomSpace -> Enhanced Response
                   ↓              ↓
            Pattern Match   Semantic Analysis
                   ↓              ↓
            Traditional     Knowledge-based
             Response        Generation
```

## Implementation Notes

- **Memory Management**: Uses smart pointers for safe memory handling
- **C++11 Compatible**: Builds with standard C++11 compiler
- **Modular Design**: OpenCog can be enabled/disabled at runtime
- **1,352+ lines** of OpenCog integration code

## Contributing


