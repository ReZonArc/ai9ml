#ifndef __CHATMACHINE_H__
#define __CHATMACHINE_H__

#include <string>
#include <vector>
#include <memory>
#include <future>

using namespace std;

// Forward declarations
namespace opencog_aiml {
    class OpenCogAIMLIntegration;
}

namespace chatgpt4o {
    class ChatGPT4oIntegration;
}

namespace pattern_lattice {
    class PatternLattice;
}

namespace constraint_engine {
    class ConstraintEngine;
    struct ResponseConstraints;
}

namespace diffusion_engine {
    class DiffusionEngine;
}

namespace aiml {
    class LearnableCategoryList;
}

class Chatmachine {
public:
    Chatmachine (string str);
    ~Chatmachine();

    void listen();
    void respond();
    void createCategoryLists();
    
    // OpenCog integration methods
    void initializeOpenCog();
    void showKnowledgeStats();
    void setOpenCogMode(bool enabled) { m_bOpenCogEnabled = enabled; }
    
    // ChatGPT-4o integration methods
    void initializeChatGPT4o();
    void setChatGPT4oMode(bool enabled) { m_bChatGPT4oEnabled = enabled; }
    void setChatGPT4oApiKey(const string& apiKey);
    void showChatGPT4oConfig();

    // NSVD pipeline control
    void setNSVDMode(bool enabled)        { m_bNSVDEnabled   = enabled; }
    void setNSVDLearning(bool enabled)    { m_bNSVDLearning  = enabled; }
    void setNSVDConstrained(bool enabled) { m_bNSVDConstrained = enabled; }
    void initializeNSVD();
    void showNSVDStats();
    
    // Public access to input for main loop
    string m_sInput;

private:
    void init_random();
    void normalize(string &input);
    string get_response(string input);
    string get_best_response(string input);
    void setResponse(string sResponse);
    void prepare_response(string &resp);
    void shuffle();

    // NSVD parallel pipeline — returns the best candidate response string.
    string nsvd_respond();

    // Symbolic path: PatternLattice lookup with context-aware scoring.
    // Returns {response, score, confidence}.
    struct SymbolicResult { string text; double score; double confidence; };
    SymbolicResult symbolicPath(const string& input);

    // Sub-symbolic path: AtomSpace + optional GPT-4o.
    SymbolicResult subSymbolicPath(const string& input);

    // Synthesise a learnable category if the GPT-4o response is novel enough.
    void maybeSynthesizeCategory(const string& input, const string& response);

    // Consolidate and decay at end of turn.
    void updateNSVDState(const string& input, const string& response);

private:
    string m_sChatBotName;
    string m_sResponse;
    string m_sPrevInput;
    string m_sPrevResponse;
    string m_sSubject;
    string m_sAimlFile;
    vector<string> m_vsInputTokens;
    bool m_bInput_prepared;
    unsigned int m_nFileIndex;
    vector<string> response_list;
    
    // OpenCog integration
    unique_ptr<opencog_aiml::OpenCogAIMLIntegration> m_pOpenCogIntegration;
    bool m_bOpenCogEnabled;
    
    // ChatGPT-4o integration
    unique_ptr<chatgpt4o::ChatGPT4oIntegration> m_pChatGPT4oIntegration;
    bool m_bChatGPT4oEnabled;
    vector<string> m_conversationHistory;

    // NSVD pipeline
    bool m_bNSVDEnabled;
    bool m_bNSVDLearning;
    bool m_bNSVDConstrained;
    unique_ptr<pattern_lattice::PatternLattice>      m_pPatternLattice;
    unique_ptr<constraint_engine::ConstraintEngine>  m_pConstraintEngine;
    unique_ptr<diffusion_engine::DiffusionEngine>    m_pDiffusionEngine;
    unique_ptr<aiml::LearnableCategoryList>          m_pLearnableCategoryList;
    vector<string> m_recentResponses;   // rolling window for anti-repetition
    int            m_turnCount;
};

#endif
