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

namespace mlp_engine {
    class MLPEngine;
}

namespace logic_classifier {
    class LogicClassifier;
    struct Classification;
}

namespace workflow_engine {
    class WorkflowEngine;
}

namespace hgnn {
    class HyperGraphNeuralNet;
}

namespace dtesnn {
    class DeepTreeEchoStateNet;
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
    void setNSVDNeural(bool enabled)      { m_bNSVDNeural    = enabled; }
    void initializeNSVD();
    void showNSVDStats();
    void showLogicWorkflowStats();
    
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

    // HGNN async path: spatially-scored PatternLattice result.
    SymbolicResult hgnnPath(const string& input);

    // DTESNN async path: temporally-scored result.
    SymbolicResult dtesnnPath(const string& input);

    // Workflow path: logic-system classification + operational workflow routing.
    SymbolicResult workflowPath(const string& input);

    // Synthesise a learnable category if the GPT-4o response is novel enough.
    void maybeSynthesizeCategory(const string& input, const string& response);

    // Consolidate and decay at end of turn.  winningPath = mlp_engine PATH_* index.
    void updateNSVDState(const string& input, const string& response,
                         int winningPath = -1);

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
    bool m_bNSVDNeural;                                              // HGNN+DTESNN+MLP
    unique_ptr<pattern_lattice::PatternLattice>      m_pPatternLattice;
    unique_ptr<constraint_engine::ConstraintEngine>  m_pConstraintEngine;
    unique_ptr<diffusion_engine::DiffusionEngine>    m_pDiffusionEngine;
    unique_ptr<aiml::LearnableCategoryList>          m_pLearnableCategoryList;

    // Neural complement modules (activated by m_bNSVDNeural).
    unique_ptr<hgnn::HyperGraphNeuralNet>            m_pHGNN;
    unique_ptr<dtesnn::DeepTreeEchoStateNet>         m_pDTESNN;
    unique_ptr<mlp_engine::MLPEngine>                m_pMLP;
    unique_ptr<logic_classifier::LogicClassifier>    m_pLogicClassifier;
    unique_ptr<workflow_engine::WorkflowEngine>      m_pWorkflowEngine;
    vector<unique_ptr<aiml::Category>>               m_runtimeCategories;

    vector<string> m_recentResponses;   // rolling window for anti-repetition
    int            m_turnCount;
    int            m_lastOuterLoopCount; // tracks outer-loop completion for lr decay
    string         m_lastLogicSystem;
    double         m_lastLogicConfidence;
};

#endif
