//
// Program Name: chatmachine
// Description: a chatbot in c++ using an AIML base
//
// Author: Simon Grandsire
//

#include "chatmachine.h"
#include "categorylist.h"
#include "opencog_aiml.h"
#include "chatgpt4o.h"
#include "pattern_lattice.h"
#include "constraint_engine.h"
#include "diffusion_engine.h"
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <string>
#include <vector>
#include <future>
#include <chrono>
#include "tinyxml.h"
#include "aimlparser.h"
#include "xml.h"

using namespace std;

string aliceAimlFiles[] = {
    "ai",
    "alice",
    "astrology",
    "atomic",
    "badanswer",
    "biography",
    "bot",
    "bot_profile",
    "client",
    "client_profile",
    "computers",
    "continuation",
    //"date",
    "default",
    "drugs",
    "emotion",
    "food",
    "geography",
    "gossip",
    "history",
    "humor",
    "imponderables",
    "inquiry",
    "interjection",
    "iu",
    "knowledge",
    "literature",
    "loebner10",
    "money",
    "movies",
    "mp0",
    "mp1",
    "mp2",
    "mp3",
    "mp4",
    "mp5",
    "mp6",
    "music",
    "numbers",
    "personality",
    "phone",
    "pickup",
    "politics",
    "primeminister",
    "primitive-math",
    "psychology",
    "reduction0.safe",
    "reduction1.safe",
    "reduction2.safe",
    "reduction3.safe",
    "reduction4.safe",
    "reductions-update",
    "religion",
    "salutations",
    "science",
    "sex",
    "sports",
    "stack",
    "stories",
    "that",
    //"update1", //Error reading end tag
    "wallace"
};

size_t aliceAimlFilesSize = sizeof(aliceAimlFiles)/sizeof(aliceAimlFiles[0]);

string basicAimlFiles[] = {
    "bot",
    "condition",
    "default",
    "random",
    "salutations",
    "setget",
    "srai",
    "srai_star",
    "star",
    "that",
    "think",
    "topic"
};

size_t basicAimlFilesSize = sizeof(basicAimlFiles)/sizeof(basicAimlFiles[0]);

string dataDir = "database/Alice/";

string sUserPrompt = "USER> ";
string sBotPrompt = "CHATMACHINE> ";

CategoryList* cl;
vector<CategoryList*> cls;

map<string, string> mVars;

string strategy = "alice";

int main(int argc, char* argv[])
{
    cout << "Chatmachine v2.1 with OpenCog + ChatGPT-4o Integration Copyright (C) 2017-2024 Simon Grandsire\n" << endl;

    Chatmachine cm("Chatmachine");

    if (argc > 1) {
        if (string(argv[1]) == "basic") {
            strategy = "basic";
            dataDir = "database/Basic/";
        } else if (string(argv[1]) == "opencog") {
            strategy = "alice";
            dataDir = "database/Alice/";
            cm.setOpenCogMode(true);
            cout << "OpenCog cognitive mode enabled!" << endl;
        } else if (string(argv[1]) == "chatgpt4o") {
            strategy = "basic";
            dataDir = "database/Basic/";
            cm.setOpenCogMode(false);  // Disable OpenCog for ChatGPT-4o only mode
            cm.setChatGPT4oMode(true);
            cout << "ChatGPT-4o mode enabled (OpenCog disabled)!" << endl;
        } else if (string(argv[1]) == "full") {
            strategy = "basic";  // Use basic for demo since Alice files don't exist
            dataDir = "database/Basic/";
            cm.setOpenCogMode(true);
            cm.setChatGPT4oMode(true);
            cout << "Full AI mode enabled (OpenCog + ChatGPT-4o)!" << endl;
        } else if (string(argv[1]) == "noopencog") {
            strategy = "alice";
            dataDir = "database/Alice/";
            cm.setOpenCogMode(false);
            cout << "OpenCog mode disabled - using traditional AIML only." << endl;
        } else if (string(argv[1]) == "nsvd") {
            strategy = "basic";
            dataDir = "database/Basic/";
            cm.setOpenCogMode(true);
            cm.setChatGPT4oMode(true);
            cm.setNSVDMode(true);
            cout << "NSVD mode enabled (PatternLattice + ConstraintEngine + DiffusionEngine)!" << endl;
        } else if (string(argv[1]) == "nsvd-learn") {
            strategy = "basic";
            dataDir = "database/Basic/";
            cm.setOpenCogMode(true);
            cm.setChatGPT4oMode(true);
            cm.setNSVDMode(true);
            cm.setNSVDLearning(true);
            cout << "NSVD-Learn mode enabled (online AIML synthesis + consolidation)!" << endl;
        } else if (string(argv[1]) == "nsvd-constrained") {
            strategy = "basic";
            dataDir = "database/Basic/";
            cm.setOpenCogMode(true);
            cm.setChatGPT4oMode(true);
            cm.setNSVDMode(true);
            cm.setNSVDLearning(true);
            cm.setNSVDConstrained(true);
            cout << "NSVD-Constrained mode enabled (strict topicality + constraint optimisation)!" << endl;
        } else {
            strategy = "alice";
            dataDir = "database/Alice/";
        }
    } else {
        strategy = "basic";
        dataDir = "database/Basic/";
    }

    cout << "Loading data..." << endl;

    cm.createCategoryLists();

    cout << "Type 'stats' to see knowledge statistics, 'gpt4o' to see ChatGPT-4o config,\n"
         << "     'nsvd' to see NSVD stats, 'quit' to exit." << endl;

    while(1) {
        try {
            cm.listen();
            
            if (cm.m_sInput == "quit" || cm.m_sInput == "exit") {
                cout << "Goodbye!" << endl;
                break;
            } else if (cm.m_sInput == "stats") {
                cm.showKnowledgeStats();
                continue;
            } else if (cm.m_sInput == "gpt4o") {
                cm.showChatGPT4oConfig();
                continue;
            } else if (cm.m_sInput == "nsvd") {
                cm.showNSVDStats();
                continue;
            }
            
            cm.respond();
        } catch(std::string message) {
            cerr << message << endl;
        } catch(...) {
            cerr << "Unexpected error." << endl;
        }
    }

    return 0;
}

void Chatmachine::init_random() {
    srand((unsigned) time(NULL));
}

Chatmachine::Chatmachine(string str)
    : m_sChatBotName(str), m_sInput(""), m_bInput_prepared(0), m_nFileIndex(0), m_sPrevResponse(""), 
      m_bOpenCogEnabled(true), m_pOpenCogIntegration(nullptr),  // Keep original default (enabled)
      m_bChatGPT4oEnabled(false), m_pChatGPT4oIntegration(nullptr),
      m_bNSVDEnabled(false), m_bNSVDLearning(false), m_bNSVDConstrained(false),
      m_pPatternLattice(nullptr), m_pConstraintEngine(nullptr),
      m_pDiffusionEngine(nullptr), m_pLearnableCategoryList(nullptr),
      m_turnCount(0)
{
    init_random();
    // OpenCog, ChatGPT-4o, and NSVD initialization will happen after categories are loaded
}

Chatmachine::~Chatmachine() {
    // Destructor - unique_ptr will automatically clean up
}

void Chatmachine::listen() {
    cout << sUserPrompt;
    getline(cin, m_sInput);

    if (m_sInput != "")
        normalize(m_sInput);
}

void Chatmachine::respond() {
    string response;

    init_random();

    if (m_sInput == "") {
        cout << sBotPrompt << "Hmm." << endl;
        return;
    }

    if (m_sPrevInput != "" && m_sInput == m_sPrevInput) {
        cout << sBotPrompt << "You have already said that." << endl;
        return;
    }

    // -----------------------------------------------------------------------
    // NSVD parallel pipeline (new modes: nsvd / nsvd-learn / nsvd-constrained)
    // -----------------------------------------------------------------------
    if (m_bNSVDEnabled) {
        response = nsvd_respond();
    } else {
        // -------------------------------------------------------------------
        // Legacy sequential pipeline (backward-compatible)
        // -------------------------------------------------------------------
        shuffle();

        // Use OpenCog enhanced response if enabled
        if (m_bOpenCogEnabled && m_pOpenCogIntegration) {
            vector<aiml::Category*> allCategories;
            for (auto& categoryList : cls) {
                for (auto& category : categoryList->getCategories()) {
                    allCategories.push_back(category);
                }
            }

            response = m_pOpenCogIntegration->enhancedPatternMatch(m_sInput, allCategories);

            // Learn from this interaction
            if (!response.empty()) {
                m_pOpenCogIntegration->learnFromInteraction(m_sInput, response, 0.8);
            }
        }

        // Fall back to traditional AIML if no OpenCog response
        if (response.empty()) {
            response = get_response(m_sInput);
        }

        // Use ChatGPT-4o as final fallback if enabled and no good response found
        if ((response.empty() || response == "I don't understand what you're saying.") &&
            m_bChatGPT4oEnabled && m_pChatGPT4oIntegration &&
            m_pChatGPT4oIntegration->isConfigured())
        {
            cout << "[Consulting ChatGPT-4o...]" << endl;
            string gptResponse = m_pChatGPT4oIntegration->generateContextualResponse(
                m_sInput, m_conversationHistory);

            if (!gptResponse.empty()) {
                response = "[GPT-4o] " + gptResponse;
            }
        }
    }

    if (response.empty()) {
        cout << sBotPrompt << "I don't understand what you're saying." << endl;
    } else {
        setResponse(response);

        // Update conversation history for ChatGPT-4o context
        if (m_bChatGPT4oEnabled) {
            m_conversationHistory.push_back(m_sInput);
            m_conversationHistory.push_back(response);

            // Keep only recent history (last 10 exchanges)
            if (m_conversationHistory.size() > 20) {
                m_conversationHistory.erase(m_conversationHistory.begin(),
                                             m_conversationHistory.begin() + 2);
            }
        }

        cout << sBotPrompt << m_sResponse << endl;
    }
}

string Chatmachine::get_response(string input) {
    string bestResponse;

    bestResponse = get_best_response(input);

    return bestResponse;
}

string Chatmachine::get_best_response(string input) {
    vector<lev_pat_templ> levTempls;
    TiXmlDocument doc;
    TiXmlElement* root;
    Template* bestTemplate;
    Pattern* bestPattern;
    string bestResponse;
    unsigned int bestLevDist, bestIndex;

    for(unsigned int i=0, clss=cls.size(); i<clss; ++i) {
        levTempls.push_back(parse_categoryList(cls[i], input, m_sPrevResponse, mVars));
    }

    bestLevDist = levTempls[0].patternLevDist;
    bestIndex = 0;

    for(int i=1, s=levTempls.size(); i<s; ++i) {
        if (bestLevDist > levTempls[i].patternLevDist && levTempls[i].templ->toString() != "") {
            bestLevDist = levTempls[i].patternLevDist;
            bestIndex = i;
        }
    }

    bestTemplate = levTempls[bestIndex].templ;
    bestPattern = levTempls[bestIndex].pat;
    bestResponse = bestTemplate->toString();

    bestResponse = parse_template(cls[bestIndex], bestPattern, bestTemplate, input, m_sPrevResponse, mVars);

    return bestResponse;
}

void Chatmachine::setResponse(string response) {
    m_sResponse = response;
    prepare_response(m_sResponse);

    m_sPrevResponse = m_sResponse;
    m_sPrevInput = m_sInput;
}

void Chatmachine::prepare_response(string &resp) {

}

void Chatmachine::normalize(string &input) {
    //subsitute(input, subs);
    toUpper(input);
    shrink(input);
    insert_spaces(input);

    m_sInput = input;

    m_bInput_prepared = 1;
}

void Chatmachine::createCategoryLists() {
    TiXmlDocument doc;
    TiXmlElement* root;
    unsigned int aimlFilesSize = strategy == "basic" ? basicAimlFilesSize : aliceAimlFilesSize;
    string* aimlFiles = strategy == "basic" ? basicAimlFiles : aliceAimlFiles;

    while(m_nFileIndex < aimlFilesSize) {
        string sAimlFile;
        const char* aimlFile;

        sAimlFile = dataDir + aimlFiles[m_nFileIndex] + ".aiml";
        aimlFile = sAimlFile.c_str();

        cl = new CategoryList(aimlFiles[m_nFileIndex]);
        cls.push_back(cl);

        if (!doc.LoadFile(aimlFile)) {
            cerr << doc.ErrorDesc() << " " << aimlFile << endl;
            return;
        }

        root = doc.FirstChildElement();

        if (root == NULL) {
            cerr << "Failed to load file: No root element. " << aimlFile << endl;
            doc.Clear();
            return;
        }

        createCategoryList(cl, root);

        m_nFileIndex++;
    }

    //to do
    //cout << cl << endl;
    
    // Initialize OpenCog with loaded categories after successful loading
    if (m_bOpenCogEnabled) {
        try {
            initializeOpenCog();
            
            vector<aiml::Category*> allCategories;
            for (auto& categoryList : cls) {
                const auto& categories = categoryList->getCategories();
                allCategories.insert(allCategories.end(), categories.begin(), categories.end());
            }
            
            if (m_pOpenCogIntegration) {
                m_pOpenCogIntegration->initializeFromCategories(allCategories);
                cout << "OpenCog knowledge base initialized with " << allCategories.size() << " categories." << endl;
            }
        } catch (const exception& e) {
            cerr << "OpenCog initialization failed: " << e.what() << endl;
            m_bOpenCogEnabled = false;
        }
    }
    
    // Initialize ChatGPT-4o if enabled
    if (m_bChatGPT4oEnabled) {
        try {
            initializeChatGPT4o();
            cout << "ChatGPT-4o integration initialized." << endl;
        } catch (const exception& e) {
            cerr << "ChatGPT-4o initialization failed: " << e.what() << endl;
            m_bChatGPT4oEnabled = false;
        }
    }
    
    // Initialize NSVD pipeline if enabled
    if (m_bNSVDEnabled) {
        try {
            initializeNSVD();
            cout << "NSVD pipeline initialized." << endl;
        } catch (const exception& e) {
            cerr << "NSVD initialization failed: " << e.what() << endl;
            m_bNSVDEnabled = false;
        }
    }
}

void Chatmachine::shuffle() {
    srand(time(NULL));

    vector<CategoryList*> cls_;

    for(unsigned int i=0, s=cls.size(); i<s; ++i) {
        cls_.push_back(cls[i]);
    }

    cls.clear();

    while(cls_.size() > 0) {
        unsigned int i = rand() % cls_.size();

        cls.push_back(cls_[i]);
        cls_.erase(cls_.begin() + i);
    }
}

void Chatmachine::initializeOpenCog() {
    try {
        m_pOpenCogIntegration = unique_ptr<opencog_aiml::OpenCogAIMLIntegration>(new opencog_aiml::OpenCogAIMLIntegration());
        cout << "OpenCog integration initialized successfully." << endl;
    } catch (const exception& e) {
        cerr << "Failed to initialize OpenCog: " << e.what() << endl;
        m_bOpenCogEnabled = false;
    }
}

void Chatmachine::showKnowledgeStats() {
    if (m_pOpenCogIntegration) {
        m_pOpenCogIntegration->printKnowledgeStats();
    } else {
        cout << "OpenCog integration not available." << endl;
    }
}

void Chatmachine::initializeChatGPT4o() {
    try {
        m_pChatGPT4oIntegration = unique_ptr<chatgpt4o::ChatGPT4oIntegration>(new chatgpt4o::ChatGPT4oIntegration());
        
        // Try to read API key from environment variable
        const char* apiKey = getenv("OPENAI_API_KEY");
        if (apiKey) {
            m_pChatGPT4oIntegration->setApiKey(string(apiKey));
            cout << "ChatGPT-4o API key loaded from environment." << endl;
        } else {
            cout << "No OPENAI_API_KEY environment variable found. ChatGPT-4o will run in simulation mode." << endl;
        }
        
        cout << "ChatGPT-4o integration initialized successfully." << endl;
    } catch (const exception& e) {
        cerr << "Failed to initialize ChatGPT-4o: " << e.what() << endl;
        m_bChatGPT4oEnabled = false;
    }
}

void Chatmachine::setChatGPT4oApiKey(const string& apiKey) {
    if (m_pChatGPT4oIntegration) {
        m_pChatGPT4oIntegration->setApiKey(apiKey);
    }
}

void Chatmachine::showChatGPT4oConfig() {
    if (m_pChatGPT4oIntegration) {
        m_pChatGPT4oIntegration->printConfiguration();
        cout << "Enabled: " << (m_bChatGPT4oEnabled ? "Yes" : "No") << endl;
        cout << "Conversation history length: " << m_conversationHistory.size() << endl;
    } else {
        cout << "ChatGPT-4o integration not available." << endl;
    }
}

// ---------------------------------------------------------------------------
// NSVD pipeline implementation
// ---------------------------------------------------------------------------

void Chatmachine::initializeNSVD() {
    // PatternLattice — build over all loaded categories.
    m_pPatternLattice.reset(new pattern_lattice::PatternLattice());
    vector<aiml::Category*> allCategories;
    for (auto& cl_item : cls) {
        for (auto& cat : cl_item->getCategories())
            allCategories.push_back(cat);
    }
    m_pPatternLattice->build(allCategories);
    cout << "PatternLattice built: " << m_pPatternLattice->size()
         << " categories (" << m_pPatternLattice->wildcardCount()
         << " wildcard, " << m_pPatternLattice->specificCount()
         << " specific)." << endl;

    // ConstraintEngine.
    m_pConstraintEngine.reset(new constraint_engine::ConstraintEngine());

    // DiffusionEngine — requires OpenCog AtomSpace.
    if (m_pOpenCogIntegration) {
        m_pDiffusionEngine.reset(
            new diffusion_engine::DiffusionEngine(
                opencog::AtomSpaceManager::getInstance()));
    }

    // LearnableCategoryList.
    if (m_bNSVDLearning) {
        m_pLearnableCategoryList.reset(new aiml::LearnableCategoryList());
        cout << "LearnableCategoryList initialised (online synthesis enabled)." << endl;
    }
}

void Chatmachine::showNSVDStats() {
    cout << "\n=== NSVD Pipeline Status ===" << endl;
    cout << "NSVD enabled:     " << (m_bNSVDEnabled     ? "Yes" : "No") << endl;
    cout << "NSVD learning:    " << (m_bNSVDLearning    ? "Yes" : "No") << endl;
    cout << "NSVD constrained: " << (m_bNSVDConstrained ? "Yes" : "No") << endl;
    cout << "Turn count:       " << m_turnCount << endl;
    if (m_pPatternLattice) {
        cout << "PatternLattice:   " << m_pPatternLattice->size()
             << " categories" << endl;
    }
    if (m_pLearnableCategoryList) {
        cout << "Learned cats:     "
             << m_pLearnableCategoryList->size() << endl;
    }
    if (m_pDiffusionEngine) {
        m_pDiffusionEngine->printDiffusionStats();
    }
    cout << "============================\n" << endl;
}

// ---------------------------------------------------------------------------
// NSVD parallel respond
// ---------------------------------------------------------------------------

string Chatmachine::nsvd_respond() {
    using namespace constraint_engine;

    // 1. Choose constraint profile.
    ResponseConstraints constraints = m_bNSVDConstrained
        ? ConstraintEngine::strictConstraints()
        : ConstraintEngine::defaultConstraints();

    // 2. Get context vector from OpenCog layer.
    map<string, double> contextVector;
    if (m_pOpenCogIntegration)
        contextVector = m_pOpenCogIntegration->getContextVector();

    // 3. Launch symbolic path asynchronously.
    auto symbFuture = std::async(std::launch::async,
        [this]() -> SymbolicResult {
            return symbolicPath(m_sInput);
        });

    // 4. Sub-symbolic path runs in parallel only if symbolic confidence is
    //    likely to be low (determined after both finish; launched eagerly to
    //    hide latency).
    auto subSymFuture = std::async(std::launch::async,
        [this]() -> SymbolicResult {
            return subSymbolicPath(m_sInput);
        });

    // 5. Collect results within a time budget.
    SymbolicResult symbResult   = {"", 0.0, 0.0};
    SymbolicResult subSymResult = {"", 0.0, 0.0};

    try {
        symbResult = symbFuture.get();
    } catch (const exception& e) {
        cerr << "[NSVD] Symbolic path error: " << e.what() << endl;
    } catch (...) {
        cerr << "[NSVD] Symbolic path: unknown error" << endl;
    }

    try {
        subSymResult = subSymFuture.get();
    } catch (const exception& e) {
        cerr << "[NSVD] Sub-symbolic path error: " << e.what() << endl;
    } catch (...) {
        cerr << "[NSVD] Sub-symbolic path: unknown error" << endl;
    }

    // 6. Build candidate list.
    vector<ResponseCandidate> candidates;
    if (!symbResult.text.empty())
        candidates.emplace_back(symbResult.text, symbResult.score,
                                 "aiml", symbResult.confidence);
    if (!subSymResult.text.empty() && subSymResult.text != symbResult.text)
        candidates.emplace_back(subSymResult.text, subSymResult.score,
                                 "opencog", subSymResult.confidence);

    // Also offer learned categories as candidates if available.
    if (m_pLearnableCategoryList && m_pPatternLattice) {
        auto scored = m_pPatternLattice->findBestCategories(
            m_sInput, contextVector, 3);
        for (const auto& sc : scored) {
            if (sc.category && sc.category->getTruthValue().immutable == false &&
                sc.category->templ() && sc.score > 0.1)
            {
                candidates.emplace_back(
                    sc.category->templ()->toString(),
                    sc.score,
                    "learned",
                    sc.category->getTruthValue().confidence);
            }
        }
    }

    if (candidates.empty())
        return "";

    // 7. Apply constraint engine.
    ResponseCandidate best = m_pConstraintEngine->selectBestCandidate(
        candidates, constraints, contextVector, m_recentResponses);

    string response = best.text;

    // 8. If still empty and GPT-4o is available, try constrained generation.
    if (response.empty() && m_bChatGPT4oEnabled && m_pChatGPT4oIntegration &&
        m_pChatGPT4oIntegration->isConfigured())
    {
        cout << "[NSVD → ChatGPT-4o constrained...]" << endl;
        string constraintPrompt = m_pConstraintEngine->buildGPT4oConstraintPrompt(
            constraints, contextVector, m_recentResponses);
        string gptResp = m_pChatGPT4oIntegration->generateConstrainedResponse(
            m_sInput, m_conversationHistory, constraintPrompt);
        if (!gptResp.empty()) {
            response = "[GPT-4o] " + gptResp;
            best = ResponseCandidate(response, 0.6, "gpt4o", 1.0);
        }
    }

    // 9. Post-response updates.
    if (!response.empty()) {
        updateNSVDState(m_sInput, response);
        if (m_bNSVDLearning && best.source == "gpt4o")
            maybeSynthesizeCategory(m_sInput, response);
    }

    return response;
}

Chatmachine::SymbolicResult Chatmachine::symbolicPath(const string& input) {
    SymbolicResult result = {"", 0.0, 0.0};
    if (!m_pPatternLattice) return result;

    map<string, double> contextVector;
    if (m_pOpenCogIntegration)
        contextVector = m_pOpenCogIntegration->getContextVector();

    auto candidates = m_pPatternLattice->findBestCategories(
        input, contextVector, 3);

    for (const auto& sc : candidates) {
        if (!sc.category || !sc.category->templ()) continue;
        string text = sc.category->templ()->toString();
        if (text.empty()) continue;
        result.text       = text;
        result.score      = sc.score;
        result.confidence = sc.category->getTruthValue().confidence;
        return result;
    }

    // Fallback to traditional AIML parser if lattice gives nothing.
    string fallback = get_response(input);
    if (!fallback.empty()) {
        result.text       = fallback;
        result.score      = 0.3;
        result.confidence = 0.9; // static AIML is always confident
    }
    return result;
}

Chatmachine::SymbolicResult Chatmachine::subSymbolicPath(const string& input) {
    SymbolicResult result = {"", 0.0, 0.0};
    if (!m_pOpenCogIntegration) return result;

    // 1. Update context vector.
    m_pOpenCogIntegration->updateContextVector(input);

    // 2. Try concept interpolation via DiffusionEngine.
    if (m_pDiffusionEngine) {
        auto topConcepts = m_pOpenCogIntegration->getTopConcepts(2);
        if (topConcepts.size() >= 2) {
            string blend = m_pDiffusionEngine->interpolateConcepts(
                topConcepts[0].first, topConcepts[1].first);
            if (!blend.empty()) {
                // Get responses related to the blended concept.
                auto relatedResponses = opencog::AtomSpaceManager::getInstance()
                    .getRelatedResponses(blend);
                if (!relatedResponses.empty()) {
                    result.text       = relatedResponses[0];
                    result.score      = 0.5;
                    result.confidence = 0.4;
                    return result;
                }
            }
        }
    }

    // 3. OpenCog knowledge-based generation.
    string kbResponse = m_pOpenCogIntegration->generateKnowledgeBasedResponse(input);
    if (!kbResponse.empty()) {
        result.text       = kbResponse;
        result.score      = 0.4;
        result.confidence = 0.5;
    }
    return result;
}

void Chatmachine::maybeSynthesizeCategory(const string& input,
                                           const string& response) {
    if (!m_pLearnableCategoryList) return;

    // Strip GPT-4o prefix for the stored template.
    string cleanResponse = response;
    const string prefix = "[GPT-4o] ";
    if (cleanResponse.size() >= prefix.size() &&
        cleanResponse.substr(0, prefix.size()) == prefix)
    {
        cleanResponse = cleanResponse.substr(prefix.size());
    }

    aiml::Category* cat = m_pLearnableCategoryList->synthesize(input,
                                                                 cleanResponse);
    if (cat && m_pPatternLattice)
        m_pPatternLattice->addLearnedCategory(cat);
}

void Chatmachine::updateNSVDState(const string& input,
                                   const string& response) {
    m_turnCount++;

    // Update context vector in OpenCog layer.
    if (m_pOpenCogIntegration) {
        m_pOpenCogIntegration->updateContextVector(input);
        m_pOpenCogIntegration->updateContextVector(response, 0.5);
        m_pOpenCogIntegration->decayContextVector();
        m_pOpenCogIntegration->learnFromInteraction(input, response, 0.7);
    }

    // Trust propagation from new sentence node.
    if (m_pDiffusionEngine) {
        auto& atomSpace = opencog::AtomSpaceManager::getInstance();
        auto sentNode = atomSpace.addSentenceNode(input);
        if (sentNode)
            m_pDiffusionEngine->propagateTrustFromAtom(sentNode, 0.2, 3);
        m_pDiffusionEngine->decayTemperature(0.97);
    }

    // Decay learned categories.
    if (m_pLearnableCategoryList)
        m_pLearnableCategoryList->decayAll(0.99);

    // Decay pattern lattice.
    if (m_pPatternLattice)
        m_pPatternLattice->decayAll(0.99);

    // Rolling recent-responses window.
    m_recentResponses.push_back(response);
    if ((int)m_recentResponses.size() > 10)
        m_recentResponses.erase(m_recentResponses.begin());

    // Conversation history for GPT-4o.
    if (m_bChatGPT4oEnabled) {
        m_conversationHistory.push_back(input);
        m_conversationHistory.push_back(response);
        if (m_conversationHistory.size() > 20)
            m_conversationHistory.erase(m_conversationHistory.begin(),
                                         m_conversationHistory.begin() + 2);
    }

    // Periodic consolidation (every 20 turns when learning is enabled).
    if (m_bNSVDLearning && m_pDiffusionEngine &&
        m_pLearnableCategoryList && m_turnCount % 20 == 0)
    {
        m_pDiffusionEngine->consolidateLearnedCategories(
            m_pLearnableCategoryList.get(), "database/Learned/");
        m_pLearnableCategoryList->prune(0.02);
        m_pDiffusionEngine->garbageCollectBlends(0.05);
    }
}