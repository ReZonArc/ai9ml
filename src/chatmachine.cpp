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
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <string>
#include <vector>
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

    cout << "Type 'stats' to see knowledge statistics, 'gpt4o' to see ChatGPT-4o config, 'quit' to exit." << endl;

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
      m_bOpenCogEnabled(false), m_pOpenCogIntegration(nullptr),  // Default to disabled
      m_bChatGPT4oEnabled(false), m_pChatGPT4oIntegration(nullptr)
{
    init_random();
    // OpenCog and ChatGPT-4o initialization will happen after categories are loaded
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
        m_bChatGPT4oEnabled && m_pChatGPT4oIntegration && m_pChatGPT4oIntegration->isConfigured()) {
        
        cout << "[Consulting ChatGPT-4o...]" << endl;
        string gptResponse = m_pChatGPT4oIntegration->generateContextualResponse(m_sInput, m_conversationHistory);
        
        if (!gptResponse.empty()) {
            response = gptResponse;
            // Add ChatGPT-4o indicator
            response = "[GPT-4o] " + response;
        }
    }

    if (response.empty()) {
        cout << sBotPrompt <<  "I don't understand what you're saying." << endl;
    } else {
        setResponse(response);
        
        // Update conversation history for ChatGPT-4o context
        if (m_bChatGPT4oEnabled) {
            m_conversationHistory.push_back(m_sInput);
            m_conversationHistory.push_back(response);
            
            // Keep only recent history (last 10 exchanges)
            if (m_conversationHistory.size() > 20) {
                m_conversationHistory.erase(m_conversationHistory.begin(), m_conversationHistory.begin() + 2);
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