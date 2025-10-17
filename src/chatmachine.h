#ifndef __CHATMACHINE_H__
#define __CHATMACHINE_H__

#include <string>
#include <vector>
#include <memory>

using namespace std;

// Forward declarations
namespace opencog_aiml {
    class OpenCogAIMLIntegration;
}

class Chatmachine {
public:
    Chatmachine (string str);
    ~Chatmachine();  // Add destructor

    void listen();
    void respond();
    void createCategoryLists();
    
    // OpenCog integration methods
    void initializeOpenCog();
    void showKnowledgeStats();
    void setOpenCogMode(bool enabled) { m_bOpenCogEnabled = enabled; }
    
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
};

#endif
