#ifndef __CHATGPT4O_H__
#define __CHATGPT4O_H__

#include <string>
#include <vector>
#include <memory>

using namespace std;

namespace chatgpt4o {
    
    /**
     * ChatGPT4oIntegration - Integration with OpenAI's ChatGPT-4o-latest model
     * This class provides access to the latest ChatGPT-4o model for enhanced responses
     */
    class ChatGPT4oIntegration {
    public:
        ChatGPT4oIntegration();
        virtual ~ChatGPT4oIntegration();
        
        // Configuration
        void setApiKey(const string& apiKey);
        void setModel(const string& model = "gpt-4o-latest");
        void setTemperature(double temperature = 0.7);
        void setMaxTokens(int maxTokens = 1000);
        
        // Core functionality
        string generateResponse(const string& input);
        string generateContextualResponse(const string& input, const vector<string>& conversationHistory);
        
        // Status and debugging
        bool isConfigured() const;
        string getLastError() const;
        void printConfiguration() const;
        
    private:
        string m_apiKey;
        string m_model;
        double m_temperature;
        int m_maxTokens;
        string m_lastError;
        
        // HTTP client functionality
        string makeHttpRequest(const string& url, const string& headers, const string& payload);
        string buildOpenAIPayload(const string& input, const vector<string>& conversationHistory = vector<string>());
        string extractResponseFromJson(const string& jsonResponse);
        string escapeJsonString(const string& input);
    };
    
    /**
     * Simple HTTP client implementation without external dependencies
     */
    class SimpleHttpClient {
    public:
        struct HttpResponse {
            int statusCode;
            string headers;
            string body;
            bool success;
        };
        
        static HttpResponse post(const string& url, const string& headers, const string& body);
        
    private:
        static string parseHostFromUrl(const string& url);
        static string parsePathFromUrl(const string& url);
        static int parsePortFromUrl(const string& url);
        static bool isHttps(const string& url);
    };
}

#endif // __CHATGPT4O_H__