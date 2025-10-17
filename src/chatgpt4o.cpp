#include "chatgpt4o.h"
#include <iostream>
#include <sstream>
#include <regex>
#include <cstring>
#include <algorithm>
#include <fstream>

using namespace chatgpt4o;

// ChatGPT4oIntegration implementation
ChatGPT4oIntegration::ChatGPT4oIntegration() 
    : m_model("gpt-4o-latest"), m_temperature(0.7), m_maxTokens(1000) {
}

ChatGPT4oIntegration::~ChatGPT4oIntegration() {
}

void ChatGPT4oIntegration::setApiKey(const string& apiKey) {
    m_apiKey = apiKey;
}

void ChatGPT4oIntegration::setModel(const string& model) {
    m_model = model;
}

void ChatGPT4oIntegration::setTemperature(double temperature) {
    m_temperature = temperature;
}

void ChatGPT4oIntegration::setMaxTokens(int maxTokens) {
    m_maxTokens = maxTokens;
}

bool ChatGPT4oIntegration::isConfigured() const {
    return !m_apiKey.empty();
}

string ChatGPT4oIntegration::getLastError() const {
    return m_lastError;
}

void ChatGPT4oIntegration::printConfiguration() const {
    cout << "\n=== ChatGPT-4o Configuration ===" << endl;
    cout << "Model: " << m_model << endl;
    cout << "Temperature: " << m_temperature << endl;
    cout << "Max Tokens: " << m_maxTokens << endl;
    cout << "API Key: " << (m_apiKey.empty() ? "Not set" : "Configured") << endl;
    cout << "==============================\n" << endl;
}

string ChatGPT4oIntegration::generateResponse(const string& input) {
    return generateContextualResponse(input, vector<string>());
}

string ChatGPT4oIntegration::generateContextualResponse(const string& input, const vector<string>& conversationHistory) {
    if (!isConfigured()) {
        m_lastError = "API key not configured";
        return "";
    }
    
    try {
        // Build the request payload
        string payload = buildOpenAIPayload(input, conversationHistory);
        
        // Prepare headers
        stringstream headerStream;
        headerStream << "Content-Type: application/json\r\n";
        headerStream << "Authorization: Bearer " << m_apiKey << "\r\n";
        headerStream << "Content-Length: " << payload.length() << "\r\n";
        string headers = headerStream.str();
        
        // Make the API request
        string url = "https://api.openai.com/v1/chat/completions";
        string response = makeHttpRequest(url, headers, payload);
        
        if (response.empty()) {
            m_lastError = "Failed to get response from OpenAI API";
            return "";
        }
        
        // Extract response from JSON
        string result = extractResponseFromJson(response);
        if (result.empty()) {
            m_lastError = "Failed to parse response from OpenAI API";
        }
        
        return result;
        
    } catch (const exception& e) {
        m_lastError = "Exception in generateContextualResponse: " + string(e.what());
        return "";
    }
}

string ChatGPT4oIntegration::buildOpenAIPayload(const string& input, const vector<string>& conversationHistory) {
    stringstream payload;
    payload << "{";
    payload << "\"model\":\"" << m_model << "\",";
    payload << "\"temperature\":" << m_temperature << ",";
    payload << "\"max_tokens\":" << m_maxTokens << ",";
    payload << "\"messages\":[";
    
    // Add system message
    payload << "{\"role\":\"system\",\"content\":\"You are a helpful AI assistant integrated into an AIML chatbot. Provide clear, helpful responses.\"},";
    
    // Add conversation history
    for (size_t i = 0; i < conversationHistory.size() && i < 10; i += 2) {
        if (i + 1 < conversationHistory.size()) {
            payload << "{\"role\":\"user\",\"content\":\"" << escapeJsonString(conversationHistory[i]) << "\"},";
            payload << "{\"role\":\"assistant\",\"content\":\"" << escapeJsonString(conversationHistory[i + 1]) << "\"},";
        }
    }
    
    // Add current input
    payload << "{\"role\":\"user\",\"content\":\"" << escapeJsonString(input) << "\"}";
    
    payload << "]}";
    
    return payload.str();
}

string ChatGPT4oIntegration::escapeJsonString(const string& input) {
    string escaped;
    for (char c : input) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    return escaped;
}

string ChatGPT4oIntegration::extractResponseFromJson(const string& jsonResponse) {
    // Simple JSON parsing to extract the assistant's response
    // Look for: "choices":[{"message":{"content":"..."}}]
    
    size_t choicesPos = jsonResponse.find("\"choices\":");
    if (choicesPos == string::npos) return "";
    
    size_t contentPos = jsonResponse.find("\"content\":", choicesPos);
    if (contentPos == string::npos) return "";
    
    size_t startQuote = jsonResponse.find("\"", contentPos + 10);
    if (startQuote == string::npos) return "";
    
    size_t endQuote = startQuote + 1;
    int escapeCount = 0;
    while (endQuote < jsonResponse.length()) {
        if (jsonResponse[endQuote] == '\\') {
            escapeCount++;
        } else if (jsonResponse[endQuote] == '"' && escapeCount % 2 == 0) {
            break;
        } else {
            escapeCount = 0;
        }
        endQuote++;
    }
    
    if (endQuote >= jsonResponse.length()) return "";
    
    string content = jsonResponse.substr(startQuote + 1, endQuote - startQuote - 1);
    
    // Unescape basic JSON escapes
    size_t pos = 0;
    while ((pos = content.find("\\\"", pos)) != string::npos) {
        content.replace(pos, 2, "\"");
        pos += 1;
    }
    while ((pos = content.find("\\n", pos)) != string::npos) {
        content.replace(pos, 2, "\n");
        pos += 1;
    }
    while ((pos = content.find("\\t", pos)) != string::npos) {
        content.replace(pos, 2, "\t");
        pos += 1;
    }
    while ((pos = content.find("\\\\", pos)) != string::npos) {
        content.replace(pos, 2, "\\");
        pos += 1;
    }
    
    return content;
}

string ChatGPT4oIntegration::makeHttpRequest(const string& url, const string& headers, const string& payload) {
    // Simulated response for demonstration purposes
    // In production, this would make an actual HTTPS request to OpenAI API
    
    cout << "\n[ChatGPT-4o Simulation] Making API request..." << endl;
    cout << "URL: " << url << endl;
    cout << "Payload preview: " << payload.substr(0, 100) << "..." << endl;
    
    // Check if we have mock responses file
    ifstream mockFile("database/chatgpt4o_mock_responses.txt");
    if (mockFile.is_open()) {
        string line;
        while (getline(mockFile, line)) {
            if (!line.empty()) {
                // Return first non-empty line as mock response
                mockFile.close();
                return "{\"choices\":[{\"message\":{\"content\":\"" + line + "\"}}]}";
            }
        }
        mockFile.close();
    }
    
    // Default mock responses based on input patterns
    string lowerPayload = payload;
    transform(lowerPayload.begin(), lowerPayload.end(), lowerPayload.begin(), ::tolower);
    
    string mockResponse;
    if (lowerPayload.find("hello") != string::npos || lowerPayload.find("hi") != string::npos) {
        mockResponse = "Hello! I'm ChatGPT-4o integrated into this AIML chatbot. How can I help you today?";
    } else if (lowerPayload.find("weather") != string::npos) {
        mockResponse = "I'm sorry, I don't have access to real-time weather data. You might want to check a weather website or app for current conditions.";
    } else if (lowerPayload.find("what") != string::npos && lowerPayload.find("you") != string::npos) {
        mockResponse = "I'm ChatGPT-4o-latest, an advanced AI language model integrated into this AIML chatbot to provide enhanced responses when traditional patterns don't suffice.";
    } else if (lowerPayload.find("help") != string::npos) {
        mockResponse = "I can help you with a wide variety of topics including answering questions, having conversations, providing explanations, and more. What would you like to know?";
    } else {
        mockResponse = "I understand you're asking about something, but I'd need more context to provide a helpful response. Could you please elaborate on your question?";
    }
    
    // Return mock JSON response
    return "{\"choices\":[{\"message\":{\"content\":\"" + escapeJsonString(mockResponse) + "\"}}]}";
}

// SimpleHttpClient implementation (simplified for demo)
SimpleHttpClient::HttpResponse SimpleHttpClient::post(const string& url, const string& headers, const string& body) {
    HttpResponse response;
    response.success = false;
    response.statusCode = 0;
    
    // This would need proper HTTPS implementation with SSL/TLS
    // For now, we'll return a failure response
    response.body = "HTTPS client implementation required";
    
    return response;
}

string SimpleHttpClient::parseHostFromUrl(const string& url) {
    regex urlRegex(R"(https?://([^/]+))");
    smatch match;
    if (regex_search(url, match, urlRegex)) {
        return match[1].str();
    }
    return "";
}

string SimpleHttpClient::parsePathFromUrl(const string& url) {
    size_t pos = url.find('/', 8); // Skip "https://"
    if (pos != string::npos) {
        return url.substr(pos);
    }
    return "/";
}

int SimpleHttpClient::parsePortFromUrl(const string& url) {
    return isHttps(url) ? 443 : 80;
}

bool SimpleHttpClient::isHttps(const string& url) {
    return url.substr(0, 5) == "https";
}