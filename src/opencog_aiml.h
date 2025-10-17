#ifndef __OPENCOG_AIML_H__
#define __OPENCOG_AIML_H__

#include "atomspace.h"
#include "aimlcategory.h"
#include "categorylist.h"
#include <string>
#include <vector>
#include <memory>

using namespace std;
using namespace aiml;
using namespace opencog;

namespace opencog_aiml {
    
    /**
     * OpenCogAIMLIntegration - Bridge between OpenCog AtomSpace and AIML
     * This class provides cognitive capabilities to the AIML chatbot
     */
    class OpenCogAIMLIntegration {
    public:
        OpenCogAIMLIntegration();
        virtual ~OpenCogAIMLIntegration();
        
        // Initialize with existing AIML categories
        void initializeFromCategories(const vector<Category*>& categories);
        
        // Enhanced pattern matching using AtomSpace
        string enhancedPatternMatch(const string& input, const vector<Category*>& categories);
        
        // Learning and adaptation
        void learnFromInteraction(const string& input, const string& response, double satisfaction = 1.0);
        
        // Knowledge-based response generation
        string generateKnowledgeBasedResponse(const string& input);
        
        // Context and memory
        void updateContext(const string& input, const string& response);
        string getContextualResponse(const string& input);
        
        // Concept understanding
        vector<string> extractConcepts(const string& input);
        string expandPattern(const string& pattern);
        
        // Statistics and debugging
        void printKnowledgeStats();
        
    private:
        AtomSpace& m_atomSpace;
        vector<string> m_conversationHistory;
        string m_currentTopic;
        
        // Helper methods
        void createConceptsFromPattern(const string& pattern);
        void createConceptsFromTemplate(const string& template_str);
        void establishRelationships(const string& pattern, const string& template_str);
        
        // Enhanced matching algorithms
        double calculateSemanticSimilarity(const string& input, const string& pattern);
        vector<Category*> rankCategoriesBySimilarity(const string& input, const vector<Category*>& categories);
        
        // Context management
        void updateTopicContext(const string& input);
        string inferTopicFromContext();
        
        // Knowledge extraction
        void extractKnowledgeFromAIML();
        void buildConceptHierarchy();
    };
    
    /**
     * CognitivePattern - Enhanced AIML pattern with cognitive capabilities
     */
    class CognitivePattern {
    public:
        CognitivePattern(const string& pattern, shared_ptr<SentenceNode> sentenceNode);
        
        bool matches(const string& input, AtomSpace& atomSpace);
        double getSimilarityScore(const string& input, AtomSpace& atomSpace);
        
        const string& getPattern() const { return m_pattern; }
        shared_ptr<SentenceNode> getSentenceNode() const { return m_sentenceNode; }
        
    private:
        string m_pattern;
        shared_ptr<SentenceNode> m_sentenceNode;
        vector<shared_ptr<ConceptNode>> m_concepts;
        
        void extractConcepts(AtomSpace& atomSpace);
    };
    
    /**
     * CognitiveTemplate - Enhanced AIML template with cognitive generation
     */
    class CognitiveTemplate {
    public:
        CognitiveTemplate(const string& template_str, shared_ptr<SentenceNode> sentenceNode);
        
        string generate(const string& input, AtomSpace& atomSpace);
        
        const string& getTemplate() const { return m_template; }
        shared_ptr<SentenceNode> getSentenceNode() const { return m_sentenceNode; }
        
    private:
        string m_template;
        shared_ptr<SentenceNode> m_sentenceNode;
        
        string substituteWithKnowledge(const string& template_str, const string& input, AtomSpace& atomSpace);
    };
}

#endif // __OPENCOG_AIML_H__