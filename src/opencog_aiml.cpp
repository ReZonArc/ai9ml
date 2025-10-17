#include "opencog_aiml.h"
#include "atomspace.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>

using namespace opencog_aiml;

// OpenCogAIMLIntegration implementation
OpenCogAIMLIntegration::OpenCogAIMLIntegration() 
    : m_atomSpace(AtomSpaceManager::getInstance()) {
    
    // Initialize with basic knowledge
    buildConceptHierarchy();
}

OpenCogAIMLIntegration::~OpenCogAIMLIntegration() {
}

void OpenCogAIMLIntegration::initializeFromCategories(const vector<Category*>& categories) {
    cout << "Initializing OpenCog from " << categories.size() << " AIML categories..." << endl;
    
    for (const auto& category : categories) {
        if (category) {
            string pattern = category->pattern() ? category->pattern()->toString() : "";
            string template_str = category->templ() ? category->templ()->toString() : "";
            
            if (!pattern.empty() && !template_str.empty()) {
                // Learn from this category
                m_atomSpace.learnFromAIMLCategory(pattern, template_str);
                
                // Create concept relationships
                establishRelationships(pattern, template_str);
            }
        }
    }
    
    cout << "OpenCog initialization complete. AtomSpace size: " << m_atomSpace.size() << endl;
}

string OpenCogAIMLIntegration::enhancedPatternMatch(const string& input, const vector<Category*>& categories) {
    // Use cognitive pattern matching instead of simple string matching
    auto rankedCategories = rankCategoriesBySimilarity(input, categories);
    
    if (!rankedCategories.empty()) {
        auto bestCategory = rankedCategories[0];
        if (bestCategory && bestCategory->templ()) {
            string response = bestCategory->templ()->toString();
            
            // Enhance response with knowledge
            response = generateKnowledgeBasedResponse(input);
            if (response.empty() && bestCategory->templ()) {
                response = bestCategory->templ()->toString();
            }
            
            return response;
        }
    }
    
    return generateKnowledgeBasedResponse(input);
}

string OpenCogAIMLIntegration::generateKnowledgeBasedResponse(const string& input) {
    // Extract concepts from input
    auto concepts = extractConcepts(input);
    
    if (concepts.empty()) {
        return "";
    }
    
    // Look for related responses in AtomSpace
    auto relatedResponses = m_atomSpace.getRelatedResponses(input);
    
    if (!relatedResponses.empty()) {
        // Choose best response based on context
        for (const auto& response : relatedResponses) {
            if (!response.empty() && response != input) {
                return response;
            }
        }
    }
    
    // Generate response based on concept relationships
    for (const auto& concept : concepts) {
        auto parents = m_atomSpace.getParentConcepts(concept);
        if (!parents.empty()) {
            return "I know that " + concept + " is related to " + parents[0] + ".";
        }
        
        auto children = m_atomSpace.getChildConcepts(concept);
        if (!children.empty()) {
            return "When you mention " + concept + ", I think of " + children[0] + ".";
        }
    }
    
    return "";
}

void OpenCogAIMLIntegration::learnFromInteraction(const string& input, const string& response, double satisfaction) {
    // Update conversation history
    m_conversationHistory.push_back("User: " + input);
    m_conversationHistory.push_back("Bot: " + response);
    
    // Keep only recent history
    if (m_conversationHistory.size() > 20) {
        m_conversationHistory.erase(m_conversationHistory.begin(), m_conversationHistory.begin() + 2);
    }
    
    // Learn new patterns if satisfaction is high
    if (satisfaction > 0.7) {
        m_atomSpace.learnFromAIMLCategory(input, response);
        
        // Update concept relationships
        auto inputConcepts = extractConcepts(input);
        auto responseConcepts = extractConcepts(response);
        
        for (const auto& inConcept : inputConcepts) {
            for (const auto& outConcept : responseConcepts) {
                auto inNode = m_atomSpace.addConceptNode(inConcept);
                auto outNode = m_atomSpace.addConceptNode(outConcept);
                m_atomSpace.addImplicationLink(inNode, outNode);
            }
        }
    }
    
    // Update context
    updateContext(input, response);
}

void OpenCogAIMLIntegration::updateContext(const string& input, const string& response) {
    updateTopicContext(input);
    updateTopicContext(response);
}

string OpenCogAIMLIntegration::getContextualResponse(const string& input) {
    // Consider current topic and conversation history
    string contextualInput = input;
    
    if (!m_currentTopic.empty()) {
        contextualInput = m_currentTopic + " " + input;
    }
    
    return generateKnowledgeBasedResponse(contextualInput);
}

vector<string> OpenCogAIMLIntegration::extractConcepts(const string& input) {
    vector<string> concepts;
    
    // Tokenize input
    regex wordRegex(R"(\b[a-zA-Z]{3,}\b)");
    sregex_iterator begin(input.begin(), input.end(), wordRegex);
    sregex_iterator end;
    
    for (auto it = begin; it != end; ++it) {
        string word = it->str();
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        // Filter out common words
        if (word != "the" && word != "and" && word != "but" && 
            word != "for" && word != "are" && word != "was" &&
            word != "you" && word != "what" && word != "how") {
            concepts.push_back(word);
        }
    }
    
    return concepts;
}

string OpenCogAIMLIntegration::expandPattern(const string& pattern) {
    // Use AtomSpace to expand pattern with related concepts
    auto expandedPattern = m_atomSpace.generateAIMLPattern(pattern);
    return expandedPattern.empty() ? pattern : expandedPattern;
}

void OpenCogAIMLIntegration::printKnowledgeStats() {
    cout << "\n=== OpenCog Knowledge Statistics ===" << endl;
    m_atomSpace.printStatistics();
    cout << "Current topic: " << (m_currentTopic.empty() ? "none" : m_currentTopic) << endl;
    cout << "Conversation history length: " << m_conversationHistory.size() << endl;
    cout << "================================\n" << endl;
}

void OpenCogAIMLIntegration::createConceptsFromPattern(const string& pattern) {
    auto concepts = extractConcepts(pattern);
    for (const auto& concept : concepts) {
        m_atomSpace.addConceptNode(concept);
    }
}

void OpenCogAIMLIntegration::createConceptsFromTemplate(const string& template_str) {
    auto concepts = extractConcepts(template_str);
    for (const auto& concept : concepts) {
        m_atomSpace.addConceptNode(concept);
    }
}

void OpenCogAIMLIntegration::establishRelationships(const string& pattern, const string& template_str) {
    auto patternConcepts = extractConcepts(pattern);
    auto templateConcepts = extractConcepts(template_str);
    
    // Create implication links between pattern and template concepts
    for (const auto& pConcept : patternConcepts) {
        for (const auto& tConcept : templateConcepts) {
            auto pNode = m_atomSpace.addConceptNode(pConcept);
            auto tNode = m_atomSpace.addConceptNode(tConcept);
            m_atomSpace.addImplicationLink(pNode, tNode);
        }
    }
}

double OpenCogAIMLIntegration::calculateSemanticSimilarity(const string& input, const string& pattern) {
    auto inputConcepts = extractConcepts(input);
    auto patternConcepts = extractConcepts(pattern);
    
    if (inputConcepts.empty() || patternConcepts.empty()) {
        return 0.0;
    }
    
    double totalSimilarity = 0.0;
    int matches = 0;
    
    for (const auto& inputConcept : inputConcepts) {
        for (const auto& patternConcept : patternConcepts) {
            // Check for exact match
            if (inputConcept == patternConcept) {
                totalSimilarity += 1.0;
                matches++;
            } else {
                // Check for inheritance relationships
                if (m_atomSpace.hasInheritance(inputConcept, patternConcept) ||
                    m_atomSpace.hasInheritance(patternConcept, inputConcept)) {
                    totalSimilarity += 0.8;
                    matches++;
                } else {
                    // Use similarity calculation
                    auto similar = m_atomSpace.findSimilarConcepts(inputConcept, 0.6);
                    for (const auto& simAtom : similar) {
                        if (simAtom->getName() == patternConcept) {
                            totalSimilarity += 0.6;
                            matches++;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    return matches > 0 ? totalSimilarity / matches : 0.0;
}

vector<Category*> OpenCogAIMLIntegration::rankCategoriesBySimilarity(const string& input, const vector<Category*>& categories) {
    vector<pair<Category*, double>> scoredCategories;
    
    for (const auto& category : categories) {
        if (category && category->pattern()) {
            string pattern = category->pattern()->toString();
            double score = calculateSemanticSimilarity(input, pattern);
            scoredCategories.push_back({category, score});
        }
    }
    
    // Sort by score (highest first)
    sort(scoredCategories.begin(), scoredCategories.end(), 
         [](const pair<Category*, double>& a, const pair<Category*, double>& b) {
             return a.second > b.second;
         });
    
    vector<Category*> rankedCategories;
    for (const auto& scored : scoredCategories) {
        rankedCategories.push_back(scored.first);
    }
    
    return rankedCategories;
}

void OpenCogAIMLIntegration::updateTopicContext(const string& input) {
    auto concepts = extractConcepts(input);
    
    if (!concepts.empty()) {
        // Use the most significant concept as current topic
        string newTopic = concepts[0];
        
        // Check if this concept has strong relationships in AtomSpace
        auto parents = m_atomSpace.getParentConcepts(newTopic);
        if (!parents.empty()) {
            newTopic = parents[0];  // Use parent concept as broader topic
        }
        
        m_currentTopic = newTopic;
    }
}

string OpenCogAIMLIntegration::inferTopicFromContext() {
    if (m_conversationHistory.size() >= 4) {
        // Analyze recent conversation for topic
        string recentContext = m_conversationHistory.back();
        auto concepts = extractConcepts(recentContext);
        
        if (!concepts.empty()) {
            return concepts[0];
        }
    }
    
    return m_currentTopic;
}

void OpenCogAIMLIntegration::buildConceptHierarchy() {
    // Create basic concept hierarchy
    auto animalConcept = m_atomSpace.addConceptNode("animal");
    auto dogConcept = m_atomSpace.addConceptNode("dog");
    auto catConcept = m_atomSpace.addConceptNode("cat");
    
    m_atomSpace.addInheritanceLink(dogConcept, animalConcept);
    m_atomSpace.addInheritanceLink(catConcept, animalConcept);
    
    auto emotionConcept = m_atomSpace.addConceptNode("emotion");
    auto happyConcept = m_atomSpace.addConceptNode("happy");
    auto sadConcept = m_atomSpace.addConceptNode("sad");
    
    m_atomSpace.addInheritanceLink(happyConcept, emotionConcept);
    m_atomSpace.addInheritanceLink(sadConcept, emotionConcept);
    
    auto colorConcept = m_atomSpace.addConceptNode("color");
    auto redConcept = m_atomSpace.addConceptNode("red");
    auto blueConcept = m_atomSpace.addConceptNode("blue");
    
    m_atomSpace.addInheritanceLink(redConcept, colorConcept);
    m_atomSpace.addInheritanceLink(blueConcept, colorConcept);
}

// CognitivePattern implementation
CognitivePattern::CognitivePattern(const string& pattern, shared_ptr<SentenceNode> sentenceNode)
    : m_pattern(pattern), m_sentenceNode(sentenceNode) {
}

bool CognitivePattern::matches(const string& input, AtomSpace& atomSpace) {
    return getSimilarityScore(input, atomSpace) > 0.5;
}

double CognitivePattern::getSimilarityScore(const string& input, AtomSpace& atomSpace) {
    // Simple similarity calculation - can be enhanced
    if (m_pattern == input) return 1.0;
    
    // Check for pattern match using AtomSpace knowledge
    auto relatedAtoms = atomSpace.findAtomsMatching(input);
    
    for (const auto& atom : relatedAtoms) {
        if (atom->getName() == m_pattern) {
            return 0.9;
        }
    }
    
    return 0.0;
}

void CognitivePattern::extractConcepts(AtomSpace& atomSpace) {
    // Extract concepts from pattern and store them
    regex wordRegex(R"(\b[a-zA-Z]{3,}\b)");
    sregex_iterator begin(m_pattern.begin(), m_pattern.end(), wordRegex);
    sregex_iterator end;
    
    for (auto it = begin; it != end; ++it) {
        string word = it->str();
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        auto conceptNode = atomSpace.addConceptNode(word);
        m_concepts.push_back(conceptNode);
    }
}

// CognitiveTemplate implementation
CognitiveTemplate::CognitiveTemplate(const string& template_str, shared_ptr<SentenceNode> sentenceNode)
    : m_template(template_str), m_sentenceNode(sentenceNode) {
}

string CognitiveTemplate::generate(const string& input, AtomSpace& atomSpace) {
    return substituteWithKnowledge(m_template, input, atomSpace);
}

string CognitiveTemplate::substituteWithKnowledge(const string& template_str, const string& input, AtomSpace& atomSpace) {
    string result = template_str;
    
    // Simple substitution - can be enhanced with more sophisticated NLG
    if (result.find("*") != string::npos) {
        // Replace wildcards with input concepts
        regex wordRegex(R"(\b[a-zA-Z]{3,}\b)");
        sregex_iterator begin(input.begin(), input.end(), wordRegex);
        sregex_iterator end;
        
        for (auto it = begin; it != end; ++it) {
            string word = it->str();
            size_t pos = result.find("*");
            if (pos != string::npos) {
                result.replace(pos, 1, word);
                break;
            }
        }
    }
    
    return result;
}