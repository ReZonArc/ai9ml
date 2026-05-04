#include "atomspace.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <cctype>
#include <unordered_set>
#include <map>

using namespace opencog;

// AtomSpace implementation
AtomSpace::AtomSpace() {
}

AtomSpace::~AtomSpace() {
    clear();
}

shared_ptr<Atom> AtomSpace::addAtom(shared_ptr<Atom> atom) {
    if (!atom) return nullptr;
    
    // Check if atom already exists
    auto existing = getAtom(atom->getType(), atom->getName());
    if (existing) {
        // Update truth value if higher
        if (atom->getTruthValue() > existing->getTruthValue()) {
            existing->setTruthValue(atom->getTruthValue());
        }
        return existing;
    }
    
    // Add new atom
    m_atoms.insert(atom);
    indexAtom(atom);
    
    return atom;
}

shared_ptr<ConceptNode> AtomSpace::addConceptNode(const string& name) {
    auto atom = make_shared<ConceptNode>(name);
    auto added = addAtom(atom);
    return dynamic_pointer_cast<ConceptNode>(added);
}

shared_ptr<WordNode> AtomSpace::addWordNode(const string& word) {
    auto atom = make_shared<WordNode>(word);
    auto added = addAtom(atom);
    return dynamic_pointer_cast<WordNode>(added);
}

shared_ptr<SentenceNode> AtomSpace::addSentenceNode(const string& sentence) {
    auto atom = make_shared<SentenceNode>(sentence);
    auto added = addAtom(atom);
    return dynamic_pointer_cast<SentenceNode>(added);
}

shared_ptr<InheritanceLink> AtomSpace::addInheritanceLink(shared_ptr<Atom> child, shared_ptr<Atom> parent) {
    auto link = make_shared<InheritanceLink>(child, parent);
    auto added = addAtom(link);
    return dynamic_pointer_cast<InheritanceLink>(added);
}

shared_ptr<ImplicationLink> AtomSpace::addImplicationLink(shared_ptr<Atom> antecedent, shared_ptr<Atom> consequent) {
    auto link = make_shared<ImplicationLink>(antecedent, consequent);
    auto added = addAtom(link);
    return dynamic_pointer_cast<ImplicationLink>(added);
}

shared_ptr<Atom> AtomSpace::getAtom(AtomType type, const string& name) const {
    string key = to_string(type) + ":" + name;
    auto it = m_nameIndex.find(key);
    return (it != m_nameIndex.end()) ? it->second : nullptr;
}

vector<shared_ptr<Atom>> AtomSpace::getAtomsByType(AtomType type) const {
    auto it = m_typeIndex.find(type);
    return (it != m_typeIndex.end()) ? it->second : vector<shared_ptr<Atom>>();
}

vector<shared_ptr<Atom>> AtomSpace::getAtomsByName(const string& name) const {
    vector<shared_ptr<Atom>> result;
    for (const auto& atom : m_atoms) {
        if (atom->getName() == name) {
            result.push_back(atom);
        }
    }
    return result;
}

vector<shared_ptr<Atom>> AtomSpace::getAllAtoms() const {
    return vector<shared_ptr<Atom>>(m_atoms.begin(), m_atoms.end());
}

vector<shared_ptr<Atom>> AtomSpace::findAtomsMatching(const string& pattern) const {
    vector<shared_ptr<Atom>> result;
    
    // Simple pattern matching - can be enhanced with regex
    for (const auto& atom : m_atoms) {
        if (matchesPattern(atom, pattern)) {
            result.push_back(atom);
        }
    }
    
    return result;
}

vector<shared_ptr<Atom>> AtomSpace::findSimilarConcepts(const string& concept, double threshold) const {
    vector<shared_ptr<Atom>> result;
    
    for (const auto& atom : m_atoms) {
        if (atom->getType() == CONCEPT_NODE) {
            double similarity = calculateSimilarity(concept, atom->getName());
            if (similarity >= threshold) {
                result.push_back(atom);
            }
        }
    }
    
    return result;
}

bool AtomSpace::hasInheritance(const string& child, const string& parent) const {
    auto childAtom = getAtom(CONCEPT_NODE, child);
    auto parentAtom = getAtom(CONCEPT_NODE, parent);
    
    if (!childAtom || !parentAtom) return false;
    
    // Check for direct inheritance links
    for (const auto& atom : m_atoms) {
        if (atom->getType() == INHERITANCE_LINK) {
            auto link = dynamic_pointer_cast<InheritanceLink>(atom);
            if (link && link->getChild() == childAtom && link->getParent() == parentAtom) {
                return true;
            }
        }
    }
    
    return false;
}

vector<string> AtomSpace::getParentConcepts(const string& concept) const {
    vector<string> parents;
    auto conceptAtom = getAtom(CONCEPT_NODE, concept);
    
    if (!conceptAtom) return parents;
    
    for (const auto& atom : m_atoms) {
        if (atom->getType() == INHERITANCE_LINK) {
            auto link = dynamic_pointer_cast<InheritanceLink>(atom);
            if (link && link->getChild() == conceptAtom) {
                parents.push_back(link->getParent()->getName());
            }
        }
    }
    
    return parents;
}

vector<string> AtomSpace::getChildConcepts(const string& concept) const {
    vector<string> children;
    auto conceptAtom = getAtom(CONCEPT_NODE, concept);
    
    if (!conceptAtom) return children;
    
    for (const auto& atom : m_atoms) {
        if (atom->getType() == INHERITANCE_LINK) {
            auto link = dynamic_pointer_cast<InheritanceLink>(atom);
            if (link && link->getParent() == conceptAtom) {
                children.push_back(link->getChild()->getName());
            }
        }
    }
    
    return children;
}

string AtomSpace::generateAIMLPattern(const string& input) const {
    // Extract keywords and generate AIML pattern
    vector<string> keywords = tokenize(input);
    stringstream pattern;
    
    for (size_t i = 0; i < keywords.size(); ++i) {
        if (i > 0) pattern << " ";
        
        // Check if keyword exists as concept in AtomSpace
        auto concept = getAtom(CONCEPT_NODE, keywords[i]);
        if (concept) {
            // Use the concept directly
            pattern << keywords[i];
        } else {
            // Look for similar concepts
            auto similar = findSimilarConcepts(keywords[i], 0.8);
            if (!similar.empty()) {
                pattern << similar[0]->getName();
            } else {
                pattern << "*";
            }
        }
    }
    
    return pattern.str();
}

vector<string> AtomSpace::getRelatedResponses(const string& input) const {
    vector<string> responses;
    
    // Find concepts related to input
    auto keywords = tokenize(input);
    for (const auto& keyword : keywords) {
        auto similar = findSimilarConcepts(keyword, 0.6);
        for (const auto& concept : similar) {
            // Find implication links that could generate responses
            for (const auto& atom : m_atoms) {
                if (atom->getType() == IMPLICATION_LINK) {
                    auto link = dynamic_pointer_cast<ImplicationLink>(atom);
                    if (link && link->getAntecedent() == concept) {
                        responses.push_back(link->getConsequent()->getName());
                    }
                }
            }
        }
    }
    
    return responses;
}

void AtomSpace::learnFromAIMLCategory(const string& pattern, const string& template_str) {
    // Create sentence nodes for pattern and template
    auto patternNode = addSentenceNode(pattern);
    auto templateNode = addSentenceNode(template_str);
    
    // Create implication link
    addImplicationLink(patternNode, templateNode);
    
    // Extract and add concept nodes from pattern
    string keywords = extractKeywords(pattern);
    auto keywordTokens = tokenize(keywords);
    for (const auto& keyword : keywordTokens) {
        addConceptNode(keyword);
        
        // Create inheritance links for related concepts
        if (keyword.find("animal") != string::npos || keyword.find("pet") != string::npos) {
            auto animalConcept = addConceptNode("animal");
            auto specificConcept = addConceptNode(keyword);
            addInheritanceLink(specificConcept, animalConcept);
        }
    }
}

void AtomSpace::printStatistics() const {
    cout << "AtomSpace Statistics:" << endl;
    cout << "Total atoms: " << m_atoms.size() << endl;
    
    map<AtomType, int> typeCounts;
    for (const auto& atom : m_atoms) {
        typeCounts[atom->getType()]++;
    }
    
    for (const auto& pair : typeCounts) {
        cout << "Type " << pair.first << ": " << pair.second << " atoms" << endl;
    }
}

void AtomSpace::clear() {
    m_atoms.clear();
    m_nameIndex.clear();
    m_typeIndex.clear();
}

void AtomSpace::indexAtom(shared_ptr<Atom> atom) {
    if (!atom) return;
    
    // Add to name index
    string key = to_string(atom->getType()) + ":" + atom->getName();
    m_nameIndex[key] = atom;
    
    // Add to type index
    m_typeIndex[atom->getType()].push_back(atom);
}

void AtomSpace::removeFromIndex(shared_ptr<Atom> atom) {
    if (!atom) return;
    
    // Remove from name index
    string key = to_string(atom->getType()) + ":" + atom->getName();
    m_nameIndex.erase(key);
    
    // Remove from type index
    auto& typeVec = m_typeIndex[atom->getType()];
    typeVec.erase(remove(typeVec.begin(), typeVec.end(), atom), typeVec.end());
}

double AtomSpace::calculateSimilarity(const string& str1, const string& str2) const {
    // Simple Levenshtein distance-based similarity
    if (str1.empty() || str2.empty()) return 0.0;
    if (str1 == str2) return 1.0;
    
    // Convert to lowercase for comparison
    string s1 = str1, s2 = str2;
    transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
    transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
    
    // Check for substring matches
    if (s1.find(s2) != string::npos || s2.find(s1) != string::npos) {
        return 0.8;
    }
    
    // Simple character overlap calculation
    int common = 0;
    for (char c : s1) {
        if (s2.find(c) != string::npos) common++;
    }
    
    return (double)common / max(s1.length(), s2.length());
}

string AtomSpace::extractKeywords(const string& text) const {
    // Extract meaningful keywords from text
    regex wordRegex(R"(\b[a-zA-Z]+\b)");
    sregex_iterator begin(text.begin(), text.end(), wordRegex);
    sregex_iterator end;
    
    vector<string> keywords;
    for (auto it = begin; it != end; ++it) {
        string word = it->str();
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        // Filter out common stop words
        if (word.length() > 2 && 
            word != "the" && word != "and" && word != "but" && 
            word != "for" && word != "are" && word != "was") {
            keywords.push_back(word);
        }
    }
    
    stringstream result;
    for (size_t i = 0; i < keywords.size(); ++i) {
        if (i > 0) result << " ";
        result << keywords[i];
    }
    
    return result.str();
}

vector<string> AtomSpace::tokenize(const string& text) const {
    vector<string> tokens;
    stringstream ss(text);
    string token;
    
    while (ss >> token) {
        // Remove punctuation and convert to lowercase
        token.erase(remove_if(token.begin(), token.end(), ::ispunct), token.end());
        transform(token.begin(), token.end(), token.begin(), ::tolower);
        
        if (!token.empty() && token.length() > 1) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

bool AtomSpace::matchesPattern(shared_ptr<Atom> atom, const string& pattern) const {
    if (!atom) return false;
    
    string atomStr = atom->getName();
    transform(atomStr.begin(), atomStr.end(), atomStr.begin(), ::tolower);
    
    string patternStr = pattern;
    transform(patternStr.begin(), patternStr.end(), patternStr.begin(), ::tolower);
    
    return atomStr.find(patternStr) != string::npos;
}

bool AtomSpace::saveToFile(const string& filename) const {
    // Placeholder – full serialisation not required for NSVD operation.
    (void)filename;
    return false;
}

bool AtomSpace::loadFromFile(const string& filename) {
    (void)filename;
    return false;
}

// ---------------------------------------------------------------------------
// NSVD additions
// ---------------------------------------------------------------------------

shared_ptr<SimilarityLink> AtomSpace::addSimilarityLink(
    shared_ptr<Atom> atom1, shared_ptr<Atom> atom2, double weight)
{
    if (!atom1 || !atom2) return nullptr;
    auto link = make_shared<SimilarityLink>(atom1, atom2);
    link->setTruthValue(weight);
    auto added = addAtom(link);
    return dynamic_pointer_cast<SimilarityLink>(added);
}

void AtomSpace::propagateTrust(shared_ptr<Atom> origin,
                                double fraction,
                                int    maxDepth)
{
    if (!origin || maxDepth <= 0 || fraction <= 0.0) return;
    unordered_set<size_t> visited;
    propagateTrustImpl(origin, fraction, 0, maxDepth, visited);
}

void AtomSpace::propagateTrustImpl(
    shared_ptr<Atom> atom,
    double           fraction,
    int              depth,
    int              maxDepth,
    unordered_set<size_t>& visited)
{
    if (!atom || depth > maxDepth) return;
    size_t h = atom->hashCode();
    if (visited.count(h)) return;
    visited.insert(h);

    double delta = atom->getTruthValue() * fraction;
    if (delta < 1e-4) return;

    // Propagate to all atoms reachable through links that mention this atom.
    for (const auto& a : m_atoms) {
        if (!a) continue;
        // Only propagate through Link atoms.
        auto lnk = dynamic_pointer_cast<Link>(a);
        if (!lnk) continue;
        for (const auto& out : lnk->getOutgoingSet()) {
            if (out == atom) {
                // Update truth value of the link itself.
                double newTV = min(1.0, lnk->getTruthValue() + delta);
                lnk->setTruthValue(newTV);
                // Recurse into the other atoms of this link.
                for (const auto& peer : lnk->getOutgoingSet()) {
                    if (peer && peer != atom) {
                        double peerTV = min(1.0, peer->getTruthValue() + delta * 0.5);
                        peer->setTruthValue(peerTV);
                        propagateTrustImpl(peer, fraction * 0.5, depth + 1, maxDepth, visited);
                    }
                }
                break;
            }
        }
    }
}

shared_ptr<ConceptNode> AtomSpace::interpolateConcepts(
    const string& conceptA, const string& conceptB, double weight)
{
    auto atomA = getAtom(CONCEPT_NODE, conceptA);
    auto atomB = getAtom(CONCEPT_NODE, conceptB);
    if (!atomA || !atomB) return nullptr;

    string blendName = conceptA + "~" + conceptB;
    auto existing = getAtom(CONCEPT_NODE, blendName);
    if (existing) return dynamic_pointer_cast<ConceptNode>(existing);

    double blendedTV = atomA->getTruthValue() * (1.0 - weight)
                     + atomB->getTruthValue() * weight;
    auto blendNode = make_shared<ConceptNode>(blendName);
    blendNode->setTruthValue(blendedTV);
    addAtom(blendNode);

    // Connect with a SimilarityLink to both parents.
    addSimilarityLink(blendNode, atomA, blendedTV);
    addSimilarityLink(blendNode, atomB, blendedTV);

    return blendNode;
}

void AtomSpace::garbageCollectBlends(double minConfidence) {
    vector<shared_ptr<Atom>> toRemove;
    for (const auto& atom : m_atoms) {
        if (!atom) continue;
        if (atom->getName().find('~') != string::npos &&
            atom->getTruthValue() < minConfidence)
        {
            toRemove.push_back(atom);
        }
    }
    for (const auto& atom : toRemove) {
        removeFromIndex(atom);
        m_atoms.erase(atom);
    }
}

// ---------------------------------------------------------------------------
// Persistence stubs (declared in header, implemented here)
// ---------------------------------------------------------------------------
unique_ptr<AtomSpace> AtomSpaceManager::s_instance = nullptr;

AtomSpace& AtomSpaceManager::getInstance() {
    if (!s_instance) {
        s_instance = unique_ptr<AtomSpace>(new AtomSpace());
    }
    return *s_instance;
}

void AtomSpaceManager::reset() {
    s_instance.reset();
}