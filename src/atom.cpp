#include "atom.h"
#include <sstream>
#include <algorithm>
#include <functional>

using namespace opencog;

// Static member initialization
size_t Atom::s_nextId = 1;

// Atom implementation
Atom::Atom(AtomType type, const string& name) 
    : m_type(type), m_name(name), m_truthValue(1.0), m_id(s_nextId++) {
}

Atom::~Atom() {
}

void Atom::addIncomingAtom(shared_ptr<Atom> atom) {
    if (find(m_incoming.begin(), m_incoming.end(), atom) == m_incoming.end()) {
        m_incoming.push_back(atom);
    }
}

void Atom::removeIncomingAtom(shared_ptr<Atom> atom) {
    m_incoming.erase(
        remove(m_incoming.begin(), m_incoming.end(), atom), 
        m_incoming.end()
    );
}

string Atom::toString() const {
    stringstream ss;
    ss << "Atom[" << m_type << "](" << m_name << ", tv=" << m_truthValue << ")";
    return ss.str();
}

string Atom::toAIMLPattern() const {
    // Default implementation - convert to AIML-compatible pattern
    if (!m_name.empty()) {
        return m_name;
    }
    return "*";
}

bool Atom::operator==(const Atom& other) const {
    return m_type == other.m_type && m_name == other.m_name;
}

size_t Atom::hashCode() const {
    return hash<string>{}(m_name) ^ (hash<int>{}(m_type) << 1);
}

// Node implementation
Node::Node(AtomType type, const string& name) : Atom(type, name) {
}

string Node::toString() const {
    stringstream ss;
    ss << "Node[" << m_type << "](" << m_name << ", tv=" << m_truthValue << ")";
    return ss.str();
}

string Node::toAIMLPattern() const {
    // Convert node to AIML pattern format
    if (!m_name.empty()) {
        // Replace spaces with underscores for AIML compatibility
        string pattern = m_name;
        replace(pattern.begin(), pattern.end(), ' ', '_');
        return pattern;
    }
    return "*";
}

// Link implementation
Link::Link(AtomType type, const vector<shared_ptr<Atom>>& outgoing) 
    : Atom(type, ""), m_outgoing(outgoing) {
    
    // Add this link to incoming sets of outgoing atoms
    // Note: We need to be careful about shared_ptr usage here
    for (auto& atom : m_outgoing) {
        if (atom) {
            // Create a weak reference to avoid circular dependencies
            // For now, skip incoming set management to avoid memory issues
            // atom->addIncomingAtom(shared_ptr<Atom>(this));
        }
    }
}

string Link::toString() const {
    stringstream ss;
    ss << "Link[" << m_type << "](";
    for (size_t i = 0; i < m_outgoing.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << (m_outgoing[i] ? m_outgoing[i]->toString() : "null");
    }
    ss << ", tv=" << m_truthValue << ")";
    return ss.str();
}

string Link::toAIMLPattern() const {
    // Convert link to AIML pattern - combine outgoing patterns
    stringstream ss;
    for (size_t i = 0; i < m_outgoing.size(); ++i) {
        if (i > 0) ss << " ";
        ss << (m_outgoing[i] ? m_outgoing[i]->toAIMLPattern() : "*");
    }
    return ss.str();
}

// ConceptNode implementation
ConceptNode::ConceptNode(const string& name) : Node(CONCEPT_NODE, name) {
}

string ConceptNode::toString() const {
    stringstream ss;
    ss << "ConceptNode(" << m_name << ", tv=" << m_truthValue << ")";
    return ss.str();
}

// WordNode implementation
WordNode::WordNode(const string& word) : Node(WORD_NODE, word) {
}

string WordNode::toString() const {
    stringstream ss;
    ss << "WordNode(" << m_name << ", tv=" << m_truthValue << ")";
    return ss.str();
}

string WordNode::toAIMLPattern() const {
    // Word nodes should match exactly in AIML patterns
    return m_name;
}

// SentenceNode implementation
SentenceNode::SentenceNode(const string& sentence) : Node(SENTENCE_NODE, sentence) {
}

string SentenceNode::toString() const {
    stringstream ss;
    ss << "SentenceNode(" << m_name << ", tv=" << m_truthValue << ")";
    return ss.str();
}

string SentenceNode::toAIMLPattern() const {
    // Convert sentence to AIML pattern with wildcards
    string pattern = m_name;
    
    // Replace common words with wildcards to make patterns more flexible
    size_t pos = 0;
    vector<string> wildcardWords = {"the", "a", "an", "is", "are", "was", "were"};
    
    for (const string& word : wildcardWords) {
        pos = 0;
        string searchWord = " " + word + " ";
        string replaceWord = " * ";
        while ((pos = pattern.find(searchWord, pos)) != string::npos) {
            pattern.replace(pos, searchWord.length(), replaceWord);
            pos += replaceWord.length();
        }
    }
    
    return pattern;
}

// InheritanceLink implementation
InheritanceLink::InheritanceLink(shared_ptr<Atom> child, shared_ptr<Atom> parent) 
    : Link(INHERITANCE_LINK, {child, parent}) {
}

string InheritanceLink::toString() const {
    stringstream ss;
    ss << "InheritanceLink(";
    ss << (m_outgoing[0] ? m_outgoing[0]->getName() : "null");
    ss << " -> ";
    ss << (m_outgoing[1] ? m_outgoing[1]->getName() : "null");
    ss << ", tv=" << m_truthValue << ")";
    return ss.str();
}

// ImplicationLink implementation
ImplicationLink::ImplicationLink(shared_ptr<Atom> antecedent, shared_ptr<Atom> consequent) 
    : Link(IMPLICATION_LINK, {antecedent, consequent}) {
}

string ImplicationLink::toString() const {
    stringstream ss;
    ss << "ImplicationLink(";
    ss << (m_outgoing[0] ? m_outgoing[0]->getName() : "null");
    ss << " => ";
    ss << (m_outgoing[1] ? m_outgoing[1]->getName() : "null");
    ss << ", tv=" << m_truthValue << ")";
    return ss.str();
}