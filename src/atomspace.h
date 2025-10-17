#ifndef __ATOMSPACE_H__
#define __ATOMSPACE_H__

#include "atom.h"
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <memory>
#include <functional>

using namespace std;

namespace opencog {
    
    /**
     * AtomSpace - Hypergraph database for knowledge representation
     * This is the core data structure that stores all atoms and their relationships
     */
    class AtomSpace {
    public:
        AtomSpace();
        virtual ~AtomSpace();
        
        // Atom creation and management
        shared_ptr<Atom> addAtom(shared_ptr<Atom> atom);
        shared_ptr<ConceptNode> addConceptNode(const string& name);
        shared_ptr<WordNode> addWordNode(const string& word);
        shared_ptr<SentenceNode> addSentenceNode(const string& sentence);
        shared_ptr<InheritanceLink> addInheritanceLink(shared_ptr<Atom> child, shared_ptr<Atom> parent);
        shared_ptr<ImplicationLink> addImplicationLink(shared_ptr<Atom> antecedent, shared_ptr<Atom> consequent);
        
        // Atom retrieval
        shared_ptr<Atom> getAtom(AtomType type, const string& name) const;
        vector<shared_ptr<Atom>> getAtomsByType(AtomType type) const;
        vector<shared_ptr<Atom>> getAtomsByName(const string& name) const;
        vector<shared_ptr<Atom>> getAllAtoms() const;
        
        // Pattern matching and search
        vector<shared_ptr<Atom>> findAtomsMatching(const string& pattern) const;
        vector<shared_ptr<Atom>> findSimilarConcepts(const string& concept, double threshold = 0.7) const;
        
        // Knowledge queries
        bool hasInheritance(const string& child, const string& parent) const;
        vector<string> getParentConcepts(const string& concept) const;
        vector<string> getChildConcepts(const string& concept) const;
        
        // AIML integration
        string generateAIMLPattern(const string& input) const;
        vector<string> getRelatedResponses(const string& input) const;
        void learnFromAIMLCategory(const string& pattern, const string& template_str);
        
        // Statistics and debugging
        size_t size() const { return m_atoms.size(); }
        void printStatistics() const;
        void clear();
        
        // Persistence
        bool saveToFile(const string& filename) const;
        bool loadFromFile(const string& filename);
        
    private:
        // Internal data structures
        unordered_set<shared_ptr<Atom>> m_atoms;
        unordered_map<string, shared_ptr<Atom>> m_nameIndex;
        unordered_map<AtomType, vector<shared_ptr<Atom>>> m_typeIndex;
        
        // Helper methods
        void indexAtom(shared_ptr<Atom> atom);
        void removeFromIndex(shared_ptr<Atom> atom);
        double calculateSimilarity(const string& str1, const string& str2) const;
        string extractKeywords(const string& text) const;
        vector<string> tokenize(const string& text) const;
        
        // Pattern matching helpers
        bool matchesPattern(shared_ptr<Atom> atom, const string& pattern) const;
        
        // Learning and inference
        void inferRelationships();
        void updateTruthValues();
    };
    
    /**
     * AtomSpaceManager - Singleton for global AtomSpace access
     */
    class AtomSpaceManager {
    public:
        static AtomSpace& getInstance();
        static void reset();
        
    private:
        static unique_ptr<AtomSpace> s_instance;
        AtomSpaceManager() = default;
    };
}

#endif // __ATOMSPACE_H__