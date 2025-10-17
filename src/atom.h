#ifndef __ATOM_H__
#define __ATOM_H__

#include <string>
#include <vector>
#include <memory>
#include <unordered_set>

using namespace std;

namespace opencog {
    
    // Forward declaration
    class AtomSpace;
    
    // Atom types enumeration
    enum AtomType {
        ATOM,
        NODE,
        LINK,
        CONCEPT_NODE,
        WORD_NODE,
        SENTENCE_NODE,
        IMPLICATION_LINK,
        INHERITANCE_LINK,
        SIMILARITY_LINK,
        PATTERN_LINK
    };
    
    /**
     * Base Atom class - fundamental unit of knowledge in OpenCog
     * Represents concepts, relationships, and knowledge in the AtomSpace
     */
    class Atom {
    public:
        Atom(AtomType type, const string& name = "");
        virtual ~Atom();
        
        // Core properties
        AtomType getType() const { return m_type; }
        const string& getName() const { return m_name; }
        double getTruthValue() const { return m_truthValue; }
        void setTruthValue(double tv) { m_truthValue = tv; }
        
        // Relationships
        void addIncomingAtom(shared_ptr<Atom> atom);
        void removeIncomingAtom(shared_ptr<Atom> atom);
        const vector<shared_ptr<Atom>>& getIncomingSet() const { return m_incoming; }
        
        // Serialization
        virtual string toString() const;
        virtual string toAIMLPattern() const;
        
        // Equality and hashing
        bool operator==(const Atom& other) const;
        size_t hashCode() const;
        
    protected:
        AtomType m_type;
        string m_name;
        double m_truthValue;  // Confidence/strength value (0.0 to 1.0)
        vector<shared_ptr<Atom>> m_incoming;  // Atoms that reference this atom
        
    private:
        static size_t s_nextId;
        size_t m_id;
    };
    
    /**
     * Node - represents a concept or entity
     */
    class Node : public Atom {
    public:
        Node(AtomType type, const string& name);
        virtual ~Node() = default;
        
        string toString() const override;
        string toAIMLPattern() const override;
    };
    
    /**
     * Link - represents a relationship between atoms
     */
    class Link : public Atom {
    public:
        Link(AtomType type, const vector<shared_ptr<Atom>>& outgoing);
        virtual ~Link() = default;
        
        const vector<shared_ptr<Atom>>& getOutgoingSet() const { return m_outgoing; }
        size_t getArity() const { return m_outgoing.size(); }
        
        string toString() const override;
        string toAIMLPattern() const override;
        
    protected:
        vector<shared_ptr<Atom>> m_outgoing;  // Atoms this link points to
    };
    
    /**
     * ConceptNode - represents a concept in the knowledge base
     */
    class ConceptNode : public Node {
    public:
        ConceptNode(const string& name);
        virtual ~ConceptNode() = default;
        
        string toString() const override;
    };
    
    /**
     * WordNode - represents a word or linguistic element
     */
    class WordNode : public Node {
    public:
        WordNode(const string& word);
        virtual ~WordNode() = default;
        
        string toString() const override;
        string toAIMLPattern() const override;
    };
    
    /**
     * SentenceNode - represents a sentence or phrase
     */
    class SentenceNode : public Node {
    public:
        SentenceNode(const string& sentence);
        virtual ~SentenceNode() = default;
        
        string toString() const override;
        string toAIMLPattern() const override;
    };
    
    /**
     * InheritanceLink - represents "is-a" relationships
     */
    class InheritanceLink : public Link {
    public:
        InheritanceLink(shared_ptr<Atom> child, shared_ptr<Atom> parent);
        virtual ~InheritanceLink() = default;
        
        shared_ptr<Atom> getChild() const { return m_outgoing[0]; }
        shared_ptr<Atom> getParent() const { return m_outgoing[1]; }
        
        string toString() const override;
    };
    
    /**
     * ImplicationLink - represents logical implications
     */
    class ImplicationLink : public Link {
    public:
        ImplicationLink(shared_ptr<Atom> antecedent, shared_ptr<Atom> consequent);
        virtual ~ImplicationLink() = default;
        
        shared_ptr<Atom> getAntecedent() const { return m_outgoing[0]; }
        shared_ptr<Atom> getConsequent() const { return m_outgoing[1]; }
        
        string toString() const override;
    };
}

#endif // __ATOM_H__