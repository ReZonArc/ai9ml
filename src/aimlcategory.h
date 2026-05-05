#ifndef __AIMLCATEGORY_H__
#define __AIMLCATEGORY_H__

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <map>
#include <algorithm>
#include "strings.h"
#include "aimlelement.h"
#include "aimltopic.h"
#include "aimlpattern.h"
#include "aimltemplate.h"
#include "aimlthat.h"

using namespace std;

namespace aiml {
    class Pattern;
    class Template;
    class That;
    class Topic;

    /**
     * TruthValue — Bayesian confidence attached to each AIML category.
     *
     * strength   : historical match quality (0.0–1.0)
     * confidence : sample-count-normalised certainty (0.0–1.0)
     * immutable  : true for categories loaded directly from AIML files
     *              (anchored; never mutated by online learning)
     *
     * weight()   : composite score used for variational pattern ranking
     */
    struct TruthValue {
        double strength;
        double confidence;
        bool   immutable;

        TruthValue(double s = 1.0, double c = 1.0, bool imm = true)
            : strength(s), confidence(c), immutable(imm) {}

        double weight() const { return strength * confidence; }

        // Called every time this category is matched and produces a used response.
        void reinforce(double amount = 0.05) {
            if (immutable) return;
            confidence = min(1.0, confidence + amount);
            strength   = min(1.0, strength   + amount * 0.5);
        }

        // Called once per conversation turn to apply recency decay.
        void decay(double factor = 0.99) {
            if (immutable) return;
            confidence = max(0.0, confidence * factor);
        }
    };

    class Category : public AIMLElement {
    public:
        Category()
            : m_pPattern(nullptr), m_tTemplate(nullptr),
              m_tThat(nullptr), m_tTopic(nullptr),
              m_truthValue(1.0, 1.0, true) {}

        Category(Pattern* pattern, That* that, Topic* topic, Template* templ)
            : m_pPattern(pattern), m_tThat(that), m_tTopic(topic), m_tTemplate(templ),
              m_truthValue(1.0, 1.0, true) {}

        Category(Pattern* pattern, Template* templ);

        // Construct a learnable (soft) category with custom TruthValue.
        Category(Pattern* pattern, Template* templ, const TruthValue& tv)
            : m_pPattern(pattern), m_tTemplate(templ),
              m_tThat(nullptr), m_tTopic(nullptr),
              m_truthValue(tv) {}

        ~Category() {
            delete(m_pPattern);
            delete(m_tTemplate);
            delete(m_tThat);
            delete(m_tTopic);
        }

        vector<string> getMatchPath();
        void appendChild(AIMLElement* child);
        void appendChildren(vector<AIMLElement*> children);

        void setTopic(Topic* topic);
        string toString();

        Pattern* pattern();
        Template* templ();

        // TruthValue accessors
        const TruthValue& getTruthValue() const { return m_truthValue; }
        TruthValue&       getTruthValue()       { return m_truthValue; }
        void setTruthValue(const TruthValue& tv) { m_truthValue = tv; }

        // Convenience wrappers
        void reinforceMatch(double amount = 0.05) { m_truthValue.reinforce(amount); }
        void decayConfidence(double factor = 0.99) { m_truthValue.decay(factor); }

    private:
        Pattern*    m_pPattern;
        Template*   m_tTemplate;
        That*       m_tThat;
        Topic*      m_tTopic;
        TruthValue  m_truthValue;
    };
}

#endif
