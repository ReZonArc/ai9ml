#ifndef __WORKFLOW_ENGINE_H__
#define __WORKFLOW_ENGINE_H__

#include "logic_meta_patterns.h"
#include <string>
#include <vector>
#include <map>

using namespace std;

namespace workflow_engine {

    struct WorkflowStep {
        logic_meta_patterns::MetaPattern metaPattern;
        map<string, string> boundVars;
        bool completed;

        WorkflowStep() : completed(false) {}
        explicit WorkflowStep(const logic_meta_patterns::MetaPattern& m)
            : metaPattern(m), completed(false) {}
    };

    struct WorkflowSequence {
        vector<WorkflowStep> steps;
        int currentIndex;
        map<string, string> variables;

        WorkflowSequence() : currentIndex(0) {}
    };

    struct WorkflowResult {
        string response;
        double score;
        double confidence;
        bool matched;
        bool completed;

        WorkflowResult()
            : score(0.0), confidence(0.0), matched(false), completed(false) {}
    };

    class WorkflowEngine {
    public:
        WorkflowEngine();
        ~WorkflowEngine() = default;

        WorkflowResult activate(logic_meta_patterns::LogicSystem system,
                                const string& initialInput);
        WorkflowResult advance(const string& input);

        bool isActive() const { return m_active; }
        void reset();

        logic_meta_patterns::LogicSystem getActiveSystem() const { return m_activeSystem; }
        int getCurrentStepIndex() const { return m_sequence.currentIndex; }
        size_t getStepCount() const { return m_sequence.steps.size(); }
        const map<string, string>& getVariables() const { return m_sequence.variables; }
        int getCompletionCount() const { return m_completionCount; }
        const map<string, int>& getActivationStats() const { return m_activationStats; }

        vector<logic_meta_patterns::MetaPattern> collectConsolidatedMetaPatterns(double threshold) const;

    private:
        bool m_active;
        logic_meta_patterns::LogicSystem m_activeSystem;
        WorkflowSequence m_sequence;
        string m_lastResponse;

        int m_completionCount;
        map<string, int> m_activationStats;
        map<string, double> m_patternConfidence;

        static bool wildcardMatch(const string& pattern,
                                  const string& input,
                                  vector<string>& captures);
        static string toUpper(const string& s);
        static string substituteStars(const string& templ,
                                      const vector<string>& captures,
                                      map<string, string>& varStore);
    };

} // namespace workflow_engine

#endif // __WORKFLOW_ENGINE_H__
