#ifndef WORKFLOW_ENGINE_H_
#define WORKFLOW_ENGINE_H_

#include "logic_meta_patterns.h"
#include <string>
#include <vector>
#include <map>

namespace workflow_engine {

    struct WorkflowStep {
        logic_meta_patterns::MetaPattern metaPattern;
        std::map<std::string, std::string> boundVars;
        bool completed;

        WorkflowStep() : completed(false) {}
        explicit WorkflowStep(const logic_meta_patterns::MetaPattern& m)
            : metaPattern(m), completed(false) {}
    };

    struct WorkflowSequence {
        std::vector<WorkflowStep> steps;
        int currentIndex;
        std::map<std::string, std::string> variables;

        WorkflowSequence() : currentIndex(0) {}
    };

    struct WorkflowResult {
        std::string response;
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
                                const std::string& initialInput);
        WorkflowResult advance(const std::string& input);

        bool isActive() const { return m_active; }
        void reset();

        logic_meta_patterns::LogicSystem getActiveSystem() const { return m_activeSystem; }
        int getCurrentStepIndex() const { return m_sequence.currentIndex; }
        std::size_t getStepCount() const { return m_sequence.steps.size(); }
        const std::map<std::string, std::string>& getVariables() const { return m_sequence.variables; }
        int getCompletionCount() const { return m_completionCount; }
        const std::map<std::string, int>& getActivationStats() const { return m_activationStats; }

        std::vector<logic_meta_patterns::MetaPattern> collectConsolidatedMetaPatterns(double threshold) const;

    private:
        bool m_active;
        logic_meta_patterns::LogicSystem m_activeSystem;
        WorkflowSequence m_sequence;
        std::string m_lastResponse;

        int m_completionCount;
        std::map<std::string, int> m_activationStats;
        std::map<std::string, double> m_patternConfidence;

        static bool wildcardMatch(const std::string& pattern,
                                  const std::string& input,
                                  std::vector<std::string>& captures);
        static std::string toUpper(const std::string& s);
        static std::string substituteStars(const std::string& templ,
                                      const std::vector<std::string>& captures,
                                      std::map<std::string, std::string>& varStore);
    };

} // namespace workflow_engine

#endif // WORKFLOW_ENGINE_H_
