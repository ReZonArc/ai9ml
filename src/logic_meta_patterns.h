#ifndef LOGIC_META_PATTERNS_H_
#define LOGIC_META_PATTERNS_H_

#include <string>
#include <vector>
#include <map>

namespace logic_meta_patterns {

    enum LogicSystem {
        LOGIC_NONE = 0,
        PL,
        FOL,
        MODAL,
        LINEAR,
        DTT,
        CAT_THEORY
    };

    struct MetaPattern {
        LogicSystem system;
        std::string topic_guard;
        std::string that_guard;
        std::string pattern;
        std::string template_action;

        MetaPattern()
            : system(LOGIC_NONE) {}

        MetaPattern(LogicSystem s,
                    const std::string& topic,
                    const std::string& that,
                    const std::string& pat,
                    const std::string& action)
            : system(s), topic_guard(topic), that_guard(that),
              pattern(pat), template_action(action) {}
    };

    class MetaPatternRegistry {
    public:
        static MetaPatternRegistry& getInstance();

        const std::vector<MetaPattern>& getPatterns() const { return m_patterns; }
        std::vector<MetaPattern> getPatternsForSystem(LogicSystem system) const;

        // Generate AIML files grouped per logic system.
        bool writeAIMLFiles(const std::string& outputDir) const;

    private:
        MetaPatternRegistry();
        std::vector<MetaPattern> m_patterns;

        void addSystemPatterns();
        static std::string systemToFilename(LogicSystem system);
    };

    std::string toString(LogicSystem system);
    LogicSystem logicSystemFromString(const std::string& text);

} // namespace logic_meta_patterns

#endif // LOGIC_META_PATTERNS_H_
