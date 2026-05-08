#ifndef __LOGIC_META_PATTERNS_H__
#define __LOGIC_META_PATTERNS_H__

#include <string>
#include <vector>
#include <map>

using namespace std;

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
        string topic_guard;
        string that_guard;
        string pattern;
        string template_action;

        MetaPattern()
            : system(LOGIC_NONE) {}

        MetaPattern(LogicSystem s,
                    const string& topic,
                    const string& that,
                    const string& pat,
                    const string& action)
            : system(s), topic_guard(topic), that_guard(that),
              pattern(pat), template_action(action) {}
    };

    class MetaPatternRegistry {
    public:
        static MetaPatternRegistry& getInstance();

        const vector<MetaPattern>& getPatterns() const { return m_patterns; }
        vector<MetaPattern> getPatternsForSystem(LogicSystem system) const;

        // Generate AIML files grouped per logic system.
        bool writeAIMLFiles(const string& outputDir) const;

    private:
        MetaPatternRegistry();
        vector<MetaPattern> m_patterns;

        void addSystemPatterns();
        static string systemToFilename(LogicSystem system);
    };

    string toString(LogicSystem system);
    LogicSystem logicSystemFromString(const string& text);

} // namespace logic_meta_patterns

#endif // __LOGIC_META_PATTERNS_H__
