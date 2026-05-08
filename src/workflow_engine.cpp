#include "workflow_engine.h"
#include <algorithm>
#include <sstream>
#include <cctype>

using namespace std;
using namespace workflow_engine;
using namespace logic_meta_patterns;

WorkflowEngine::WorkflowEngine()
    : m_active(false),
      m_activeSystem(LOGIC_NONE),
      m_completionCount(0)
{}

string WorkflowEngine::toUpper(const string& s) {
    string out = s;
    transform(out.begin(), out.end(), out.begin(),
              [](unsigned char c) { return (char)::toupper(c); });
    return out;
}

bool WorkflowEngine::wildcardMatch(const string& pattern,
                                   const string& input,
                                   vector<string>& captures)
{
    captures.clear();
    string p = toUpper(pattern);
    string i = toUpper(input);

    if (p == i) return true;
    if (p.find('*') == string::npos) return false;

    vector<string> parts;
    {
        string cur;
        for (char c : p) {
            if (c == '*') {
                parts.push_back(cur);
                cur.clear();
            } else {
                cur += c;
            }
        }
        parts.push_back(cur);
    }

    size_t pos = 0;
    for (size_t idx = 0; idx + 1 < parts.size(); ++idx) {
        const string& left = parts[idx];
        const string& right = parts[idx + 1];

        if (!left.empty()) {
            if (i.compare(pos, left.size(), left) != 0) return false;
            pos += left.size();
        }

        if (right.empty()) {
            captures.push_back(i.substr(pos));
            pos = i.size();
            continue;
        }

        size_t nextPos = i.find(right, pos);
        if (nextPos == string::npos) return false;
        captures.push_back(i.substr(pos, nextPos - pos));
        pos = nextPos;
    }

    const string& tail = parts.back();
    if (!tail.empty()) {
        if (i.size() < tail.size()) return false;
        if (i.substr(i.size() - tail.size()) != tail) return false;
    }
    return true;
}

string WorkflowEngine::substituteStars(const string& templ,
                                       const vector<string>& captures,
                                       map<string, string>& varStore)
{
    string out = templ;

    // Store captures for later workflow steps.
    for (size_t i = 0; i < captures.size(); ++i)
        varStore["$" + to_string(i + 1)] = captures[i];

    // Replace <star/> with first capture (AIML-style shorthand).
    size_t pos = out.find("<star/>");
    if (pos != string::npos && !captures.empty())
        out.replace(pos, 7, captures[0]);

    // Replace variable placeholders.
    for (const auto& kv : varStore) {
        size_t p = 0;
        while ((p = out.find(kv.first, p)) != string::npos) {
            out.replace(p, kv.first.size(), kv.second);
            p += kv.second.size();
        }
    }

    return out;
}

WorkflowResult WorkflowEngine::activate(LogicSystem system,
                                        const string& initialInput)
{
    WorkflowResult result;
    reset();

    auto patterns = MetaPatternRegistry::getInstance().getPatternsForSystem(system);
    if (patterns.empty()) {
        result.response = "No workflow available for requested logic system.";
        return result;
    }

    m_active = true;
    m_activeSystem = system;
    for (const auto& p : patterns)
        m_sequence.steps.push_back(WorkflowStep(p));

    m_activationStats[toString(system)]++;
    return advance(initialInput);
}

WorkflowResult WorkflowEngine::advance(const string& input)
{
    WorkflowResult result;
    if (!m_active || m_sequence.steps.empty())
        return result;

    if (m_sequence.currentIndex < 0 ||
        m_sequence.currentIndex >= (int)m_sequence.steps.size()) {
        result.completed = true;
        return result;
    }

    WorkflowStep& step = m_sequence.steps[m_sequence.currentIndex];
    vector<string> captures;

    if (!wildcardMatch(step.metaPattern.pattern, input, captures)) {
        result.response = "Workflow active (" + toString(m_activeSystem) + "), awaiting pattern: " +
                          step.metaPattern.pattern;
        result.score = 0.1;
        result.confidence = 0.2;
        return result;
    }

    step.completed = true;
    step.boundVars.clear();
    for (size_t i = 0; i < captures.size(); ++i)
        step.boundVars["$" + to_string(i + 1)] = captures[i];

    result.response = substituteStars(step.metaPattern.template_action,
                                      captures,
                                      m_sequence.variables);
    result.matched = true;
    result.score = 0.8;
    result.confidence = 0.85;

    string key = toString(m_activeSystem) + ":" + step.metaPattern.pattern;
    m_patternConfidence[key] = min(1.0, m_patternConfidence[key] + 0.05);

    m_sequence.currentIndex++;
    if (m_sequence.currentIndex >= (int)m_sequence.steps.size()) {
        result.completed = true;
        m_completionCount++;
        m_lastResponse = result.response;
    }

    return result;
}

void WorkflowEngine::reset() {
    m_active = false;
    m_activeSystem = LOGIC_NONE;
    m_sequence = WorkflowSequence();
    m_lastResponse.clear();
}

vector<MetaPattern> WorkflowEngine::collectConsolidatedMetaPatterns(double threshold) const {
    vector<MetaPattern> out;
    const auto& all = MetaPatternRegistry::getInstance().getPatterns();

    for (const auto& p : all) {
        string key = toString(p.system) + ":" + p.pattern;
        auto it = m_patternConfidence.find(key);
        if (it != m_patternConfidence.end() && it->second >= threshold)
            out.push_back(p);
    }
    return out;
}
