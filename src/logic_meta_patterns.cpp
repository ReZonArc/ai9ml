#include "logic_meta_patterns.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>

using namespace std;
using namespace logic_meta_patterns;

static string upperCopy(const string& s) {
    string out = s;
    transform(out.begin(), out.end(), out.begin(),
              [](unsigned char c) { return (char)::toupper(c); });
    return out;
}

string logic_meta_patterns::toString(LogicSystem system) {
    switch (system) {
        case PL:         return "PL";
        case FOL:        return "FOL";
        case MODAL:      return "MODAL";
        case LINEAR:     return "LINEAR";
        case DTT:        return "DTT";
        case CAT_THEORY: return "CAT_THEORY";
        default:         return "NONE";
    }
}

LogicSystem logic_meta_patterns::logicSystemFromString(const string& text) {
    string t = upperCopy(text);
    if (t == "PL" || t == "PROPOSITIONAL") return PL;
    if (t == "FOL" || t == "FIRST_ORDER") return FOL;
    if (t == "MODAL") return MODAL;
    if (t == "LINEAR") return LINEAR;
    if (t == "DTT" || t == "DEPENDENT_TYPES") return DTT;
    if (t == "CAT_THEORY" || t == "CATEGORY" || t == "CATEGORY_THEORY") return CAT_THEORY;
    return LOGIC_NONE;
}

MetaPatternRegistry& MetaPatternRegistry::getInstance() {
    static MetaPatternRegistry instance;
    return instance;
}

MetaPatternRegistry::MetaPatternRegistry() {
    addSystemPatterns();
}

void MetaPatternRegistry::addSystemPatterns() {
    // PL
    m_patterns.emplace_back(PL, "PL", "*", "CNF *", "Converted to CNF: <star/>");
    m_patterns.emplace_back(PL, "PL", "*", "DNF *", "Converted to DNF: <star/>");
    m_patterns.emplace_back(PL, "PL", "*", "RESOLVE * AND *", "Applied propositional resolution on <star/>.");

    // FOL
    m_patterns.emplace_back(FOL, "FOL", "*", "UNIFY * WITH *", "Unifier candidate generated for <star/>.");
    m_patterns.emplace_back(FOL, "FOL", "*", "HERBRAND *", "Herbrand expansion initiated for <star/>.");
    m_patterns.emplace_back(FOL, "FOL", "*", "ROBINSON *", "Robinson resolution step executed: <star/>.");

    // MODAL
    m_patterns.emplace_back(MODAL, "MODAL", "*", "NECESSARILY *", "□ <star/>");
    m_patterns.emplace_back(MODAL, "MODAL", "*", "POSSIBLY *", "◇ <star/>");
    m_patterns.emplace_back(MODAL, "MODAL", "*", "ACCESS WORLD * TO *", "Kripke accessibility updated for <star/>.");

    // LINEAR
    m_patterns.emplace_back(LINEAR, "LINEAR", "*", "TENSOR * WITH *", "Linear tensor introduction: <star/>.");
    m_patterns.emplace_back(LINEAR, "LINEAR", "*", "PROMOTE *", "Bang promotion applied to <star/>.");
    m_patterns.emplace_back(LINEAR, "LINEAR", "*", "CONSUME *", "Resource consumed: <star/>.");

    // DTT
    m_patterns.emplace_back(DTT, "DTT", "*", "TYPECHECK *", "Typechecked in context: <star/>.");
    m_patterns.emplace_back(DTT, "DTT", "*", "PI_FORM * TO *", "Π-type formed from <star/>.");
    m_patterns.emplace_back(DTT, "DTT", "*", "SIGMA_ELIM *", "Σ-elimination on <star/>.");

    // Category theory
    m_patterns.emplace_back(CAT_THEORY, "CAT_THEORY", "*", "FUNCTOR * TO *", "Functor action computed for <star/>.");
    m_patterns.emplace_back(CAT_THEORY, "CAT_THEORY", "*", "NATURAL_TRANSFORMATION *", "Natural transformation component: <star/>.");
    m_patterns.emplace_back(CAT_THEORY, "CAT_THEORY", "*", "ADJUNCTION *", "Adjunction unit/counit checked for <star/>.");
}

vector<MetaPattern> MetaPatternRegistry::getPatternsForSystem(LogicSystem system) const {
    vector<MetaPattern> out;
    for (const auto& p : m_patterns) {
        if (p.system == system) out.push_back(p);
    }
    return out;
}

string MetaPatternRegistry::systemToFilename(LogicSystem system) {
    switch (system) {
        case PL:         return "pl_logic.aiml";
        case FOL:        return "fol_logic.aiml";
        case MODAL:      return "modal_logic.aiml";
        case LINEAR:     return "linear_logic.aiml";
        case DTT:        return "dtt_logic.aiml";
        case CAT_THEORY: return "cat_theory.aiml";
        default:         return "none_logic.aiml";
    }
}

bool MetaPatternRegistry::writeAIMLFiles(const string& outputDir) const {
    if (mkdir(outputDir.c_str(), 0755) != 0 && errno != EEXIST) {
        cerr << "[MetaPatternRegistry] Failed to create directory "
             << outputDir << ": " << strerror(errno) << endl;
        return false;
    }

    vector<LogicSystem> systems = {PL, FOL, MODAL, LINEAR, DTT, CAT_THEORY};
    for (auto system : systems) {
        string path = outputDir + "/" + systemToFilename(system);
        ofstream out(path.c_str());
        if (!out.is_open()) return false;

        out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        out << "<aiml version=\"1.0\">\n";
        out << "  <topic name=\"" << toString(system) << "\">\n";

        auto patterns = getPatternsForSystem(system);
        for (const auto& p : patterns) {
            out << "    <category>\n";
            out << "      <pattern>" << p.pattern << "</pattern>\n";
            if (!p.that_guard.empty() && p.that_guard != "*")
                out << "      <that>" << p.that_guard << "</that>\n";
            out << "      <template>" << p.template_action << "</template>\n";
            out << "    </category>\n";
        }

        out << "  </topic>\n";
        out << "</aiml>\n";
    }

    return true;
}
