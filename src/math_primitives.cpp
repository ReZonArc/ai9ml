#include "math_primitives.h"
#include "aimlpattern.h"
#include "aimltemplate.h"
#include "aimltext.h"
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace math_primitives;

static string joinTokens(const vector<string>& t, const string& sep = " ") {
    ostringstream os;
    for (size_t i = 0; i < t.size(); ++i) {
        if (i) os << sep;
        os << t[i];
    }
    return os.str();
}

MathPrimitiveRegistry& MathPrimitiveRegistry::getInstance() {
    static MathPrimitiveRegistry instance;
    return instance;
}

MathPrimitiveRegistry::MathPrimitiveRegistry() {
    init();
}

void MathPrimitiveRegistry::init() {
    // Peano / naturals
    m_primitives.emplace_back("succ", 1, "SUCC *", "Peano successor of <star/>.",
        [](const vector<string>& a) { return "SUCC(" + joinTokens(a) + ")"; });
    m_primitives.emplace_back("pred", 1, "PRED *", "Peano predecessor of <star/>.",
        [](const vector<string>& a) { return "PRED(" + joinTokens(a) + ")"; });
    m_primitives.emplace_back("add", 2, "ADD * AND *", "Additive reduction for <star/>.",
        [](const vector<string>& a) { return "ADD(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("mul", 2, "MUL * BY *", "Multiplicative reduction for <star/>.",
        [](const vector<string>& a) { return "MUL(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("iszero", 1, "IS * ZERO", "Zero-test result for <star/>.",
        [](const vector<string>& a) { return "IS_ZERO(" + joinTokens(a) + ")"; });

    // Lambda calculus
    m_primitives.emplace_back("beta", 2,
        "BETA_REDUCE LAMBDA _ DOT * APPLIED_TO *",
        "β-reduction step on <star/>.",
        [](const vector<string>& a) { return "BETA(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("alpha", 2,
        "ALPHA_RENAME * TO *",
        "α-renaming performed for <star/>.",
        [](const vector<string>& a) { return "ALPHA(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("eta", 1,
        "ETA_REDUCE *",
        "η-reduction performed for <star/>.",
        [](const vector<string>& a) { return "ETA(" + joinTokens(a) + ")"; });

    // Set theory
    m_primitives.emplace_back("member", 2, "IS * MEMBER_OF *", "Membership check for <star/>.",
        [](const vector<string>& a) { return "MEMBER(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("union", 2, "UNION * WITH *", "Union computed for <star/>.",
        [](const vector<string>& a) { return "UNION(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("intersect", 2, "INTERSECT * AND *", "Intersection computed for <star/>.",
        [](const vector<string>& a) { return "INTERSECT(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("powerset", 1, "POWER_SET_OF *", "Power set generation for <star/>.",
        [](const vector<string>& a) { return "POWERSET(" + joinTokens(a) + ")"; });
    m_primitives.emplace_back("subset", 2, "IS * SUBSET_OF *", "Subset relation check for <star/>.",
        [](const vector<string>& a) { return "SUBSET(" + joinTokens(a, ",") + ")"; });

    // Propositional
    m_primitives.emplace_back("and", 2, "* AND *", "Conjunction simplification for <star/>.",
        [](const vector<string>& a) { return "AND(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("or", 2, "* OR *", "Disjunction simplification for <star/>.",
        [](const vector<string>& a) { return "OR(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("not", 1, "NOT *", "Negation simplification for <star/>.",
        [](const vector<string>& a) { return "NOT(" + joinTokens(a) + ")"; });
    m_primitives.emplace_back("implies", 2, "* IMPLIES *", "Implication rewrite for <star/>.",
        [](const vector<string>& a) { return "IMPLIES(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("iff", 2, "* IFF *", "Biconditional rewrite for <star/>.",
        [](const vector<string>& a) { return "IFF(" + joinTokens(a, ",") + ")"; });

    // Quantifiers
    m_primitives.emplace_back("forall", 2, "FORALL * SUCH_THAT *", "Universal quantifier reduction for <star/>.",
        [](const vector<string>& a) { return "FORALL(" + joinTokens(a, ",") + ")"; });
    m_primitives.emplace_back("exists", 2, "EXISTS * SUCH_THAT *", "Existential quantifier reduction for <star/>.",
        [](const vector<string>& a) { return "EXISTS(" + joinTokens(a, ",") + ")"; });
}

vector<unique_ptr<Category>> MathPrimitiveRegistry::generateCategories() const {
    vector<unique_ptr<Category>> out;
    out.reserve(m_primitives.size());

    for (const auto& p : m_primitives) {
        vector<TemplateElement*> children;
        string templ = "<think><set name=\"last_math_primitive\">" + p.name +
                       "</set></think>" + p.templateText;
        children.push_back(new Text(templ));

        unique_ptr<Pattern> pat(new Pattern(p.pattern));
        unique_ptr<Template> tmp(new Template(children));
        unique_ptr<Category> cat(new Category(pat.release(), tmp.release()));
        out.push_back(std::move(cat));
    }

    return out;
}

bool MathPrimitiveRegistry::writeAIMLFile(const string& outputPath) const {
    ofstream out(outputPath.c_str());
    if (!out.is_open()) return false;

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    out << "<aiml version=\"1.0\">\n";
    for (const auto& p : m_primitives) {
        out << "  <category>\n";
        out << "    <pattern>" << p.pattern << "</pattern>\n";
        out << "    <template>" << p.templateText << "</template>\n";
        out << "  </category>\n";
    }
    out << "</aiml>\n";
    return true;
}
