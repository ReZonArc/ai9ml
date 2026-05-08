#ifndef __MATH_PRIMITIVES_H__
#define __MATH_PRIMITIVES_H__

#include "aimlcategory.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>

using namespace std;
using namespace aiml;

namespace math_primitives {

    struct MathPrimitive {
        string name;
        int arity;
        string pattern;
        string templateText;
        function<string(const vector<string>&)> reducer;

        MathPrimitive() : arity(0) {}
        MathPrimitive(const string& n, int a,
                      const string& p,
                      const string& t,
                      function<string(const vector<string>&)> r)
            : name(n), arity(a), pattern(p), templateText(t), reducer(r) {}
    };

    class MathPrimitiveRegistry {
    public:
        static MathPrimitiveRegistry& getInstance();

        const vector<MathPrimitive>& primitives() const { return m_primitives; }

        vector<unique_ptr<Category>> generateCategories() const;
        bool writeAIMLFile(const string& outputPath) const;

    private:
        MathPrimitiveRegistry();
        vector<MathPrimitive> m_primitives;
        void init();
    };

} // namespace math_primitives

#endif // __MATH_PRIMITIVES_H__
