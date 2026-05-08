#ifndef MATH_PRIMITIVES_H_
#define MATH_PRIMITIVES_H_

#include "aimlcategory.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace math_primitives {

    struct MathPrimitive {
        std::string name;
        int arity;
        std::string pattern;
        std::string templateText;
        std::function<std::string(const std::vector<std::string>&)> reducer;

        MathPrimitive() : arity(0) {}
        MathPrimitive(const std::string& n, int a,
                      const std::string& p,
                      const std::string& t,
                      std::function<std::string(const std::vector<std::string>&)> r)
            : name(n), arity(a), pattern(p), templateText(t), reducer(r) {}
    };

    class MathPrimitiveRegistry {
    public:
        static MathPrimitiveRegistry& getInstance();

        const std::vector<MathPrimitive>& primitives() const { return m_primitives; }

        std::vector<std::unique_ptr<aiml::Category>> generateCategories() const;
        bool writeAIMLFile(const std::string& outputPath) const;

    private:
        MathPrimitiveRegistry();
        std::vector<MathPrimitive> m_primitives;
        void init();
    };

} // namespace math_primitives

#endif // MATH_PRIMITIVES_H_
