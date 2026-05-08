#ifndef LOGIC_CLASSIFIER_H_
#define LOGIC_CLASSIFIER_H_

#include "logic_meta_patterns.h"
#include "hgnn.h"
#include "dtesnn.h"
#include <string>
#include <vector>
#include <map>

namespace logic_classifier {

    struct Classification {
        logic_meta_patterns::LogicSystem system;
        double confidence;
        std::vector<double> probabilities;
        bool isLogic;

        Classification()
            : system(logic_meta_patterns::LOGIC_NONE), confidence(0.0), isLogic(false) {}
    };

    class LogicClassifier {
    public:
        LogicClassifier();
        ~LogicClassifier() = default;

        Classification classify(const std::string& input,
                                const std::map<std::string, double>& contextVector,
                                hgnn::HyperGraphNeuralNet* hgnnNet,
                                dtesnn::DeepTreeEchoStateNet* dtesnnNet);

        bool parseOverride(const std::string& input,
                           logic_meta_patterns::LogicSystem& system,
                           std::string& remainingInput) const;

        void reinforce(const std::string& input,
                       const std::map<std::string, double>& contextVector,
                       hgnn::HyperGraphNeuralNet* hgnnNet,
                       dtesnn::DeepTreeEchoStateNet* dtesnnNet,
                       logic_meta_patterns::LogicSystem target,
                       double learningRate = 0.01);

        int getUpdateCount() const { return m_updateCount; }
        Classification getLastClassification() const { return m_lastClassification; }

    private:
        static const int INPUT_DIM = 28;
        static const int HIDDEN_DIM = 16;
        static const int OUTPUT_DIM = 7; // NONE + 6 systems

        std::vector<std::vector<double>> m_W1;
        std::vector<double> m_b1;
        std::vector<std::vector<double>> m_W2;
        std::vector<double> m_b2;

        std::vector<double> m_lastInput;
        std::vector<double> m_lastH;
        std::vector<double> m_lastOut;

        int m_updateCount;
        Classification m_lastClassification;

        std::vector<double> buildFeatures(const std::string& input,
                                     const std::map<std::string, double>& contextVector,
                                     hgnn::HyperGraphNeuralNet* hgnnNet,
                                     dtesnn::DeepTreeEchoStateNet* dtesnnNet) const;

        std::vector<double> forward(const std::vector<double>& input);
        void backward(const std::vector<double>& input,
                      int targetIndex,
                      double learningRate);

        static std::vector<std::string> tokenize(const std::string& text);
        static std::vector<double> softmax(const std::vector<double>& x);
        static std::vector<double> tanhVec(const std::vector<double>& x);
        static int systemToIndex(logic_meta_patterns::LogicSystem s);
        static logic_meta_patterns::LogicSystem indexToSystem(int idx);
    };

} // namespace logic_classifier

#endif // LOGIC_CLASSIFIER_H_
