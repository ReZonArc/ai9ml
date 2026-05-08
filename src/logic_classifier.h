#ifndef __LOGIC_CLASSIFIER_H__
#define __LOGIC_CLASSIFIER_H__

#include "logic_meta_patterns.h"
#include "hgnn.h"
#include "dtesnn.h"
#include <string>
#include <vector>
#include <map>

using namespace std;

namespace logic_classifier {

    struct Classification {
        logic_meta_patterns::LogicSystem system;
        double confidence;
        vector<double> probabilities;
        bool isLogic;

        Classification()
            : system(logic_meta_patterns::LOGIC_NONE), confidence(0.0), isLogic(false) {}
    };

    class LogicClassifier {
    public:
        LogicClassifier();
        ~LogicClassifier() = default;

        Classification classify(const string& input,
                                const map<string, double>& contextVector,
                                hgnn::HyperGraphNeuralNet* hgnnNet,
                                dtesnn::DeepTreeEchoStateNet* dtesnnNet);

        bool parseOverride(const string& input,
                           logic_meta_patterns::LogicSystem& system,
                           string& remainingInput) const;

        void reinforce(const string& input,
                       const map<string, double>& contextVector,
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

        vector<vector<double>> m_W1;
        vector<double> m_b1;
        vector<vector<double>> m_W2;
        vector<double> m_b2;

        vector<double> m_lastInput;
        vector<double> m_lastH;
        vector<double> m_lastOut;

        int m_updateCount;
        Classification m_lastClassification;

        vector<double> buildFeatures(const string& input,
                                     const map<string, double>& contextVector,
                                     hgnn::HyperGraphNeuralNet* hgnnNet,
                                     dtesnn::DeepTreeEchoStateNet* dtesnnNet) const;

        vector<double> forward(const vector<double>& input);
        void backward(const vector<double>& input,
                      int targetIndex,
                      double learningRate);

        static vector<string> tokenize(const string& text);
        static vector<double> softmax(const vector<double>& x);
        static vector<double> tanhVec(const vector<double>& x);
        static int systemToIndex(logic_meta_patterns::LogicSystem s);
        static logic_meta_patterns::LogicSystem indexToSystem(int idx);
    };

} // namespace logic_classifier

#endif // __LOGIC_CLASSIFIER_H__
