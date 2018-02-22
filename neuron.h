#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {
    public:
        Neuron(unsigned numOutputs, unsigned myIndex);
        void setOutputVal(double val) { m_outputVal = val; }
        double getOutputVal(void) const { return m_outputVal; }
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);

    private:
        static double eta;      // [0.0..1.0] overal net training rate
        static double alpha;    // [0.0..n] multiplier of last weight change (momentum)
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        static double randomWeight(void) { return rand() / double(RAND_MAX); }
        double sumDOW(const Layer &nextLayer) const;
        double m_outputVal;
        std::vector<Connection> m_outputWeights;
        unsigned m_myIndex;
        double m_gradient;
};

double Neuron::eta = 0.2; // overall net learning rate [0.0..1.0]
double Neuron::alpha = 0.5;// momentum, multiplier of last deltaWeight

void Neuron::updateInputWeights(Layer &prevLayer) {
    // The weights to be updated are in the connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            // Individual input, magnified by the gradient and training rate
            eta
            * neuron.getOutputVal()
            * m_gradient
            // Also add momentum = a fraction of the previous delta weight
            + alpha
            * oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;

    // sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) {
    // tanh - output range [-1.0...1.0]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    // tanh derivative
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

#endif
