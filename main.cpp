// neural net
#include "net.h"
#include "iris_data.h"
#include <vector>
#include <iostream>
#include <istream>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>


void showVectorVals(std::string label, std::vector<double> &v) {
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

int main(void) {
    int maxPass;
    int nodes;
    std::string line;
    std::vector<unsigned> topology;

    std::cout << "Enter topology of neural net.(for a net with two input hidden"
              << "layer of 3 nodes and one output you would type:2 3 1)\n";
    std::getline(std::cin, line);
    std::stringstream ss(line);
    while(ss >> nodes) {
        topology.push_back(nodes);
    }

    std::cout << "Enter number of training passes for neural net: ";
    std::cin >> maxPass;

    bool flag = true;
    std::ifstream df;
    df.open("data.txt");
    std::vector<double> inputVals, targetVals(3), resultVals;


    Net myNet(topology);

    int trainingPass = 0;

    while (!df.eof()) {
        ++trainingPass;
        getData(df, inputVals, targetVals);
        std::cout << std::endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        std::cout << "Net recent average error: "
                << myNet.getRecentAverageError() << std::endl;

        // Continue to reset training file if training is not finished
        if (df.eof() && trainingPass < maxPass) {
            df.clear();
            df.seekg(0, std::ios::beg);
        }

        // Test the neural net once training is done
        if (df.eof() && flag) {
            df.close();
            df.clear();
            df.open("test.txt");
            std::cout << std::endl << std::endl << std::endl;
            flag = false;
            myNet.resetError();
        }

    }

    std::cout << std::endl << "Done" << std::endl;
}
