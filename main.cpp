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
    bool flag = true;
    std::ifstream df;
    df.open("data.txt");
    std::vector<double> inputVals, targetVals(3), resultVals;

    std::vector<unsigned> topology{4,8,3};

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

        if (df.eof() && trainingPass < 2000) {
            df.clear();
            df.seekg(0, std::ios::beg);
        }

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
