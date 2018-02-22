#ifndef IRIS_DATA_H
#define IRIS_DATA_H
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>


void getData(std::ifstream &df, std::vector<double> &inputVals,std::vector<double> &targetVals) {
    inputVals.clear();
    std::fill(targetVals.begin(), targetVals.end(), 0);
    std::string line;
    getline(df, line);
    std::stringstream ss(line);

    double input;
    for (int i = 0; i < 4; ++i) {
        ss >> input;
        inputVals.push_back(tanh(input/2));
        if (ss.peek() == ',')
            ss.ignore();
    }

    std::string target;
    ss >> target;
    if (target == "Iris-setosa") {
        targetVals[0] = 1;
    }
    else if (target == "Iris-versicolor") {
        targetVals[1] = 1;
    }
    else if (target == "Iris-virginica") {
        targetVals[2] = 1;
    }
}

#endif
