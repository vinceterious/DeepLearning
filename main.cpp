#include <iostream>
#include <string>
#include <fstream>
#include <limits>
#include "matrix/Matrix.hpp"

namespace 
{
std::vector<std::string> split(std::string line, std::string delim)
{
    std::vector<std::string> result;
    size_t pos = 0;
    while ((pos = line.find(delim)) != std::string::npos)
    {
        result.push_back(line.substr(0, pos));
        line.erase(0, pos + delim.length());
    }
    return result;
}

std::size_t readPredicitionIndexFromValue(double value)
{
    if( value < 4 ) return 0;
    if( value < 5 ) return 1;
    if( value < 6 ) return 2;
    if( value < 7 ) return 3;
    if( value < 8 ) return 4;
    if( value < 9 ) return 5;
    if( value < 10 ) return 6;
    return std::numeric_limits<std::size_t>::max();
}

}

int main()
{
    constexpr std::size_t M1{4899};
    constexpr std::size_t N1{11};
    constexpr std::size_t N2{7};
    using MatFeature = Matrix<double,M1,N1>;
    MatFeature* mRawData = new MatFeature{1};

    using MatPrediction = Matrix<double,M1,N2>;
    MatPrediction* y = new MatPrediction{0};

    std::ifstream infile("winequality-white.csv");
    std::string line;
    if (infile.is_open()) 
    {
        std::size_t i{0};
        while (std::getline(infile, line))
        {
            line += ';';
            std::vector<std::string> r = split(line,";");
            if(i==0) {i++; continue;}
            for( std::size_t j{0} ; j < r.size() - 1; j++ )
            {
                mRawData->access(i,j) = std::stod(r.at(j));
            }
            auto readValue = r.back();
            auto index = readPredicitionIndexFromValue(std::stod(readValue));
            y->access(i,index) = 1;
            i++;
        }
    }
    auto normaData = mRawData->minMaxNormalisationByColumn();
    auto [ X_train, X_test ] = normaData.split<3919>();
    auto [ Y_train, Y_test ] = y->split<3919>();

    auto X_trainT = X_train.T();
    auto X_testT = X_test.T();
    auto Y_trainT = Y_train.T();
    auto Y_testT = Y_test.T();
    std::cout << X_trainT << std::endl;
    std::cout << X_testT << std::endl;
    std::cout << Y_trainT << std::endl;
    std::cout << Y_testT << std::endl;
    
}