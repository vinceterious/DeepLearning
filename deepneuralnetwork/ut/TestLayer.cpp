#include <gtest/gtest.h>
#include <iostream>

#include "../Layer.hpp"
#include "sys/types.h"
#include "sys/sysinfo.h"

namespace testing{

using deepneuralnetwork::Layer;

TEST(TestLayer, TestDimension) 
{
    constexpr std::size_t M1{4};
    constexpr std::size_t N1{100};
    constexpr std::size_t NumOfNeurone{32};
    
    using Dataset = Matrix<double,M1,N1>;
    using L1 = Layer<Dataset,NumOfNeurone>;
    ASSERT_TRUE( L1::m() == NumOfNeurone );
    ASSERT_TRUE( L1::n() == N1 );
}

TEST(TestLayer, TestLayerImbrication) 
{
    constexpr std::size_t M1{100};
    constexpr std::size_t N1{4};
    constexpr std::size_t NumOfNeurone{16};
    constexpr std::size_t NumOfNeurone2{32};
    
    using Dataset = Matrix<double,M1,N1>;
    using L1 = Layer<Dataset,NumOfNeurone>;

    using L2 =  Layer<L1,NumOfNeurone2>;

    ASSERT_TRUE( L2::m() == NumOfNeurone2 );
    ASSERT_TRUE( L2::n() == N1 );
}

TEST(TestLayer, TestForwardPropag) 
{
    constexpr std::size_t M1{2};
    constexpr std::size_t N1{4};
    constexpr std::size_t NumOfNeurone{16};
    constexpr std::size_t NumOfNeurone2{32};
    
    using Dataset = Matrix<double,M1,N1>;
    using L1 = Layer<Dataset,NumOfNeurone>;
    using L2 =  Layer<L1,NumOfNeurone2>;

    Dataset data{{
        Dataset::Row{1, 2, 3, 4},
        Dataset::Row{5, 6, 7, 8}
    }};
    L1 l1{};
    l1.forward_propag(data);
    std::cout << l1.getActivation() << std::endl;

}

}