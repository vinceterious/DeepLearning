
#include <gtest/gtest.h>

#include "../Matrix.hpp"
#include "sys/types.h"
#include "sys/sysinfo.h"

namespace testing{

TEST(TestMatrix, TestConstructor) 
{
    constexpr std::size_t M1{3};
    constexpr std::size_t N1{4};
    Matrix<int,M1,N1> mat1;
    for(auto i{0}; i< M1 ; i++)
    {
        for(auto y{0}; y < N1 ; y++)
        {
            ASSERT_EQ(0, mat1.at(i,y));
            ASSERT_EQ(0, mat1.access(i,y));
        }
    }
    std::cout << "Mat1:\n" << mat1 << "\n";
}

TEST(TestMatrix, TestAccess) 
{
    constexpr std::size_t M1{3};
    constexpr std::size_t N1{2};
    using Mat = Matrix<int,M1,N1>;
    Mat mat2({ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} });
    ASSERT_EQ(1, mat2.at(0,0));
    ASSERT_EQ(2, mat2.at(0,1));
    ASSERT_EQ(3, mat2.at(1,0));
    ASSERT_EQ(4, mat2.at(1,1));
    ASSERT_EQ(5, mat2.at(2,0));
    ASSERT_EQ(6, mat2.at(2,1));
    std::cout << "Mat2:\n" << mat2 << "\n";


    Mat mat3({ Mat::Column{1,3,5},  Mat::Column{2,4,6}});
    ASSERT_EQ(1, mat3.at(0,0));
    ASSERT_EQ(2, mat3.at(0,1));
    ASSERT_EQ(3, mat3.at(1,0));
    ASSERT_EQ(4, mat3.at(1,1));
    ASSERT_EQ(5, mat3.at(2,0));
    ASSERT_EQ(6, mat3.at(2,1));
    std::cout << "Mat3:\n" << mat3 << "\n";

}

TEST(TestMatrix, TestAtAndPartialAlloc) 
{
    constexpr std::size_t M1{5};
    constexpr std::size_t N1{2};
    using Mat = Matrix<int,M1,N1>;
    Mat mat2({ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} });
    ASSERT_EQ(1, mat2.at(0,0));
    ASSERT_EQ(2, mat2.at(0,1));
    ASSERT_EQ(3, mat2.at(1,0));
    ASSERT_EQ(4, mat2.at(1,1));
    ASSERT_EQ(5, mat2.at(2,0));
    ASSERT_EQ(6, mat2.at(2,1));
    ASSERT_EQ(0, mat2.at(3,0));
    ASSERT_EQ(0, mat2.at(3,1));
    ASSERT_EQ(0, mat2.at(4,0));
    ASSERT_EQ(0, mat2.at(4,1));
    std::cout << "Mat2:\n" << mat2 << "\n";
}

TEST(TestMatrix, TestAssertOutOfBond) 
{
    constexpr std::size_t M1{3};
    constexpr std::size_t N1{2};
    using Mat = Matrix<int,M1,N1>;
    Mat mat2({ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} });
    EXPECT_THROW(mat2.at(3,0), std::out_of_range);
    EXPECT_THROW(mat2.at(1,3), std::out_of_range);
}

TEST(TestMatrix, TestIdentityMatrix) 
{
    constexpr std::size_t M1{3};
    constexpr std::size_t N1{2};
    using Mat = Matrix<int,M1,N1>;
    Mat matIdentity{1};
    ASSERT_EQ(1, matIdentity.at(0,0));
    ASSERT_EQ(1, matIdentity.at(0,1));
    ASSERT_EQ(1, matIdentity.at(1,0));
    ASSERT_EQ(1, matIdentity.at(1,1));
    ASSERT_EQ(1, matIdentity.at(2,0));
    ASSERT_EQ(1, matIdentity.at(2,1));
    std::cout << "matIdentity:\n" << matIdentity << "\n";
}

TEST(TestMatrix, TestDimension) 
{

    constexpr std::size_t M1{11154};
    constexpr std::size_t N1{1584};
    using Mat = Matrix<int,M1,N1>;
    Mat* bigMat = new Mat(1);
    ASSERT_EQ(M1, bigMat->m());
    ASSERT_EQ(N1, bigMat->n());
    ASSERT_EQ(N1*M1, bigMat->size());
}

TEST(TestMatrix, TestConcate) 
{
    constexpr std::size_t M1{5};
    constexpr std::size_t N1{4};
    using Mat = Matrix<int,M1,N1>;
    Mat mat1{{ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} }};

    constexpr std::size_t M_Expect{8};
    constexpr std::size_t N_Expect{4};
    using Mat_Expect = Matrix<int,M_Expect,N_Expect>;
    Mat_Expect mat_expect{{ 
        Mat::Row{1,2},  
        Mat::Row{3,4},  
        Mat::Row{5,6},
        Mat::Row{0},
        Mat::Row{0},
        Mat::Row{1,2},  
        Mat::Row{3,4},  
        Mat::Row{5,6} }};

    auto mat2 = mat1.concatRow<3>({Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6}});
    ASSERT_TRUE(mat2 == mat_expect);
}

TEST(TestMatrix, TestInverted) 
{
    constexpr std::size_t M1{5};
    constexpr std::size_t N1{2};
    using Mat = Matrix<int,M1,N1>;
    Mat mat1{{ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6}, Mat::Row{7,8}, Mat::Row{9,10}}};
    using MatTranspose = Matrix<int,N1,M1>;
    MatTranspose mat2{{ MatTranspose::Row{1,3,5,7,9},  MatTranspose::Row{2,4,6,8,10}, }};
    auto mat1T{mat1.T()};
    std::cout << "mat1:\n" << mat1 << "\n";
    std::cout << "mat1T:\n" << mat1.T() << "\n";
    std::cout << "mat2:\n" << mat2 << "\n";
    ASSERT_TRUE(mat1T == mat2);
}

TEST(TestMatrix, TestMultiplication) 
{
    //Check with similar dimension
    constexpr std::size_t M1{2};
    constexpr std::size_t N1{3};
    using Mat = Matrix<int,M1,N1>;
    Mat mat1{{ Mat::Row{1,2,0},  Mat::Row{4,3,-1} }};
    using Mat2 = Matrix<int,N1,M1>;
    Mat2 mat2{{ Mat2::Row{5,1},  Mat2::Row{2,3}, Mat2::Row{3,4} }};
    std::cout << "mat1:\n" << mat1 << "\n";
    std::cout << "mat2:\n" << mat2 << "\n";
    using Mat_ResultExpect = Matrix<int,M1,M1>;
    Mat_ResultExpect mat_expect{{ Mat_ResultExpect::Row{9,7},  Mat_ResultExpect::Row{23,9} }};
    auto mat = mat1 * mat2;
    ASSERT_TRUE(mat_expect == mat);

    //Check with different dimension
    constexpr std::size_t M2{3};
    constexpr std::size_t N2{2};
    using Mat3x2 = Matrix<int,M2,N2>;
    Mat3x2 mat3{{ Mat3x2::Row{1,2},  Mat3x2::Row{3,4}, Mat3x2::Row{5,1} }};
    constexpr std::size_t M3{2};
    constexpr std::size_t N3{1};
    using Mat2x1 = Matrix<int,M3,N3>;
    Mat2x1 mat4{{ Mat2x1::Row{2},  Mat2x1::Row{4}}};
    std::cout << "mat3:\n" << mat3 << "\n";
    std::cout << "mat4:\n" << mat4 << "\n";
    auto matVector = mat3 * mat4;
    using Mat_ResultExpect2 = Matrix<int,M2,N3>;
    Mat_ResultExpect2 mat_expect2{{ 
        Mat_ResultExpect2::Row{10},  
        Mat_ResultExpect2::Row{22}, 
        Mat_ResultExpect2::Row{14}
    }};
    ASSERT_TRUE(mat_expect2 == matVector);

    //Check with identity Matrix
    constexpr std::size_t M4{1};
    constexpr std::size_t N4{5};
    Matrix<int,M4,N4> matIdentity{1};
    using Mat_ResultExpect3 = Matrix<int,M2,N4>;
    Mat_ResultExpect3 mat_expect3{{ 
        Mat_ResultExpect3::Row{10, 10, 10, 10, 10},  
        Mat_ResultExpect3::Row{22, 22, 22, 22, 22}, 
        Mat_ResultExpect3::Row{14, 14, 14, 14, 14}
    }};
    ASSERT_TRUE(mat_expect3 == matVector * matIdentity);
}

TEST(TestMatrix, TestSoustraction) 
{
    constexpr std::size_t M1{5};
    using Mat = Matrix<int,M1,M1>;
    Mat mat1{{ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} }};
    Mat mat2{1};

    using Mat_ResultExpect = Matrix<int,M1,M1>;
    Mat_ResultExpect mat_expect{{ 
        Mat_ResultExpect::Row{0, 1, -1, -1, -1},  
        Mat_ResultExpect::Row{2, 3, -1, -1, -1},
        Mat_ResultExpect::Row{4, 5, -1, -1, -1},  
        Mat_ResultExpect::Row{-1, -1, -1, -1, -1},
        Mat_ResultExpect::Row{-1, -1, -1, -1, -1}
    }};
    auto matSoustract = mat1 - mat2;
    ASSERT_TRUE(mat_expect == matSoustract);

    auto matSoustractWithInteger = mat1 - 1;
    ASSERT_TRUE(mat_expect == matSoustractWithInteger);
}

TEST(TestMatrix, TestSum) 
{
    constexpr std::size_t M1{5};
    constexpr int expectValue{-4};
    using Mat = Matrix<int,M1,M1>;
    Mat mat1{{ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} }};
    Mat mat2{1};
    auto matSoustraction = mat1 - mat2;
    ASSERT_EQ(matSoustraction.sum(),expectValue);
}

TEST(TestMatrix, TestCarreEach) 
{
    constexpr std::size_t M1{5};
    using Mat = Matrix<int,M1,M1>;
    Mat mat1{{ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} }};
    Mat mat2{1};
    std::cout << "mat1:\n" << mat1 << "\n";
    auto matSoustraction = mat1 - mat2;
    std::cout << "matSoustraction:\n" << matSoustraction << "\n";
    Mat carreMat{{ Mat::Row{0,1,1,1,1},  Mat::Row{4,9,1,1,1},  Mat::Row{16,25,1,1,1},  Mat::Row{1,1,1,1,1}, Mat::Row{1,1,1,1,1} }};
    auto po = matSoustraction.squaredEach();
    std::cout << "carre:\n" << po << "\n";
    ASSERT_TRUE(carreMat == po);
}

TEST(TestMatrix, decompositionLU) 
{
    constexpr std::size_t M1{3};
    using Mat = Matrix<double,M1,M1>;
    Mat A{{ Mat::Row{2,-1,0},  Mat::Row{-1,2,-1},  Mat::Row{0,-1,2} }};
    auto [ L , U] = A.decompositionLU();
    ASSERT_TRUE(A == L * U);
    std::cout << "Mat A:\n" << A << "\n";
    std::cout << "Mat U:\n" << U << "\n";
    std::cout << "Mat L:\n" << L << "\n";
    std::cout << "Mat LU:\n" << L * U << "\n";
}

TEST(TestMatrix, minMaxNormalisation) 
{
    constexpr std::size_t M1{3};
    using Mat = Matrix<double,M1,M1>;
    Mat A{{ Mat::Row{2,-1,0},  Mat::Row{-1,2,-1},  Mat::Row{0,-1,2} }};
    auto ANorma = A.minMaxNormalisationByColumn();
    for( std::size_t i{0}; i < M1 ; ++i )
    {
        for( std::size_t j{0}; j < M1 ; ++j )
        {
            ASSERT_TRUE(ANorma.at(i,j) <= 1);
            ASSERT_TRUE(ANorma.at(i,j) >= 0);
        }
    }
    std::cout << ANorma << std::endl;
}

TEST(TestMatrix, TestOperatorMultiWithDifferentType) 
{
    constexpr std::size_t M1{5};
    using Mat = Matrix<int,M1,M1>;
    Mat mat1{{ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} }};
    Mat expectResult{{ Mat::Row{2,4},  Mat::Row{6,8},  Mat::Row{10,12} }};

    ASSERT_TRUE(expectResult == mat1 * 2.0);
}

TEST(TestMatrix, TestSplit) 
{
    constexpr std::size_t M1{5};
    using Mat = Matrix<int,M1,M1>;
    Mat mat1{{ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} }};

    constexpr std::size_t M2{2};
    using MatExpect1 = Matrix<int,M2,M1>;
    MatExpect1 expectResult{{ MatExpect1::Row{1,2},  MatExpect1::Row{3,4}}};
    
    constexpr std::size_t M3{3};
    using MatExpect2 = Matrix<int,M3,M1>;
    MatExpect2 expectResult2{{ MatExpect2::Row{5,6}}};
    auto [ upper, lower ] = mat1.split<2>();
    std::cout << "Mat mat1:\n" << mat1 << "\n";
    std::cout << "Mat upper:\n" << upper << "\n";
    std::cout << "Mat lower:\n" << lower << "\n";
    ASSERT_TRUE(expectResult == upper);
    ASSERT_TRUE(expectResult2 == lower);
}

TEST(TestMatrix, TestAdditionWithVector) 
{
    constexpr std::size_t M1{5};
    using Mat = Matrix<int,M1,M1>;
    Mat mat1{{ Mat::Row{1,2},  Mat::Row{3,4},  Mat::Row{5,6} }};
    Mat expectResult{{ 
        Mat::Row{2,3,1,1,1},  
        Mat::Row{5,6,2,2,2},  
        Mat::Row{8,9,3,3,3},
        Mat::Row{4,4,4,4,4},  
        Mat::Row{5,5,5,5,5}
    }};

    using Vector = Matrix<int,M1,1>;
    Vector vect{{ Mat::Column{
        1,
        2,
        3,
        4,
        5
    }}};

    ASSERT_TRUE(expectResult == (mat1 + vect));
}

TEST(TestMatrix, TestSolveLinearEquation) 
{

}

}