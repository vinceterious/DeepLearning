#pragma once

#include <array>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <type_traits>

template<class Type, std::size_t M, std::size_t N>
class Matrix
{
public:

    using Row = std::array<Type,N>;
    using Column = std::array<Type,M>;
    using Mat = std::array<Row,M>;

    Matrix()
    {
        std::for_each(mat.begin(),mat.end(),[](auto& row){ row.fill(0); });
    };

    Matrix(Type filler)
    {
        std::for_each(mat.begin(),mat.end(),[&filler](auto& row){ row.fill(filler); });
    };

    Matrix(const Mat& m) : mat{ m }
    {
    };

    template<class T = Column>
    Matrix(const std::array<T,N>& c) requires (std::is_same<T,Row>::value == false) 
        //Avoid same contructor declaration as above when M=N
        : mat{}
    {
        for(std::size_t i{0} ; i < M ; i++)
        {
            for(std::size_t y{0}; y < N; y++)
            {
                mat.at(i).at(y) = c.at(y).at(i);
            }
        }
    };

    constexpr std::size_t m() const { return M; };
    constexpr std::size_t n() const { return N; };
    constexpr std::size_t size() const { return M*N; };

    Type& access(std::size_t iRow, std::size_t iColumn)
    {
        return mat.at(iRow).at(iColumn);
    }

    const auto& elemn0()
    {
        return mat.cbegin()->cbegin();
    }

    Type at(std::size_t m, std::size_t n) const
    {
        if (m*N+n >= size() or n >= N)
        {
            std::__throw_out_of_range_fmt(__N("Matrix::at: A_mn (which is m=%zu n=%zu) "
                    ">= MxN (which is %zux%zu)"),
                m,n, M,N);
        }
        return *(mat.cbegin()->cbegin() + (m*N + n));
    }

    template<class InputIt>
    void copyStartAt(InputIt first, InputIt last, const std::size_t m, const std::size_t n)
    {
        if (m*N+n >= size() or n >= N)
        {
            std::__throw_out_of_range_fmt(__N("Matrix::copyStartAt: A_mn (which is m=%zu n=%zu) "
                    ">= MxN (which is %zux%zu)"),
                m,n, N,M);
        }
        auto d_first = (mat.begin() + m)->begin() + n; 
        for (; first != last; (void)++first, (void)++d_first)
            *d_first = *first;
    }

    template<std::size_t T>
    Matrix<Type,M+T,N> concatRow(const std::array<std::array<Type,N>,T> row)
    {
        Matrix<Type,M+T,N> newMat{};
        newMat.copyStartAt(mat.cbegin()->cbegin(),mat.cend()->cbegin(), 0,0);
        newMat.copyStartAt(row.cbegin()->cbegin(),row.cend()->cbegin(), M,0);
        return newMat;
    }

    Matrix<Type,N,M> T()
    {
        Matrix<Type,N,M> transposeMat{};
        for(std::size_t x{0} ; x < M ; x++)
        {
            for(std::size_t y{0} ; y < N ; y++)
            {
                transposeMat.access(y,x) = at(x,y);
            }
        }
        return transposeMat;
    }

    std::tuple<Matrix<Type,M,N>,Matrix<Type,M,N>> decompositionLU()
    {
        static_assert(M ==N, "decomposition LU only works on square matrice.");
        Matrix<Type,M,N> l {0};
        Matrix<Type,M,N> u {0};
        for (std::size_t i = 0; i < M; i++) 
        {
            for (std::size_t j = i; j < M; j++) 
            {
                l.access(j,i) = at(j,i);
                for (std::size_t k = 0; k < i; k++) 
                {
                    l.access(j,i) -= l.access(j,k)*u.at(k,i);
                }
            }
            
            for (std::size_t j = i; j < M; j++) 
            {
                if (j == i)
                {
                    u.access(i,j) = 1;
                }
                else 
                {
                    u.access(i,j) = at(i,j) / l.at(i,i);
                    for (std::size_t k = 0; k < i; k++) 
                    {
                        u.access(i,j) -= l.at(i,k) * u.at(k,j) / l.at(i,i);
                    }  
                }
            }
        }
        return {l,u};
    }

    Matrix<Type,M,1> solveLinearEquationViaLU(const std::tuple<Matrix<Type,M,M>,Matrix<Type,M,M>> lu )
    {
        static_assert(N==1, "Solving on vector b");
        using Vector = Matrix<Type,M,1>;
        constexpr std::size_t L{0};
        constexpr std::size_t U{1};
        Vector y{0};
        Vector x{0};

        for( std::size_t i{0} ; i < M ; ++i )
        {
            y.access(i,0) = at(i,0);
            for(std::size_t j{0} ; j < i; ++j)
            {
                y.access(i,0)-= std::get<L>(lu).at(i,j)*y.at(j,0);
            }
            y.access(i,0) /= std::get<L>(lu).at(i,i);
        }
        for( std::size_t i{M-1} ; i < M ; --i )
        {
            x.access(i,0) = y.at(i,0);
            for(std::size_t j{M-1} ; j > i; --j)
            {
                x.access(i,0)-= std::get<U>(lu).at(i,j)*x.at(j,0);
            }
            x.access(i,0) /= std::get<U>(lu).at(i,i);
        }
        return x;
    }

    template <typename T>
    Matrix<Type,M,N> multiplicationEach(T factor) requires std::integral<T> || std::floating_point<T>
    {
        Matrix<Type,M,N> matResult{};
        for(std::size_t x{0} ; x < M ; x++)
        {
            for(std::size_t y{0} ; y < N ; y++)
            {
                matResult.access(x,y) = at(x,y)*factor;
            }
        }
        return matResult;
    }


    Matrix<Type,M,N> squaredEach()
    {
        Matrix<Type,M,N> matResult{};
        for(std::size_t x{0} ; x < M ; x++)
        {
            for(std::size_t y{0} ; y < N ; y++)
            {
                matResult.access(x,y) = at(x,y)*at(x,y);
            }
        }
        return matResult;
    }

    Matrix<Type,M,N> minMaxNormalisationByColumn()
    {
        Matrix<Type,M,N> matResult{};
        for(std::size_t y{0} ; y < N ; y++)
        {
            auto min{0.0};
            auto max{0.0};
            for(std::size_t x{0} ; x < M ; x++)
            {
                min = std::min(static_cast<Type>(min),at(x,y));
                max = std::max(static_cast<Type>(max),at(x,y));
            }
            for(std::size_t x{0} ; x < M ; x++)
            {
                matResult.access(x,y) = (at(x,y) - min) / (max - min);
            }
        }
        return matResult;
    }

    double moyenne()
    {
        Type res{sum()};
        return static_cast<double>(res/static_cast<double>(M));
    }

    Type sum() const
    {
        return std::accumulate(mat.cbegin()->cbegin(), mat.cend()->cbegin(),0.0,std::plus<Type>());        
    }

    template <std::size_t Cut>
    std::tuple<Matrix<Type,Cut,N>,Matrix<Type,M-Cut,N>> split() const requires (M > Cut)
    {
        Matrix<Type,Cut,N> matUpper{};
        Matrix<Type,M-Cut,N> matLower{};
        matUpper.copyStartAt(mat.cbegin()->cbegin(),(mat.cbegin() + Cut)->cbegin(), 0,0);
        matLower.copyStartAt((mat.cbegin() + Cut)->cbegin(),mat.cend()->cbegin(), 0,0);
        return {matUpper , matLower};        
    }

    bool operator==(const Matrix<Type,M,N>& lhs)
    {
        return std::ranges::equal(lhs.mat,mat);
    }

    Matrix<Type,M,N> operator-(const Matrix<Type,M,N>& lhs)
    {
        Matrix<Type,M,N> matResult{};
        for(std::size_t x{0} ; x < M ; x++)
        {
            for(std::size_t y{0} ; y < N ; y++)
            {
                matResult.access(x,y) = static_cast<Type>(at(x,y) - lhs.at(x,y));
            }
        }
        return matResult;
    }

    template<typename T>
    Matrix<Type,M,N> operator-(const T& lhs) requires std::integral<T> || std::floating_point<T>
    {
        Matrix<Type,M,N> matResult{};
        for(std::size_t x{0} ; x < M ; x++)
        {
            for(std::size_t y{0} ; y < N ; y++)
            {
                matResult.access(x,y) = static_cast<Type>(at(x,y) - lhs);
            }
        }
        return matResult;
    }


    template<std::size_t T>
    Matrix<Type,M,T> operator*(Matrix<Type,N,T> mat1)
    {
        Matrix<Type,M,T> matrix{};   
        for(std::size_t i=0; i< M ;i++)    
        {    
            for(std::size_t j=0; j<T;j++)    
            {    
                matrix.access(i,j)=0;    
                for(std::size_t k=0; k<N; k++)    
                {    
                    matrix.access(i,j)+=at(i,k)*mat1.at(k,j);    
                }
            }
        }
        return matrix;
    }
    
    template <typename T>
    Matrix<Type,M,N> operator*(const T& lhs) requires std::integral<T> || std::floating_point<T>
    {
        Matrix<Type,M,N> matResult{};
        for(std::size_t x{0} ; x < M ; x++)
        {
            for(std::size_t y{0} ; y < N ; y++)
            {
                matResult.access(x,y) = static_cast<Type>(at(x,y)*lhs);
            }
        }
        return matResult;
    }

    friend std::ostream &operator<<( std::ostream &output, const Matrix<Type,M,N> &m ) { 
        std::for_each( m.mat.cbegin(), m.mat.cend(), [&output](const auto& row){
            output << "{";
            std::for_each(row.cbegin(), row.cend(), [&output](const auto& c){
                output << c << ",";
              });
            output << "\b},\n";
         });
        return output;            
    }


private:
    Mat mat;

};
