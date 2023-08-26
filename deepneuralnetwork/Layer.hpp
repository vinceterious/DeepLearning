#pragma once
#include "../matrix/Matrix.hpp"

namespace deepneuralnetwork
{

namespace 
{

}
struct LogLoss
{
    //to be implemnted 
    auto operator()(auto a)
    {
        return a;
    }
};

template<class PreviousLayer, std::size_t NumberOfNeurone, class ActivationMethod = LogLoss >
class Layer
{
public:
    using W = Matrix<double,NumberOfNeurone,PreviousLayer::m()>;
    using A = Matrix<double,NumberOfNeurone,PreviousLayer::n()>;
    using b = Matrix<double,NumberOfNeurone,1>;

    static constexpr std::size_t m() { return A::m(); };
    static constexpr std::size_t n() { return A::n(); };

    void forward_propag(const auto& prevActivationMethod)
    {
        activation = ActivationMethod{}(w * prevActivationMethod + bias);
    }

    const A& getActivation() const
    {
        return activation;
    }

private:
    W w{1};
    A activation{};
    b bias{};
};

}