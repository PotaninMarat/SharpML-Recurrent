using SharpML.Models;

namespace SharpML.Loss
{
    public interface ILoss
    {
        void Backward(NNValue actualOutput, NNValue targetOutput);
        double Measure(NNValue actualOutput, NNValue targetOutput);
    }
}
