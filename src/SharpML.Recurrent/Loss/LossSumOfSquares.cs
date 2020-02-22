using SharpML.Loss;
using SharpML.Models;

namespace SharpML.Loss
{
    public class LossSumOfSquares : ILoss
    {

        public void Backward(NNValue actualOutput, NNValue targetOutput)
        {
            for (int i = 0; i < targetOutput.DataInTensor.Length; i++)
            {
                double errDelta = actualOutput.DataInTensor[i] - targetOutput.DataInTensor[i];
                actualOutput.DifData[i] += errDelta;
            }
        }

        public double Measure(NNValue actualOutput, NNValue targetOutput)
        {
            double sum = 0;
            for (int i = 0; i < targetOutput.DataInTensor.Length; i++)
            {
                double errDelta = actualOutput.DataInTensor[i] - targetOutput.DataInTensor[i];
                sum += 0.5 * errDelta * errDelta;
            }
            return sum;
        }
    }
}
