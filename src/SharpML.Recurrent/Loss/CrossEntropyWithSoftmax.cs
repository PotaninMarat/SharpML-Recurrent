using SharpML.Models;
using System;

namespace SharpML.Loss
{
    public class CrossEntropyWithSoftmax : ILoss
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
            var crossentropy = 0.0;

            for (int i = 0; i < actualOutput.DataInTensor.Length; i++)
            {
                crossentropy += targetOutput.DataInTensor[i] * Math.Log(actualOutput.DataInTensor[i] + 1e-15);
            }

            if (double.IsNaN(crossentropy))
            {
                int q = 1;
            }

            return -crossentropy;
        }
    }
}

