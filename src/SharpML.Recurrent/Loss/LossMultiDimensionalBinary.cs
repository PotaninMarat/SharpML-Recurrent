using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Loss
{
    public class LossMultiDimensionalBinary : ILoss
    {

        public void Backward(NNValue actualOutput, NNValue targetOutput)
        {
            throw new NotImplementedException("not implemented");
        }

        public double Measure(NNValue actualOutput, NNValue targetOutput)
        {
            if (actualOutput.DataInTensor.Length != targetOutput.DataInTensor.Length)
            {
                throw new Exception("mismatch");
            }

            for (int i = 0; i < targetOutput.DataInTensor.Length; i++)
            {
                if (targetOutput.DataInTensor[i] >= 0.5 && actualOutput.DataInTensor[i] < 0.5)
                {
                    return 1;
                }
                if (targetOutput.DataInTensor[i] < 0.5 && actualOutput.DataInTensor[i] >= 0.5)
                {
                    return 1;
                }
            }
            return 0;
        }

    }
}
