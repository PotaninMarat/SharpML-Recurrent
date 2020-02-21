using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Loss
{
    public class LossArgMax : ILoss
    {

        public void Backward(NNValue actualOutput, NNValue targetOutput)
        {
            throw new Exception("not implemented");

        }

        public double Measure(NNValue actualOutput, NNValue targetOutput)
        {
            if (actualOutput.DataInTensor.Length != targetOutput.DataInTensor.Length)
            {
                throw new Exception("mismatch");
            }
            double maxActual = Double.PositiveInfinity;
            double maxTarget = Double.NegativeInfinity;
            int indxMaxActual = -1;
            int indxMaxTarget = -1;
            for (int i = 0; i < actualOutput.DataInTensor.Length; i++)
            {
                if (actualOutput.DataInTensor[i] > maxActual)
                {
                    maxActual = actualOutput.DataInTensor[i];
                    indxMaxActual = i;
                }
                if (targetOutput.DataInTensor[i] > maxTarget)
                {
                    maxTarget = targetOutput.DataInTensor[i];
                    indxMaxTarget = i;
                }
            }
            if (indxMaxActual == indxMaxTarget)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }
    }
}
