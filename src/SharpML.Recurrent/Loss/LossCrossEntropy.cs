using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Loss
{
    public class CrossEntropy : ILoss
    {

        public void Backward(NNValue actualOutput, NNValue targetOutput)
        {
            throw new Exception("not implemented");

        }

        public double Measure(NNValue target, NNValue actual)
        {
            var crossentropy = 0.0;

            for (int i = 0; i < actual.DataInTensor.Length; i++)
            {

                crossentropy -= (target.DataInTensor[i] * Math.Log(actual.DataInTensor[i] + 1e-15)) +
                                ((1 - target.DataInTensor[i]) * Math.Log((1 + 1e-15) - actual.DataInTensor[i]));
            }


            return crossentropy;
        }




    }
}
