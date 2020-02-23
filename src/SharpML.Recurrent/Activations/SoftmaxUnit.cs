using System;
using SharpML.Models;

namespace SharpML.Activations
{
    public class SoftmaxUnit : INonlinearity
    {

        static double maxActivate = 1e+7;

        public NNValue Forward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W, x.D);
            int len = x.DataInTensor.Length;
            double summ = 0;

            for (int i = 0; i < len; i++)
            {
                valueMatrix.DataInTensor[i] = Math.Exp(x.DataInTensor[i]);
                valueMatrix.DataInTensor[i] = valueMatrix.DataInTensor[i] > maxActivate ? maxActivate : valueMatrix.DataInTensor[i];
                summ += valueMatrix.DataInTensor[i];
            }

            for (int i = 0; i < len; i++)
            {
                valueMatrix.DataInTensor[i] /= summ;
            }

            return valueMatrix;
        }

        public NNValue Backward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W, x.D);
            int len = x.DataInTensor.Length;

            for (int i = 0; i < len; i++)
            {
                valueMatrix.DataInTensor[i] = 1;
            }

            return valueMatrix;
        }
    }
}
