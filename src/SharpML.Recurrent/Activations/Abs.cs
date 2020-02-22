using SharpML.Models;
using System;

namespace SharpML.Activations
{
    [Serializable]
    public class AbsUnit : INonlinearity
    {


        public double Forward(double x)
        {
            return Math.Abs(x);
        }

        public double Backward(double x)
        {
            if (x >= 0)
            {
                return 1.0;
            }
            else
            {
                return -1.0;
            }
        }

        public NNValue Forward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W);
            int len = x.DataInTensor.Length;

            for (int i = 0; i < len; i++)
            {
                valueMatrix.DataInTensor[i] = Forward(x.DataInTensor[i]);
            }

            return valueMatrix;
        }

        public NNValue Backward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W);
            int len = x.DataInTensor.Length;

            for (int i = 0; i < len; i++)
            {
                valueMatrix.DataInTensor[i] = Backward(x.DataInTensor[i]);
            }

            return valueMatrix;
        }
    }
}