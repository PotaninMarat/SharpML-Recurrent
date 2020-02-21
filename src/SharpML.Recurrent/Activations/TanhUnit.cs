using SharpML.Recurrent.Models;
using System;

namespace SharpML.Recurrent.Activations
{
    [Serializable]
    public class TanhUnit : INonlinearity
    {
        private static long _serialVersionUid = 1L;
        private readonly long _id;

        public long Id
        {
            get { return _id; }
        }

        public TanhUnit()
        {
            _id = _serialVersionUid + 1;
        }

        public double Forward(double x)
        {
            return Math.Tanh(x);
        }

        public double Backward(double x)
        {
            double coshx = Math.Cosh(x);
            double denom = (Math.Cosh(2 * x) + 1);
            return 4 * coshx * coshx / (denom * denom);
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
