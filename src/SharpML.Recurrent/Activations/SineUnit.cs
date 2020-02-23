using SharpML.Models;
using System;

namespace SharpML.Activations
{
    [Serializable]
    public class SineUnit : INonlinearity
    {
        private static long _serialVersionUid = 1L;
        private readonly long _id;

        public long Id
        {
            get { return _id; }
        }

        public SineUnit()
        {
            _id = _serialVersionUid + 1;
        }

        public double Forward(double x)
        {
            return Math.Sin(x);
        }

        public double Backward(double x)
        {
            return Math.Cos(x);
        }

        public NNValue Forward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W, x.D);
            int len = x.DataInTensor.Length;

            for (int i = 0; i < len; i++)
            {
                valueMatrix.DataInTensor[i] = Forward(x.DataInTensor[i]);
            }

            return valueMatrix;
        }

        public NNValue Backward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W, x.D);
            int len = x.DataInTensor.Length;

            for (int i = 0; i < len; i++)
            {
                valueMatrix.DataInTensor[i] = Backward(x.DataInTensor[i]);
            }

            return valueMatrix;
        }
    }
}
