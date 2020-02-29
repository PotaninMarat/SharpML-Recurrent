using SharpML.Models;
using System;

namespace SharpML.Activations
{
    [Serializable]
    public class SigmoidUnit : INonlinearity
    {
        private static long _serialVersionUid = 1L;
        private readonly long _id;

        public long Id
        {
            get { return _id; }
        }

        public SigmoidUnit()
        {
            _id = _serialVersionUid + 1;
        }

        public double Forward(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public double Backward(double x)
        {
            double act = Forward(x);
            return act * (1 - act);
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

        public override string ToString()
        {
            return "Sigmoid";
        }
    }
}
