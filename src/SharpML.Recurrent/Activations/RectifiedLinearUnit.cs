﻿using System;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Activations
{
    [Serializable]
    public class RectifiedLinearUnit : INonlinearity
    {

        private readonly double _slope;

        

        public RectifiedLinearUnit()
        {
            this._slope = 0;
        }

        public RectifiedLinearUnit(double slope)
        {
            this._slope = slope;
        }

        double Forward(double x)
        {
            if (x >= 0)
            {
                return x;
            }
            else
            {
                return x * _slope;
            }
        }

        double Backward(double x)
        {
            if (x >= 0)
            {
                return 1.0;
            }
            else
            {
                return _slope;
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
