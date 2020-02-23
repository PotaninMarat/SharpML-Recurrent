using SharpML.Models;
using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.Activations
{
    public class SqnlUnit : INonlinearity
    {
        public NNValue Forward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W, x.D);
            int len = x.Len;

            for (int i = 0; i < len; i++)
            {
                if (x[i] > 2) valueMatrix[i] = 1;
                else if (x[i] < -2) valueMatrix[i] = -1;
                else if (x[i] < 0) valueMatrix[i] = x[i]+x[i]*x[i]/4;
                else valueMatrix[i] = x[i] - x[i] * x[i] / 4;
            }

            return valueMatrix;
        }

        public NNValue Backward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W, x.D);
            int len = x.Len;

            for (int i = 0; i < len; i++)
            {
                valueMatrix.DataInTensor[i] = 1+x[i]/2;
            }

            return valueMatrix;
        }
    }
}
