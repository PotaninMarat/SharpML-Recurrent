using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Models;

namespace SharpML.Activations
{
    public class EliotSigUnit : INonlinearity
    {
        /// <summary>
        /// Прямой проход
        /// </summary>
        /// <param name="x">Тензор входа</param>
        /// <returns></returns>
        public NNValue Forward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W, x.D);
            int len = x.Len;

            for (int i = 0; i < len; i++)
            {
                valueMatrix[i] = x[i]/(1+Math.Abs(x[i]));
            }

            return valueMatrix;
        }

        /// <summary>
        /// Обратный проход(производные)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public NNValue Backward(NNValue x)
        {
            NNValue valueMatrix = new NNValue(x.H, x.W, x.D);
            int len = x.Len;

            for (int i = 0; i < len; i++)
            {
                double z = (1.0 + Math.Abs(x[i]));
                z *= z;
                valueMatrix.DataInTensor[i] = 1.0/z;
            }

            return valueMatrix;
        }

       
    }
}
