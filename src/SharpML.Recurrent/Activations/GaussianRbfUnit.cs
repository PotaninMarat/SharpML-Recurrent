using SharpML.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.Activations
{
    public class GaussianRbfUnit : INonlinearity
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
                valueMatrix[i] = Math.Exp(-Math.Pow(x[i],2));
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
                valueMatrix.DataInTensor[i] = 1.0 / z;
            }

            return valueMatrix;
        }


    }
}
