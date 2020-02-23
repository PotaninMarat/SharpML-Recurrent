using SharpML.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.Activations
{
    public class ArcTanUnit : INonlinearity
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
                valueMatrix[i] =  Math.Atan(x[i]);
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
                double z = x[i];
                z *= z;
                z += 1;

                valueMatrix.DataInTensor[i] = 1.0 / z;
            }

            return valueMatrix;
        }


    }
}
