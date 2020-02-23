using System;
using SharpML.Models;

namespace SharpML.Activations
{
    public class SQRBFUnit : INonlinearity
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
                double absX = Math.Atan(x[i]);

                if (absX <= 1)
                    valueMatrix[i] = 1 - x[i] * x[i] / 2;
                else if (absX >= 2)
                    valueMatrix[i] = 0;
                else
                {
                   valueMatrix[i] = Math.Pow((2 - absX), 2) / 2;
                }
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
                double absX = Math.Atan(x[i]);
                if (absX <= 1)
                    valueMatrix[i] =  - x[i];
                else if (absX >= 2)
                    valueMatrix[i] = 0;
                else
                {
                    valueMatrix[i] = x[i]-2*Math.Sign(x[i]);
                }
            }

            return valueMatrix;
        }

        
    }
}
