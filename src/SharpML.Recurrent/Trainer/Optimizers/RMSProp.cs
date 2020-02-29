using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpML.Trainer.Optimizers
{
    public class RMSProp : IOptimizer
    {

        public double DecayRate = 0.999;
        public double SmoothEpsilon = 1e-8;

        

        /// <summary>
        /// Обновление параметров 
        /// </summary>
        /// <param name="network">Нейросеть</param>
        /// <param name="learningRate">Скорость обучения</param>
        /// <param name="gradClip">Максимальное значение градиента</param>
        public void UpdateModelParams(INetwork network, double learningRate, double gradClip, double L1, double L2)
        {
            Parallel.ForEach(network.GetParameters(), new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, m =>
            {
                for (int i = 0; i < m.Len; i++)
                {
                    double g = m.DifData[i];
                    g += L2 * m[i] + (m[i] > 0 ? L1 : -L1); // Градиент с учетом регуляризации


                    // gradient clip
                    if (g > gradClip)
                    {
                        g = gradClip;
                    }
                    if (g < -gradClip)
                    {
                        g = -gradClip;
                    }

                    m.StepCache[i] = m.StepCache[i] * DecayRate + (1 - DecayRate) * g * g;

                    // update (and regularize)
                    m[i] -= learningRate * g / Math.Sqrt(m.StepCache[i] + SmoothEpsilon);
                    m.DifData[i] = 0;
                }

            });
        }

        public void Reset()
        {
        }
    }
}
