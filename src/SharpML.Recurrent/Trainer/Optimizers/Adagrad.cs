using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpML.Networks.Base;

namespace SharpML.Trainer.Optimizers
{
    public class Adagrad : IOptimizer
    {

        public double SmoothEpsilon = 1e-2;

        public void Reset()
        {

        }

        public void UpdateModelParams(INetwork network, double learningRate, double gradClip, double L1, double L2)
        {
            Parallel.ForEach(network.GetParameters(), new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, m =>
            {
                for (int i = 0; i < m.Len; i++)
                {

                    double g = m.DifData[i];
                    g += L2 * m[i] + (m[i] > 0 ? L1 : -L1); // Градиент с учетом регуляризации

                    if (g > gradClip)
                    {
                        g = gradClip;
                    }
                    if (g < -gradClip)
                    {
                        g = -gradClip;
                    }


                    double delt = learningRate * g/Math.Sqrt(m.StepCache[i]+SmoothEpsilon);
                    m[i] -= delt;
                    m.StepCache[i] += g*g;
                    m.DifData[i] = 0;
                }

            });
        }
    }
}
