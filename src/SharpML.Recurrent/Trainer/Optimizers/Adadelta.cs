using System;
using System.Threading.Tasks;
using SharpML.Networks.Base;

namespace SharpML.Trainer.Optimizers
{
    public class Adadelta : IOptimizer
    {
        public double DecayRate = 0.999;
        public double SmoothEpsilon = 1e-8;
        public double SmoothEpsilon2 = 1e-4;

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

                    // gradient clip
                    if (g > gradClip)
                    {
                        g = gradClip;
                    }
                    if (g < -gradClip)
                    {
                        g = -gradClip;
                    }

                    m.StepCache2[i] = (m.StepCache2[i]==0)? SmoothEpsilon2 : m.StepCache2[i];

                    m.StepCache[i] = m.StepCache[i] * DecayRate + (1 - DecayRate) * g * g;
                    double delta = g*Math.Sqrt(m.StepCache2[i] + SmoothEpsilon)/Math.Sqrt(m.StepCache[i] + SmoothEpsilon);
                    m.StepCache2[i] = m.StepCache2[i] * DecayRate + (1 - DecayRate) * delta * delta;

                    m[i] -= learningRate * delta;
                    m.DifData[i] = 0;
                }

            });
        }





    }
}
