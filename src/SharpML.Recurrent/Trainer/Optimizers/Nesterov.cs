using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpML.Networks.Base;

namespace SharpML.Trainer.Optimizers
{
    public class Nesterov : IOptimizer
    {
        public double Momentum { get; set; }
        double MomentumInv;


        public Nesterov()
        {
            Momentum = 0;
            MomentumInv = 1;
        }

        public Nesterov(double momentum)
        {
            double m = Math.Abs(momentum);
            Momentum = (m > 0.99) ? 0.99 : m;
            MomentumInv = 1 - Momentum;
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

                    double delt = MomentumInv*(learningRate * g) + Momentum * m.StepCache[i] ;
                    m[i] -= delt;
                    m.StepCache[i] = delt;
                    m.DifData[i] = 0;
                }

            });
        }

        public void Reset()
        {

        }
    }
}
