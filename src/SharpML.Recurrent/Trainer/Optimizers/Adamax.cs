﻿using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpML.Trainer.Optimizers
{
    public class Adamax : IOptimizer
    {
        public double b1 = 0.9;
        public double b2 = 0.999;
        public double SmoothEpsilon = 1e-8;
        double newB1, newB2;
        int p = 10;

        public Adamax()
        {
            newB1 = b1;
            newB2 = b2;
        }


        public void UpdateModelParams(INetwork network, double learningRate, double gradClip, double L1, double L2)
        {
            double b2p = Math.Pow(b2, p);
            double mt = 0, vt = 0;

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

                    m.StepCache[i] = b1 * m.StepCache[i] + (1 - b1) * g;
                    m.StepCache2[i] = b2p * m.StepCache2[i] + (1 - b2p) * Math.Pow(Math.Abs(g), p);

                    mt = m.StepCache[i] / (1 - newB1);
                    vt = Math.Pow(m.StepCache2[i], 1.0 / p) / (1 - newB2);

                    m[i] -= learningRate * mt / (Math.Sqrt(vt + SmoothEpsilon));

                    m.DifData[i] = 0;
                }

            });

            newB1 *= b1;
            newB2 *= b2;
        }


        public void Reset()
        {
            newB1 = b1;
            newB2 = b2;
        }
    }
}
