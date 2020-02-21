using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent;
using SharpML.Recurrent.DataStructs;
using SharpML.Recurrent.Models;
using SharpML.Recurrent.Networks;

namespace SharpML.Recurrent.Loss
{
    public class LossSoftmax : ILoss
    {

        public void Backward(NNValue logprobs, NNValue targetOutput)
        {
            int targetIndex = GetTargetIndex(targetOutput);
            NNValue probs = GetSoftmaxProbs(logprobs, 1.0);
            for (int i = 0; i < probs.DataInTensor.Length; i++)
            {
                logprobs.DifData[i] = probs.DataInTensor[i];
            }
            logprobs.DifData[targetIndex] -= 1;
        }

        public double Measure(NNValue logprobs, NNValue targetOutput)
        {
            int targetIndex = GetTargetIndex(targetOutput);
            NNValue probs = GetSoftmaxProbs(logprobs, 1.0);
            double cost = -Math.Log(probs.DataInTensor[targetIndex]);
            return cost;
        }

        public static double CalculateMedianPerplexity(ILayer layer, List<DataSequence> sequences)
        {
            double temperature = 1.0;
            List<Double> ppls = new List<Double>();
            foreach (DataSequence seq in sequences)
            {
                double n = 0;
                double neglog2Ppl = 0;

                Graph g = new Graph(false);
                layer.ResetState();
                foreach (DataStep step in seq.Steps)
                {
                    NNValue logprobs = layer.Activate(step.Input, g);
                    NNValue probs = GetSoftmaxProbs(logprobs, temperature);
                    int targetIndex = GetTargetIndex(step.TargetOutput);
                    double probOfCorrect = probs.DataInTensor[targetIndex];
                    double log2Prob = Math.Log(probOfCorrect) / Math.Log(2); //change-of-base
                    neglog2Ppl += -log2Prob;
                    n += 1;
                }

                n -= 1; //don't count first symbol of sentence
                double ppl = Math.Pow(2, (neglog2Ppl / (n - 1)));
                ppls.Add(ppl);
            }
            return Util.Util.Median(ppls);
        }

        public static NNValue GetSoftmaxProbs(NNValue logprobs, double temperature)
        {
            NNValue probs = new NNValue(logprobs.DataInTensor.Length);
            if (temperature != 1.0)
            {
                for (int i = 0; i < logprobs.DataInTensor.Length; i++)
                {
                    logprobs.DataInTensor[i] /= temperature;
                }
            }
            double maxval = Double.NegativeInfinity;
            for (int i = 0; i < logprobs.DataInTensor.Length; i++)
            {
                if (logprobs.DataInTensor[i] > maxval)
                {
                    maxval = logprobs.DataInTensor[i];
                }
            }
            double sum = 0;
            for (int i = 0; i < logprobs.DataInTensor.Length; i++)
            {
                probs.DataInTensor[i] = Math.Exp(logprobs.DataInTensor[i] - maxval); //all inputs to exp() are non-positive
                sum += probs.DataInTensor[i];
            }
            for (int i = 0; i < probs.DataInTensor.Length; i++)
            {
                probs.DataInTensor[i] /= sum;
            }
            return probs;
        }

        private static int GetTargetIndex(NNValue targetOutput)
        {
            for (int i = 0; i < targetOutput.DataInTensor.Length; i++)
            {
                if (targetOutput.DataInTensor[i] == 1.0)
                {
                    return i;
                }
            }
            throw new Exception("no target index selected");
        }
    }
}
