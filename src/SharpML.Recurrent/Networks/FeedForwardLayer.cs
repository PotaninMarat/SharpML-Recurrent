using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
    [Serializable]
    public class FeedForwardLayer : ILayer
    {

        private static long _serialVersionUid = 1L;
        public readonly NNValue _w;
        public readonly NNValue _b;
        public readonly INonlinearity _f;

        public FeedForwardLayer(int inputDimension, int outputDimension, INonlinearity f, double initParamsStdDev, Random rng)
        {
            _w = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _b = new NNValue(outputDimension);
            this._f = f;
        }

        public NNValue Activate(NNValue input, Graph g)
        {
            NNValue sum = g.Add(g.Mul(_w, input), _b);
            NNValue returnObj = g.Nonlin(_f, sum);
            return returnObj;
        }

        public void ResetState()
        {

        }

        public List<NNValue> GetParameters()
        {
            List<NNValue> result = new List<NNValue>();
            result.Add(_w);
            result.Add(_b);
            return result;
        }
    }
}
