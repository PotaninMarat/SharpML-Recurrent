using System;
using System.Collections.Generic;
using SharpML.Models;
using SharpML.Networks.Base;

namespace SharpML.Networks
{
     [Serializable]
    public class LinearLayer : ILayer
    {

        private static long _serialVersionUid = 1L;
         readonly NNValue _w;
        //no biases

        public LinearLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rng)
        {
            _w = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
        }

        public NNValue Activate(NNValue input, IGraph g)
        {
            NNValue returnObj = g.Mul(_w, input);
            return returnObj;
        }

        public void ResetState()
        {

        }

        public List<NNValue> GetParameters()
        {
            List<NNValue> result = new List<NNValue>();
            result.Add(_w);
            return result;
        }
    }
}
