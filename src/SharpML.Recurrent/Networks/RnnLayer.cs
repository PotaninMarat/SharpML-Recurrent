using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
     [Serializable]
    public class RnnLayer : ILayer
    {

        private static long _serialVersionUid = 1L;
        private int _inputDimension;
        private readonly int _outputDimension;

        private readonly NNValue _w;
         private readonly NNValue _b;

         private NNValue _context;

        private readonly INonlinearity _f;

        public RnnLayer(int inputDimension, int outputDimension, INonlinearity hiddenUnit, double initParamsStdDev,
            Random rng)
        {
            this._inputDimension = inputDimension;
            this._outputDimension = outputDimension;
            this._f = hiddenUnit;
            _w = NNValue.Random(outputDimension, inputDimension + outputDimension, initParamsStdDev, rng);
            _b = new NNValue(outputDimension);
            ResetState();
        }

        public NNValue Activate(NNValue input, Graph g)
        {
            NNValue concat = g.ConcatVectors(input, _context);
            NNValue sum = g.Mul(_w, concat); sum = g.Add(sum, _b);
            NNValue output = g.Nonlin(_f, sum);

            //rollover activations for next iteration
            _context = output;

            return output;
        }


        public void ResetState()
        {
            _context = new NNValue(_outputDimension);
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
