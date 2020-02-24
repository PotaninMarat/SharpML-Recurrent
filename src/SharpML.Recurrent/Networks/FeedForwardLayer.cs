using SharpML.Activations;
using SharpML.DataStructs;
using SharpML.Models;
using SharpML.Networks.Base;
using System;
using System.Collections.Generic;

namespace SharpML.Networks
{
    [Serializable]
    public class FeedForwardLayer : ILayer
    {
        public NNValue _w;
        public NNValue _b;
        public INonlinearity _f;

        /// <summary>
        /// Входная размерность
        /// </summary>
        public Shape InputShape { get; set; }
        /// <summary>
        /// Выходная размерность
        /// </summary>
        public Shape OutputShape { get; private set; }

        public FeedForwardLayer(int inputDimension, int outputDimension, INonlinearity f, double initParamsStdDev, Random rng)
        {
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);
            _w = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _b = new NNValue(outputDimension);
            this._f = f;
        }

        public FeedForwardLayer(Shape inputShape, int outputDimension, INonlinearity f, double initParamsStdDev, Random rng)
        {
            InputShape = inputShape;
            OutputShape = new Shape(outputDimension);
            _w = NNValue.Random(OutputShape.H, InputShape.H, initParamsStdDev, rng);
            _b = new NNValue(outputDimension);
            this._f = f;
        }

        public FeedForwardLayer(int outputDimension, INonlinearity f)
        {
            OutputShape = new Shape(outputDimension);
            this._f = f;
        }

        public NNValue Activate(NNValue input, IGraph g)
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

        public void Generate(Shape inpShape, Random random, double std)
        {
            _w = NNValue.Random(OutputShape.H, inpShape.H, std, random);
            _b = new NNValue(OutputShape.H);
        }
    }
}
