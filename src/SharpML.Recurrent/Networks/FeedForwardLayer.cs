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


        public int TrainableParameters => _w.Len+_b.H;

        /// <summary>
        /// Входная размерность
        /// </summary>
        public Shape InputShape { get; set; }
        /// <summary>
        /// Выходная размерность
        /// </summary>
        public Shape OutputShape { get; private set; }


        public FeedForwardLayer(int inputDimension, int outputDimension, INonlinearity f, Random rng)
        {
            double initParamsStdDev = 1.0 / Math.Sqrt(outputDimension);
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);
            _w = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _b = new NNValue(outputDimension);
            this._f = f;
        }

        public FeedForwardLayer(Shape inputShape, int outputDimension, INonlinearity f, Random rng)
        {
            double initParamsStdDev = 1.0 / Math.Sqrt(outputDimension);
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

        public void Generate(Shape inpShape, Random random, double бесполезный_аргумент)
        {
            InputShape = inpShape;
            double std = 1.0 / Math.Sqrt(OutputShape.H);
            _w = NNValue.Random(OutputShape.H, inpShape.H, std, random);
            _b = new NNValue(OutputShape.H);
        }

        /// <summary>
        /// Описание слоя
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return string.Format("FeedForwardLayer\t|inp: {0} |outp: {1} |Non lin. activate: {3} |TrainParams: {2}", InputShape, OutputShape, TrainableParameters, _f);
        }
    }
}
