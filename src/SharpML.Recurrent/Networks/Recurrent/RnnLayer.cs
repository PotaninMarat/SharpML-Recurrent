using System;
using System.Collections.Generic;
using SharpML.Networks.Base;
using SharpML.Activations;
using SharpML.Models;
using SharpML.DataStructs;

namespace SharpML.Networks.Recurrent
{
    [Serializable]
    public class RnnLayer : ILayer
    {

        /// <summary>
        /// Входная размерность
        /// </summary>
        public Shape InputShape { get; set; }
        /// <summary>
        /// Выходная размерность
        /// </summary>
        public Shape OutputShape { get; private set; }

        NNValue _w;
        NNValue _b;

        NNValue _context;

        INonlinearity _f;

        public RnnLayer(int inputDimension, int outputDimension, INonlinearity hiddenUnit, double initParamsStdDev,
            Random rng)
        {
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);
            _f = hiddenUnit;
            _w = NNValue.Random(outputDimension, inputDimension + outputDimension, initParamsStdDev, rng);
            _b = new NNValue(outputDimension);
            ResetState();
        }

        public RnnLayer(Shape inputShape, int outputDimension, INonlinearity hiddenUnit, double initParamsStdDev,
            Random rng)
        {
            int inputDimension = inputShape.H;
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);
            _f = hiddenUnit;
            _w = NNValue.Random(outputDimension, inputDimension + outputDimension, initParamsStdDev, rng);
            _b = new NNValue(outputDimension);
            ResetState();
        }

        public RnnLayer(int outputDimension, INonlinearity hiddenUnit)
        {
            OutputShape = new Shape(outputDimension);
            _f = hiddenUnit;
        }

        public NNValue Activate(NNValue input, IGraph g)
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
            _context = new NNValue(OutputShape.H);
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
            Init(inpShape, OutputShape.H, _f, std, random);
        }

        void Init(Shape inputShape, int outputDimension, INonlinearity hiddenUnit, double initParamsStdDev,
            Random rng)
        {
            int inputDimension = inputShape.H;
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);
            _f = hiddenUnit;
            _w = NNValue.Random(outputDimension, inputDimension + outputDimension, initParamsStdDev, rng);
            _b = new NNValue(outputDimension);
            ResetState();
        }
    }
}
