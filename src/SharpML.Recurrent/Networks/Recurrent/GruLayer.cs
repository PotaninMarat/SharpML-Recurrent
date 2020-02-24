using System;
using System.Collections.Generic;
using SharpML.Activations;
using SharpML.DataStructs;
using SharpML.Models;
using SharpML.Networks.Base;

namespace SharpML.Networks.Recurrent
{
    [Serializable]
    public class GruLayer : ILayer
    {

        /// <summary>
        /// Входная размерность
        /// </summary>
        public Shape InputShape { get; set; }
        /// <summary>
        /// Выходная размерность
        /// </summary>
        public Shape OutputShape { get; private set; }


        NNValue _hmix;
        NNValue _hHmix;
        NNValue _bmix;
        NNValue _hnew;
        NNValue _hHnew;
        NNValue _bnew;
        NNValue _hreset;
        NNValue _hHreset;
        NNValue _breset;

        NNValue _context;

        INonlinearity _fMix = new SigmoidUnit();
        INonlinearity _fReset = new SigmoidUnit();
        INonlinearity _fNew = new TanhUnit();

        public GruLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rng)
        {
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);

            _hmix = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _hHmix = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _bmix = new NNValue(outputDimension);
            _hnew = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _hHnew = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _bnew = new NNValue(outputDimension);
            _hreset = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _hHreset = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _breset = new NNValue(outputDimension);
        }

        public GruLayer(Shape inputShape, int outputDimension, double initParamsStdDev, Random rng)
        {
            int inputDimension = inputShape.H;
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);

            _hmix = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _hHmix = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _bmix = new NNValue(outputDimension);
            _hnew = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _hHnew = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _bnew = new NNValue(outputDimension);
            _hreset = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _hHreset = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _breset = new NNValue(outputDimension);
        }

        public GruLayer(int outputDimension)
        {
            OutputShape = new Shape(outputDimension);
        }



        public NNValue Activate(NNValue input, IGraph g)
        {

            NNValue sum0 = g.Mul(_hmix, input);
            NNValue sum1 = g.Mul(_hHmix, _context);
            NNValue actMix = g.Nonlin(_fMix, g.Add(g.Add(sum0, sum1), _bmix));

            NNValue sum2 = g.Mul(_hreset, input);
            NNValue sum3 = g.Mul(_hHreset, _context);
            NNValue actReset = g.Nonlin(_fReset, g.Add(g.Add(sum2, sum3), _breset));

            NNValue sum4 = g.Mul(_hnew, input);
            NNValue gatedContext = g.Elmul(actReset, _context);
            NNValue sum5 = g.Mul(_hHnew, gatedContext);
            NNValue actNewPlusGatedContext = g.Nonlin(_fNew, g.Add(g.Add(sum4, sum5), _bnew));

            NNValue memvals = g.Elmul(actMix, _context);
            NNValue newvals = g.Elmul(g.OneMinus(actMix), actNewPlusGatedContext);
            NNValue output = g.Add(memvals, newvals);

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
            result.Add(_hmix);
            result.Add(_hHmix);
            result.Add(_bmix);
            result.Add(_hnew);
            result.Add(_hHnew);
            result.Add(_bnew);
            result.Add(_hreset);
            result.Add(_hHreset);
            result.Add(_breset);
            return result;
        }

        /// <summary>
        /// Генерация слоя НС
        /// </summary>
        /// <param name="inpShape"></param>
        /// <param name="random"></param>
        /// <param name="std"></param>
        /// <returns></returns>
        public void Generate(Shape inpShape, Random random, double std)
        {
            Init(inpShape, OutputShape.H, std, random);
        }

        void Init(Shape inputShape, int outputDimension, double initParamsStdDev, Random rng)
        {
            int inputDimension = inputShape.H;
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);

            _hmix = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _hHmix = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _bmix = new NNValue(outputDimension);
            _hnew = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _hHnew = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _bnew = new NNValue(outputDimension);
            _hreset = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _hHreset = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _breset = new NNValue(outputDimension);
        }

    }
}
