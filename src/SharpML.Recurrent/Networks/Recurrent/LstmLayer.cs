using System;
using System.Collections.Generic;
using SharpML.Activations;
using SharpML.DataStructs;
using SharpML.Models;
using SharpML.Networks.Base;

namespace SharpML.Networks.Recurrent
{
    [Serializable]
    public class LstmLayer : ILayer
    {

        /// <summary>
        /// Входная размерность
        /// </summary>
        public Shape InputShape { get; set; }
        /// <summary>
        /// Выходная размерность
        /// </summary>
        public Shape OutputShape { get; private set; }

        NNValue _wix;
        NNValue _wih;
        NNValue _inputBias;
        NNValue _wfx;
        NNValue _wfh;
        NNValue _forgetBias;
        NNValue _wox;
        NNValue _woh;
        NNValue _outputBias;
        NNValue _wcx;
        NNValue _wch;
        NNValue _cellWriteBias;

        NNValue _hiddenContext;
        NNValue _cellContext;

        INonlinearity _inputGateActivation = new SigmoidUnit();
        INonlinearity _forgetGateActivation = new SigmoidUnit();
        INonlinearity _outputGateActivation = new SigmoidUnit();
        INonlinearity _cellInputActivation = new TanhUnit();
        INonlinearity _cellOutputActivation = new TanhUnit();

        public LstmLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rnd)
        {
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);
            _wix = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _wih = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _inputBias = new NNValue(outputDimension);
            _wfx = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _wfh = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            //set forget bias to 1.0, as described here: http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
            _forgetBias = NNValue.Ones(outputDimension, 1);
            _wox = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _woh = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _outputBias = new NNValue(outputDimension);
            _wcx = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _wch = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _cellWriteBias = new NNValue(outputDimension);
            ResetState(); // Запуск НС
        }

        public LstmLayer(Shape inputShape, int outputDimension, double initParamsStdDev, Random rnd)
        {
            int inputDimension = inputShape.H;
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);
            _wix = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _wih = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _inputBias = new NNValue(outputDimension);
            _wfx = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _wfh = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            //set forget bias to 1.0, as described here: http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
            _forgetBias = NNValue.Ones(outputDimension, 1);
            _wox = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _woh = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _outputBias = new NNValue(outputDimension);
            _wcx = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _wch = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _cellWriteBias = new NNValue(outputDimension);
            ResetState(); // Запуск НС
        }

        public LstmLayer(int outputDimension)
        {
            OutputShape = new Shape(outputDimension);
        }

        public NNValue Activate(NNValue input, IGraph g)
        {

            //input gate
            NNValue sum0 = g.Mul(_wix, input);
            NNValue sum1 = g.Mul(_wih, _hiddenContext);
            NNValue inputGate = g.Nonlin(_inputGateActivation, g.Add(g.Add(sum0, sum1), _inputBias));

            //forget gate
            NNValue sum2 = g.Mul(_wfx, input);
            NNValue sum3 = g.Mul(_wfh, _hiddenContext);
            NNValue forgetGate = g.Nonlin(_forgetGateActivation, g.Add(g.Add(sum2, sum3), _forgetBias));

            //output gate
            NNValue sum4 = g.Mul(_wox, input);
            NNValue sum5 = g.Mul(_woh, _hiddenContext);
            NNValue outputGate = g.Nonlin(_outputGateActivation, g.Add(g.Add(sum4, sum5), _outputBias));

            //write operation on cells
            NNValue sum6 = g.Mul(_wcx, input);
            NNValue sum7 = g.Mul(_wch, _hiddenContext);
            NNValue cellInput = g.Nonlin(_cellInputActivation, g.Add(g.Add(sum6, sum7), _cellWriteBias));

            //compute new cell activation
            NNValue retainCell = g.Elmul(forgetGate, _cellContext);
            NNValue writeCell = g.Elmul(inputGate, cellInput);
            NNValue cellAct = g.Add(retainCell, writeCell);

            //compute hidden state as gated, saturated cell activations
            NNValue output = g.Elmul(outputGate, g.Nonlin(_cellOutputActivation, cellAct));

            //rollover activations for next iteration
            _hiddenContext = output;
            _cellContext = cellAct;

            return output;
        }

        public void ResetState()
        {
            _hiddenContext = new NNValue(OutputShape.H);
            _cellContext = new NNValue(OutputShape.H);
        }

        public List<NNValue> GetParameters()
        {
            List<NNValue> result = new List<NNValue>();
            result.Add(_wix);
            result.Add(_wih);
            result.Add(_inputBias);
            result.Add(_wfx);
            result.Add(_wfh);
            result.Add(_forgetBias);
            result.Add(_wox);
            result.Add(_woh);
            result.Add(_outputBias);
            result.Add(_wcx);
            result.Add(_wch);
            result.Add(_cellWriteBias);
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

        /// <summary>
        /// Инициализация сети
        /// </summary>
        void Init(Shape inputShape, int outputDimension, double initParamsStdDev, Random rnd)
        {
            int inputDimension = inputShape.H;
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);
            _wix = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _wih = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _inputBias = new NNValue(outputDimension);
            _wfx = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _wfh = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _forgetBias = NNValue.Ones(outputDimension, 1);
            _wox = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _woh = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _outputBias = new NNValue(outputDimension);
            _wcx = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rnd);
            _wch = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rnd);
            _cellWriteBias = new NNValue(outputDimension);
            ResetState();
        }
    }
}
