using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
    [Serializable]
    public class LstmLayer : ILayer
    {

        private static long _serialVersionUid = 1L;
        int _inputDimension;
        readonly int _outputDimension;

        readonly NNValue _wix;
        readonly NNValue _wih;
        readonly NNValue _inputBias;
        readonly NNValue _wfx;
        readonly NNValue _wfh;
        readonly NNValue _forgetBias;
        readonly NNValue _wox;
        readonly NNValue _woh;
        readonly NNValue _outputBias;
        readonly NNValue _wcx;
        readonly NNValue _wch;
        readonly NNValue _cellWriteBias;

        NNValue _hiddenContext;
        NNValue _cellContext;

        readonly INonlinearity _inputGateActivation = new SigmoidUnit();
        readonly INonlinearity _forgetGateActivation = new SigmoidUnit();
        readonly INonlinearity _outputGateActivation = new SigmoidUnit();
        readonly INonlinearity _cellInputActivation = new TanhUnit();
        readonly INonlinearity _cellOutputActivation = new TanhUnit();

        public LstmLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rng)
        {
            this._inputDimension = inputDimension;
            this._outputDimension = outputDimension;
            _wix = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _wih = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _inputBias = new NNValue(outputDimension);
            _wfx = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _wfh = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            //set forget bias to 1.0, as described here: http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
            _forgetBias = NNValue.Ones(outputDimension, 1);
            _wox = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _woh = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _outputBias = new NNValue(outputDimension);
            _wcx = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _wch = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _cellWriteBias = new NNValue(outputDimension);
            ResetState(); // Запуск НС
        }

        public NNValue Activate(NNValue input, Graph g)
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
            _hiddenContext = new NNValue(_outputDimension);
            _cellContext = new NNValue(_outputDimension);
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
    }
}
