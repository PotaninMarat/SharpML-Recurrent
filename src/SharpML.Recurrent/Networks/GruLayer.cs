using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
     [Serializable]
    public class GruLayer : ILayer {

	private static  long _serialVersionUid = 1L;
	int _inputDimension;
         readonly int _outputDimension;

         readonly NNValue _hmix;
         readonly NNValue _hHmix;
         readonly NNValue _bmix;
         readonly NNValue _hnew;
         readonly NNValue _hHnew;
         readonly NNValue _bnew;
         readonly NNValue _hreset;
         readonly NNValue _hHreset;
         readonly NNValue _breset;

         NNValue _context;

         readonly INonlinearity _fMix = new SigmoidUnit();
         readonly INonlinearity _fReset = new SigmoidUnit();
         readonly INonlinearity _fNew = new TanhUnit();
	
	public GruLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rng) {
		this._inputDimension = inputDimension;
		this._outputDimension = outputDimension;
		_hmix = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
		_hHmix = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
		_bmix = new NNValue(outputDimension);
		_hnew = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
		_hHnew = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
		_bnew = new NNValue(outputDimension);
		_hreset = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
		_hHreset = NNValue.Random(outputDimension, outputDimension, initParamsStdDev, rng);
		_breset= new NNValue(outputDimension);
	}
	
	public NNValue Activate(NNValue input, Graph g)  {
		
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

	public void ResetState() {
		_context = new NNValue(_outputDimension);
	}

	public List<NNValue> GetParameters() {
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

}
}
