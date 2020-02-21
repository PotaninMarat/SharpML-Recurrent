using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Activations
{
    public interface INonlinearity
    {
	NNValue Forward(NNValue x);
    NNValue Backward(NNValue x);
    }
}
