using SharpML.Models;

namespace SharpML.Activations
{
    public interface INonlinearity
    {
	NNValue Forward(NNValue x);
    NNValue Backward(NNValue x);
    }
}
