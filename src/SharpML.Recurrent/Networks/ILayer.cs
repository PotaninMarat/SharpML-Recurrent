using System.Collections.Generic;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
    public interface ILayer 
    {
        NNValue Activate(NNValue input, Graph g);
        void ResetState();
        List<NNValue> GetParameters();
    }
}
