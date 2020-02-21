using System.Collections.Generic;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
    public interface INetwork 
    {
        List<ILayer> Layers { get; set; }
        NNValue Activate(NNValue input, Graph g);
        void ResetState();
        List<NNValue> GetParameters();
    }
}
