using System.Collections.Generic;
using SharpML.DataStructs;
using SharpML.Models;

namespace SharpML.Networks.Base
{
    public interface INetwork 
    {
        Shape InputShape { get; set; }
        Shape OutputShape { get; }
        int TrainableParameters { get; }
        List<ILayer> Layers { get; set; }
        NNValue Activate(NNValue input, IGraph g);
        void ResetState();
        List<NNValue> GetParameters();
    }
}
