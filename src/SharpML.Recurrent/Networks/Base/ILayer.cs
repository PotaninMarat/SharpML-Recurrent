using System.Collections.Generic;
using SharpML.Models;

namespace SharpML.Networks.Base
{
    public interface ILayer 
    {
        NNValue Activate(NNValue input, IGraph g);
        void ResetState();
        List<NNValue> GetParameters();
    }
}
