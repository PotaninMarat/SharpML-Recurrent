using System;
using System.Collections.Generic;
using SharpML.DataStructs;
using SharpML.Models;

namespace SharpML.Networks.Base
{
    public interface ILayer 
    {
        Shape InputShape { get; set; }
        Shape OutputShape { get;}


        NNValue Activate(NNValue input, IGraph g);
        void ResetState();
        List<NNValue> GetParameters();
        void Generate(Shape inpShape, Random random, double std);
    }
}
