using SharpML.DataStructs;
using SharpML.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.Networks.ConvDeconv
{
    public class ReShape
    {
        float _gain = 1.0f;
        Shape shape;

        public ReShape() { }

        public ReShape(Shape newShape, float gain = 1.0f)
        {
            shape = newShape;
            _gain = gain;
        }

        public NNValue Activate(NNValue input, IGraph g)
        {
            return g.ReShape(input, shape, _gain);
        }

        public List<NNValue> GetParameters()
        {
            return new List<NNValue>();
        }

        public void ResetState()
        {

        }
    }
}
