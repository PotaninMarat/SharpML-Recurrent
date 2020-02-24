using SharpML.DataStructs;
using SharpML.Models;
using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.Networks.ConvDeconv
{
    public class Flatten : ILayer
    {
        public Shape InputShape { get; set; }
        public Shape OutputShape { get; private set; }

        float _gain = 1.0f;

        public Flatten(Shape inputShape)
        {
            InputShape = inputShape;
            OutputShape = new Shape(InputShape.Len);
        }

        public Flatten(Shape inputShape, float gain)
        {
            InputShape = inputShape;
            OutputShape = new Shape(InputShape.Len);
            _gain = gain;
        }

        public Flatten() { }
        

        public NNValue Activate(NNValue input, IGraph g)
        {
            Shape shape = new Shape(input.Len);
            return g.ReShape(input, shape, _gain);

        }

        public List<NNValue> GetParameters()
        {
            return new List<NNValue>();
        }

        public void ResetState()
        {

        }

        public void Generate(Shape inpShape, Random random, double std)
        {
            InputShape = inpShape;
            OutputShape = new Shape(InputShape.Len);
        }
    }
}
