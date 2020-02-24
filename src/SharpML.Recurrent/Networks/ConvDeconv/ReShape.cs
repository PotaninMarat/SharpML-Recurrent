using SharpML.DataStructs;
using SharpML.Models;
using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.Networks.ConvDeconv
{
    public class ReShape : ILayer
    {

        /// <summary>
        /// Входная размерность
        /// </summary>
        public Shape InputShape { get; set; }
        /// <summary>
        /// Выходная размерность
        /// </summary>
        public Shape OutputShape { get; private set; }
        float _gain = 1.0f;



        public ReShape(Shape inputShape, Shape newShape)
        {
            InputShape = inputShape;
            OutputShape = newShape;
        }

        public ReShape(Shape inputShape, Shape newShape, float gain = 1.0f)
        {
            InputShape = inputShape;
            OutputShape = newShape;
            _gain = gain;
        }

        public ReShape(Shape newShape)
        {
            OutputShape = newShape;
        }

        public NNValue Activate(NNValue input, IGraph g)
        {
            return g.ReShape(input, OutputShape, _gain);
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
        }
    }
}
