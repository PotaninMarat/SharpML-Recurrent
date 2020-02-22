using SharpML.Models;
using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.Networks.ConvDeconv
{
    public class MaxPooling : ILayer
    {

        int _h, _w;

        public MaxPooling(int h =2, int w =2)
        {
            _h = h;
            _w = w;
        }

        public NNValue Activate(NNValue input, IGraph g)
        {
            NNValue res = g.MaxPooling(input, _h, _w);
            return res;
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
