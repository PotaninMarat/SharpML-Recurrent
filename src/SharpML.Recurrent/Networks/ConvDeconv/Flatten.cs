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
        public NNValue Activate(NNValue input, IGraph g)
        {
            NNValue outp = input.Clone();
            int len = outp.Len;
            outp.H = len;
            outp.W = 1;
            outp.D = 1;
            return outp;
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
