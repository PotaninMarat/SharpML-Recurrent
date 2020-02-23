﻿using SharpML.DataStructs;
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
        float _gain = 1.0f;

        public Flatten() { }

        public Flatten(float gain)
        {
            _gain = gain;
        }

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
    }
}