﻿using System;
using System.Collections.Generic;
using SharpML.Networks.Base;
using SharpML.Models;

namespace SharpML.Networks
{
    [Serializable]
    public class NeuralNetwork : INetwork
    {

        public List<ILayer> Layers { get; set; }

        public NeuralNetwork(List<ILayer> layers)
        {
            Layers = layers;
        }

        public NNValue Activate(NNValue input, IGraph g)
        {
            NNValue prev = input;
            foreach (ILayer layer in Layers)
            {
                prev = layer.Activate(prev, g);
            }
            return prev;
        }

        public void ResetState()
        {
            foreach (ILayer layer in Layers)
            {
                layer.ResetState();
            }
        }

        public List<NNValue> GetParameters()
        {
            List<NNValue> result = new List<NNValue>();
            foreach (ILayer layer in Layers)
            {
                result.AddRange(layer.GetParameters());
            }
            return result;
        }
    }
}
