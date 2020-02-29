using System;
using System.Collections.Generic;
using SharpML.Networks.Base;
using SharpML.Models;
using SharpML.DataStructs;

namespace SharpML.Networks
{
    [Serializable]
    public class NeuralNetwork : INetwork
    {

        public List<ILayer> Layers { get; set; }
        public Shape InputShape { get; set; }
        public Shape OutputShape { get; private set; }

        public int TrainableParameters
        {
            get
            {
                var trpar = 0;

                for (int i = 0; i < Layers.Count; i++)
                {
                    trpar += Layers[i].TrainableParameters;
                }

                return trpar;
            }
        }

        Random random;
        double std;

        public NeuralNetwork(Random rand, double stdParams)
        {
            random = rand;
            std = stdParams;
            Layers = new List<ILayer>();
        }

        public NeuralNetwork(List<ILayer> layers)
        {
            Layers = layers;
            InputShape = layers[0].InputShape;
            OutputShape = layers[layers.Count - 1].OutputShape;
        }

        /// <summary>
        /// Добавление НОВОГО слоя в НС
        /// </summary>
        /// <param name="layer">Слой</param>
        public void AddNewLayer(ILayer layer)
        {
            OutputShape = layer.OutputShape; 
            var shape = Layers[Layers.Count - 1].OutputShape;
            layer.Generate(shape, random, std);
            Layers.Add(layer);

            if (Layers.Count == 1)
                InputShape = Layers[0].InputShape;
        }

        /// <summary>
        /// Добавление НОВОГО слоя в НС
        /// </summary>
        /// <param name="inpShape">Размерность тензора входа</param>
        /// <param name="layer">Слой</param>
        public void AddNewLayer(Shape inpShape, ILayer layer)
        {

            OutputShape = layer.OutputShape;
            layer.Generate(inpShape, random, std);
            Layers.Add(layer);

            if (Layers.Count == 1)
                InputShape = inpShape;
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

        /// <summary>
        /// Описание слоя
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string layersInfo = "";

            for (int i = 0; i < Layers.Count; i++)
            {
                layersInfo += Layers[i].ToString() + "\n";
            }

            layersInfo += string.Format("\n\ninp: {0} | outp: {1} | trainable parameters: {2}", InputShape, OutputShape, TrainableParameters);

            return layersInfo; 
        }
    }
}
