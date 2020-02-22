using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpML.Activations;
using SharpML.DataStructs.DataSets;
using SharpML.Models;
using SharpML.Networks;
using SharpML.Networks.Base;
using SharpML.Networks.ConvDeconv;
using SharpML.Loss;
using SharpML.Trainer;

namespace CnnTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Random random = new Random(13);
            CNN cNN = new CNN();
            NNValue nValue = NNValue.Random(10, 10, 2, random);
            GraphCPU graph = new GraphCPU(false);
            NNValue outp = new NNValue(new double[]{ -1});
            DataSetNoReccurent data = new DataSetNoReccurent(new NNValue[] { nValue }, new NNValue[] { outp }, new LossSumOfSquares());
            TrainerCPU trainer = new TrainerCPU();
            trainer.Train(120000, 0.002, cNN, data, 2, 0.0001);
            double[] dbs = cNN.Activate(nValue, graph).DataInTensor;
        }
    }

    public class CNN : INetwork
    {
        public List<ILayer> Layers { get; set; }
        double std = 0.01;
        Random random = new Random(12);

        public CNN()
        {
            Layers = new List<ILayer>();

            


            Layers.Add(new ConvolutionLayer(1, new FilterStruct()
            {
                FilterH = 3,
                FilterW = 3,
                FilterCount = 10
            }, new RectifiedLinearUnit(0.01), std, random
            ));
            Layers.Add(new ConvolutionLayer(10, new FilterStruct()
            {
                FilterH = 3,
                FilterW = 3,
                FilterCount = 10
            }, new RectifiedLinearUnit(0.01), std, random
            ));

            Layers.Add(new ConvolutionLayer(10, new FilterStruct()
            {
                FilterH = 5,
                FilterW = 5,
                FilterCount = 5
            }, new RectifiedLinearUnit(0.01), std, random
            ));

            Layers.Add(new MaxPooling());

            Layers.Add(new Flatten());
            Layers.Add(new FeedForwardLayer(5, 1, new LinearUnit(), std, random));

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
