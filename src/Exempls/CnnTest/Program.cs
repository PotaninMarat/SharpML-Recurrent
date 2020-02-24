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
using SharpML.DataStructs;
using SharpML.Networks.Recurrent;

namespace CnnTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Random random = new Random(13);

            NeuralNetwork cNN = new NeuralNetwork(random, 0.1);

            cNN.AddNewLayer(new Shape(28, 28), new ConvolutionLayer(new RectifiedLinearUnit(0.01), 8, 3, 3));
            cNN.AddNewLayer(new MaxPooling(2, 2));

            cNN.AddNewLayer(new ConvolutionLayer(new RectifiedLinearUnit(0.01), 16, 3, 3));
            cNN.AddNewLayer(new MaxPooling(2, 2));

            cNN.AddNewLayer(new ConvolutionLayer(new RectifiedLinearUnit(0.01), 32, 3, 3));
            cNN.AddNewLayer(new MaxPooling(2, 2));

            cNN.AddNewLayer(new Flatten());
            cNN.AddNewLayer(new LstmLayer(10));
            cNN.AddNewLayer(new FeedForwardLayer(2, new SoftmaxUnit()));


            GraphCPU graph = new GraphCPU(false);



            NNValue nValue = NNValue.Random(28, 28, 2, random);
            NNValue nValue1 = NNValue.Random(28, 28, 2, random);
            NNValue outp = new NNValue(new double[]{ 0,1});
            NNValue outp1 = new NNValue(new double[]{ 1,0});



            DataSetNoReccurent data = new DataSetNoReccurent(new NNValue[] { nValue, nValue1 }, new NNValue[] { outp, outp1 }, new CrossEntropyWithSoftmax());



            TrainerCPU trainer = new TrainerCPU();
            trainer.Train(10000, 0.001, cNN, data, 2, 0.0001);
            double[] dbs = cNN.Activate(nValue, graph).DataInTensor;
            double[] dbs1 = cNN.Activate(nValue1, graph).DataInTensor;
        }
    }
}
