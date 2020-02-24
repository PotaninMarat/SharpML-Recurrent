using System;
using System.Collections.Generic;
using System.Linq;
using SharpML.Activations;
using SharpML.Loss;
using SharpML.Trainer;
using SharpML.Models;
using SharpML.Util;
using SharpML.DataStructs;
using SharpML.Networks.Base;

namespace RNN_Seq2Seq
{
    class Program
    {
        static void Main(string[] args)
        {
            Random random = new Random(10);

            List<int[]> inp = new List<int[]>();
            List<int[]> outp = new List<int[]>();

            for (int i = 0; i < 1000; i++)
            {

                int len = random.Next(2, 6);

                int[] inpData = new int[len];


                for (int j = 0; j < len; j++)
                {
                   inpData[j] = random.Next(10);
                }

                inp.Add(inpData);
                outp.Add(inpData.Reverse().ToArray());
            }



            Seq2Seq seq2Seq = new Seq2Seq();
            seq2Seq.Train(inp, outp);

            Console.Write("\n\n Тестирование seq2seq на базе LSTM, нейросеть переворачивает число \n\n");

            while (true)
            {
                try
                {
                    int[] inpSeq = GetArr(Console.ReadLine());
                    seq2Seq.Forward(inpSeq);
                    Console.WriteLine("---------");
                }
                catch { }
            }
        }

        static int[] GetArr(string str)
        {
            int[] ints = new int[str.Length];

            for (int i = 0; i < str.Length; i++)
            {
                ints[i] = Convert.ToInt32(new string(new char[] { str[i]}));
            }

            return ints;
        }
    }

    public class Seq2Seq
    {
        INetwork network;
        Random random = new Random(10);
        TrainerCPU trainer = new TrainerCPU();


        public Seq2Seq()
        {
            network = NetworkBuilder.MakeLstm(10, 20, 2, 10, new SoftmaxUnit(), 0.001, random);
            network.ResetState();
        }

        public void Train(List<int[]> inp, List<int[]> outp)
        {
            trainer.Regularization = 1e-7;
            trainer.Train(30, 0.002, network, new DataSetSeq2Seq(inp, outp), 2, 0.0001);
            Console.WriteLine();
            Console.WriteLine();
        }

        public void Forward(int[] inp)
        {
            GraphCPU graph = new GraphCPU(false);

            int indOld = -1;

            network.ResetState();
            for (int i = 0; i < inp.Length; i++)
            {
                NNValue valueMatrix = new NNValue(DataSetSeq2Seq.GetValue(inp[i]));
                network.Activate(valueMatrix, graph);
            }


            for (int i = 0; i < inp.Length; i++)
            {
                NNValue valueMatrix = new NNValue(DataSetSeq2Seq.GetValue(indOld));
                Console.Write(GetInd(network.Activate(valueMatrix, graph)));
            }

            Console.WriteLine();
        }


        int GetInd(NNValue nNValue)
        {
            double[] data = nNValue.DataInTensor;
            double dataMax = data.Max();

            for (int i = 0; i < data.Length; i++)
            {
                if(data[i] == dataMax)
                {
                    return i;
                }
            }
            return -1;
        }
    }

    public class DataSetSeq2Seq : DataSet
    {
        public DataSetSeq2Seq(List<int[]> dataInp, List<int[]> dataOutp)
        {
            InputShape = new Shape(10);
            OutputShape = new Shape(10);
            LossFunction = new CrossEntropyWithSoftmax();

            var r = GetDataSequences(dataInp, dataOutp);

            Training = r;
            Validation = r;
            Testing = r;
        }

        public static List<DataSequence> GetDataSequences(List<int[]> dataInp, List<int[]> dataOutp)
        {
            List<DataSequence> lds = new List<DataSequence>(); 

            for (int i = 0; i < dataInp.Count; i++)
            {
                DataSequence sequence = new DataSequence();
                sequence.Steps = new List<DataStep>();

                for (int j = 0; j < dataInp[i].Length; j++)
                {
                    DataStep ds = new DataStep(GetValue(dataInp[i][j]));
                    sequence.Steps.Add(ds);
                }


                for (int j = 0; j < dataOutp[i].Length; j++)
                {
                    DataStep ds = new DataStep(GetValue(-1), GetValue(dataOutp[i][j]));
                    sequence.Steps.Add(ds);
                }

                lds.Add(sequence);
            }

            return lds;
        }



        public static double[] GetValue(int ind)
        {
            if (ind < 0)
            {
                double[] dbs = new double[10];
                return dbs;
            }
            else
            {
                double[] dbs = new double[10];
                dbs[ind] = 1;
                return dbs;
            }
        }
    }
}
