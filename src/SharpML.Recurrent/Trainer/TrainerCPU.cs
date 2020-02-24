using SharpML.DataStructs;
using SharpML.Loss;
using SharpML.Models;
using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SharpML.Trainer
{
    public class TrainerCPU : ITrainer
    {

        public double DecayRate = 0.999;
        public double SmoothEpsilon = 1e-8;
        public double GradientClipValue = 5;
        public double Regularization = 0.000001; // L2 regularization strength


        /// <summary>
        /// Обучение сети
        /// </summary>
        /// <typeparam name="T">Тип (сеть)</typeparam>
        /// <param name="trainingEpochs">Число эпох</param>
        /// <param name="learningRate">Скорость обучения</param>
        /// <param name="network">Нейронная сеть</param>
        /// <param name="data">Набор данных(датасет)</param>
        /// <param name="reportEveryNthEpoch">Как часто выводить данные в консоль</param>
        /// <param name="rng">Датчик случайных чисел</param>
        /// <returns>Ошибка</returns>
        public List<double>[] Train(int trainingEpochs, double learningRate, INetwork network, DataSet data, int reportEveryNthEpoch, double minLoss)
        {
            Console.WriteLine("--------------------------------------------------------------");
            List<double>[] result = new List<double>[3];

            for (int i = 0; i < 3; i++)
            {
                result[i] = new List<double>();
            }

            for (int epoch = 0; epoch < trainingEpochs; epoch++)
            {
                double reportedLossTrain = Pass(learningRate, network, data.Training, true, data.LossFunction);
                double reportedLossValidation = 0;
                double reportedLossTesting = 0;
                result[0].Add(reportedLossTrain);

                //if (Double.IsNaN(reportedLossTrain) || Double.IsInfinity(reportedLossTrain))
                //{
                //    throw new Exception("WARNING: invalid value for training loss. Try lowering learning rate.");
                //}

                if (data.Validation != null)
                {
                    reportedLossValidation = Pass(learningRate, network, data.Validation, false, data.LossFunction);
                    result[1].Add(reportedLossValidation);
                }

                if (data.Testing != null)
                {
                    reportedLossTesting = Pass(learningRate, network, data.Testing, false, data.LossFunction);
                    result[2].Add(reportedLossTesting);
                }

                if (epoch % reportEveryNthEpoch == 0)
                {
                    String show = "Эпоха[" + (epoch + 1) + "/" + trainingEpochs + "]";
                    show += "\tОшибка обучения = " + String.Format("{0:N5}", reportedLossTrain);
                    if (data.Validation != null)
                    {
                        show += "\tОшибка валидации = " + String.Format("{0:N5}", reportedLossValidation);
                    }
                    if (data.Testing != null)
                    {
                        show += "\tОшибка тестирования  = " + String.Format("{0:N5}", reportedLossTesting);
                    }
                    Console.WriteLine(show);
                }

                if (reportedLossTrain < minLoss && reportedLossValidation < minLoss)
                {
                    Console.WriteLine("--------------------------------------------------------------");
                    Console.WriteLine("\nОбучение завершено.");
                    break;
                }
            }

            return result;
        }

        /// <summary>
        /// Обучение сети
        /// </summary>
        /// <typeparam name="T">Тип (сеть)</typeparam>
        /// <param name="trainingEpochs">Число эпох</param>
        /// <param name="learningRate">Скорость обучения</param>
        /// <param name="network">Нейронная сеть</param>
        /// <param name="data">Набор данных(датасет)</param>
        /// <param name="reportEveryNthEpoch">Как часто выводить данные в консоль</param>
        /// <param name="rng">Датчик случайных чисел</param>
        /// <returns>Ошибка</returns>
        public List<double>[] TrainWithoutConsole(int trainingEpochs, double learningRate, INetwork network, DataSet data, int reportEveryNthEpoch, double minLoss)
        {
            List<double>[] result = new List<double>[3];

            for (int i = 0; i < 3; i++)
            {
                result[i] = new List<double>();
            }

            for (int epoch = 0; epoch < trainingEpochs; epoch++)
            {
                double reportedLossTrain = Pass(learningRate, network, data.Training, true, data.LossFunction);
                double reportedLossValidation = 0;
                double reportedLossTesting = 0;
                result[0].Add(reportedLossTrain);

                reportedLossValidation = Pass(learningRate, network, data.Validation, false, data.LossFunction);
                result[1].Add(reportedLossValidation);

                reportedLossTesting = Pass(learningRate, network, data.Testing, false, data.LossFunction);
                result[2].Add(reportedLossTesting);


                if (reportedLossTrain < minLoss && reportedLossValidation < minLoss)
                {
                    break;
                }
            }

            return result;
        }



        public double Pass(double learningRate, INetwork network, List<DataSequence> sequences,
            bool applyTraining, ILoss lossTraining)
        {
            double numerLoss = 0;
            double denomLoss = 0;

            foreach (DataSequence seq in sequences)
            {
                network.ResetState();
                GraphCPU g = new GraphCPU(applyTraining);
                foreach (DataStep step in seq.Steps)
                {
                    NNValue output = network.Activate(step.Input, g);
                    if (step.TargetOutput != null)
                    {
                        double loss = lossTraining.Measure(output, step.TargetOutput);
                        if (Double.IsNaN(loss) || Double.IsInfinity(loss))
                        {
                            return loss;
                        }
                        numerLoss += loss;
                        denomLoss++;
                        if (applyTraining)
                        {
                            lossTraining.Backward(output, step.TargetOutput);
                        }
                    }
                }
                List<DataSequence> thisSequence = new List<DataSequence>();
                thisSequence.Add(seq);
                if (applyTraining)
                {
                    g.Backward(); //backprop dw values
                    UpdateModelParams(network, learningRate); //update params
                }
            }
            return numerLoss / denomLoss;
        }


        public void UpdateModelParams(INetwork network, double stepSize)
        {
            
                Parallel.ForEach(network.GetParameters(), new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount} , m =>
                {
                    for (int i = 0; i < m.DataInTensor.Length; i++)
                    {

                    // rmsprop adaptive learning rate
                    double mdwi = m.DifData[i];
                        m.StepCache[i] = m.StepCache[i] * DecayRate + (1 - DecayRate) * mdwi * mdwi;

                    // gradient clip
                    if (mdwi > GradientClipValue)
                        {
                            mdwi = GradientClipValue;
                        }
                        if (mdwi < -GradientClipValue)
                        {
                            mdwi = -GradientClipValue;
                        }

                    // update (and regularize)
                    m.DataInTensor[i] -= stepSize * mdwi / Math.Sqrt(m.StepCache[i] + SmoothEpsilon) + Regularization * m.DataInTensor[i];
                        m.DifData[i] = 0;
                    }
                });
        }
    }
}
