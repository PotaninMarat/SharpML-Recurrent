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

        public double GradientClipValue = 5;
        double gradClip;
        public double L2Regularization { get; set; } // L2 regularization strength
        public double L1Regularization { get; set; } // L1 regularization strength
        public int BatchSize { get; set; } // Размер батча
        public TrainType TrainTypeSetting { get; set; }
        public IOptimizer TrainOptimizer { get; set; }

        Random random;
        int batchSize = 2;
        public int RandomSeed { get; set; } 

        public TrainerCPU()
        {
            TrainOptimizer = new Optimizers.RMSProp();
            RandomSeed = 12;
            TrainTypeSetting = TrainType.Online;
            L1Regularization = 0;
            L2Regularization = 0;
        }

        public TrainerCPU(TrainType trainType)
        {
            TrainOptimizer = new Optimizers.RMSProp();
            RandomSeed = 12;
            TrainTypeSetting = trainType;
            BatchSize = batchSize;
            random = new Random(RandomSeed);

            L1Regularization = 0;
            L2Regularization = 0;
        }

        public TrainerCPU(TrainType trainType, IOptimizer optimizer)
        {
            TrainOptimizer = optimizer;
            RandomSeed = 12;
            TrainTypeSetting = trainType;
            BatchSize = batchSize;
            random = new Random(RandomSeed);

            L1Regularization = 0;
            L2Regularization = 0;
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
        public List<double>[] Train(int trainingEpochs, double learningRate, INetwork network, DataSet data, int reportEveryNthEpoch, double minLoss)
        {
            double lr = learningRate;
            List<double>[] result = new List<double>[3];

            for (int i = 0; i < 3; i++)
            {
                result[i] = new List<double>();
            }
            #region Info
            if (TrainTypeSetting == TrainType.Online)
            {
                Console.WriteLine("----------------------- ONLINE MODE ---------------------------");
            }

            if (TrainTypeSetting == TrainType.Offline)
            {
                Console.WriteLine("----------------------- Offline MODE ---------------------------");
            }

            if (TrainTypeSetting == TrainType.MiniBatch)
            {
                Console.WriteLine("----------------------- Mini-Batch MODE ---------------------------");
            }
            #endregion


            for (int epoch = 0; epoch < trainingEpochs; epoch++)
            {
               // TrainOptimizer.Reset();

                double reportedLossTrain = 0;

                if (TrainTypeSetting == TrainType.Online)
                {
                    gradClip = GradientClipValue;
                    reportedLossTrain = Pass(lr, network, data.Training, true, data.LossFunction);
                }
                if (TrainTypeSetting == TrainType.Offline)
                {
                    lr /= data.Training.Count;
                    gradClip = GradientClipValue * data.Training.Count;

                    reportedLossTrain = PassOffline(learningRate, network, data.Training, true, data.LossFunction);
                }
                if (TrainTypeSetting == TrainType.MiniBatch)
                {
                    lr /= BatchSize;
                    gradClip = GradientClipValue * BatchSize;

                    reportedLossTrain = PassBatch(learningRate, network, data.Training, true, data.LossFunction);
                }


                double reportedLossValidation = 0;
                double reportedLossTesting = 0;
                result[0].Add(reportedLossTrain);


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
                double reportedLossTrain = 0;

                if (TrainTypeSetting == TrainType.Online)
                    reportedLossTrain = Pass(learningRate, network, data.Training, true, data.LossFunction);
                if (TrainTypeSetting == TrainType.Offline)
                    reportedLossTrain = PassOffline(learningRate, network, data.Training, true, data.LossFunction);
                if (TrainTypeSetting == TrainType.MiniBatch)
                    reportedLossTrain = PassBatch(learningRate, network, data.Training, true, data.LossFunction);

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


        /// <summary>
        /// Онлайн обучение
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="network"></param>
        /// <param name="sequences"></param>
        /// <param name="applyTraining"></param>
        /// <param name="lossTraining"></param>
        /// <returns></returns>
        public double Pass(double learningRate, INetwork network, List<DataSequence> sequences,
            bool applyTraining, ILoss lossTraining)
        {
            double numerLoss = 0;
            double denomLoss = 0;



            foreach (DataSequence seq in sequences)
            {
                GraphCPU g = new GraphCPU(applyTraining);

                network.ResetState();
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

                if (applyTraining)
                {
                    g.Backward(); //backprop dw values
                    UpdateModelParams(network, learningRate); //update params
                }

            }



            return numerLoss / denomLoss;
        }

        /// <summary>
        /// Оффлайн обучение
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="network"></param>
        /// <param name="sequences"></param>
        /// <param name="applyTraining"></param>
        /// <param name="lossTraining"></param>
        /// <returns></returns>
        public double PassOffline(double learningRate, INetwork network, List<DataSequence> sequences,
            bool applyTraining, ILoss lossTraining)
        {
            double numerLoss = 0;
            double denomLoss = 0;


            GraphCPU g = new GraphCPU(applyTraining);

            foreach (DataSequence seq in sequences)
            {

                network.ResetState();
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



            }

            if (applyTraining)
            {
                g.Backward(); //backprop dw values
                UpdateModelParams(network, learningRate); //update params
            }

            return numerLoss / denomLoss;
        }

        /// <summary>
        /// Один проход для минипакетного обучения
        /// </summary>
        /// <param name="learningRate">Скорость обучения</param>
        /// <param name="network">Нейросеть</param>
        /// <param name="sequences">Датасет</param>
        /// <param name="isTraining">Производится ли обучение</param>
        /// <param name="lossFunction">Функция ошибки</param>
        /// <returns></returns>
        public double PassBatch(double learningRate, INetwork network, List<DataSequence> sequences,
            bool isTraining, ILoss lossFunction)
        {
            double numerLoss = 0;
            double denomLoss = 0;
            int index, passes = (sequences.Count % BatchSize == 0) ? sequences.Count / BatchSize : sequences.Count / BatchSize + 1;

            for (int j = 0; j < passes; j++)
            {

                GraphCPU g = new GraphCPU(isTraining);

                for (int i = 0; i < BatchSize; i++)
                {
                    index = random.Next(sequences.Count);
                    var seq = sequences[index];

                    network.ResetState();
                    foreach (DataStep step in seq.Steps)
                    {
                        NNValue output = network.Activate(step.Input, g);
                        if (step.TargetOutput != null)
                        {
                            double loss = lossFunction.Measure(output, step.TargetOutput);
                            if (Double.IsNaN(loss) || Double.IsInfinity(loss))
                            {
                                return loss;
                            }
                            numerLoss += loss;
                            denomLoss++;
                            if (isTraining)
                            {
                                lossFunction.Backward(output, step.TargetOutput);
                            }
                        }
                    }
                }

                if (isTraining)
                {
                    g.Backward(); //backprop dw values
                    UpdateModelParams(network, learningRate); //update params
                }
            }
            return numerLoss / denomLoss;
        }


        /// <summary>
        /// Обновление весов
        /// </summary>
        /// <param name="network">Сеть</param>
        /// <param name="learningRate">Скорость обучения</param>
        public void UpdateModelParams(INetwork network, double learningRate)
        {
            TrainOptimizer.UpdateModelParams(network, learningRate, gradClip, L1Regularization, L2Regularization);
        }

       
    }

    /// <summary>
    /// Тип обучения
    /// </summary>
    public enum TrainType
    {
        Offline,
        Online,
        MiniBatch
    }
}
