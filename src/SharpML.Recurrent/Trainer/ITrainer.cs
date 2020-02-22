using SharpML.DataStructs;
using SharpML.Loss;
using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.Trainer
{
    public interface ITrainer
    {
        

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
        List<double>[] Train(int trainingEpochs, double learningRate, INetwork network, DataSet data, int reportEveryNthEpoch, double minLoss);

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
        List<double>[] TrainWithoutConsole(int trainingEpochs, double learningRate, INetwork network, DataSet data, int reportEveryNthEpoch, double minLoss);

        double Pass(double learningRate, INetwork network, List<DataSequence> sequences,
            bool applyTraining, ILoss lossTraining);

        void UpdateModelParams(INetwork network, double stepSize);
        


    }
}
