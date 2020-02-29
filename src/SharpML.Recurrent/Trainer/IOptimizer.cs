using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpML.Trainer
{
    /// <summary>
    /// Оптимизатор
    /// </summary>
    public interface IOptimizer
    {
        /// <summary>
        /// Обновление параметров 
        /// </summary>
        /// <param name="network">Нейросеть</param>
        /// <param name="learningRate">Скорость обучения</param>
        /// <param name="gradClip"></param>
        /// <param name="L1"></param>
        /// <param name="L2"></param>
        void UpdateModelParams(INetwork network, double learningRate, double gradClip, double L1, double L2);


        void Reset();
    }
}
