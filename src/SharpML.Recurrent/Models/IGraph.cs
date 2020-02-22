using SharpML.Activations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.Models
{
    public interface IGraph
    {
        bool ApplyBackprop { get; set; }

        List<IRunnable> Backprop { get; set; }

        void Backward();

        NNValue ConcatVectors(NNValue m1, NNValue m2);

        NNValue Nonlin(INonlinearity neuron, NNValue m);

        NNValue Mul(NNValue m1, NNValue m2);

        NNValue Add(NNValue m1, NNValue m2);

        NNValue OneMinus(NNValue m);

        NNValue Subtract(NNValue m1, NNValue m2);

        NNValue smul(NNValue m, double s);

        NNValue smul(double s, NNValue m);

        NNValue Neg(NNValue m);

        NNValue Elmul(NNValue m1, NNValue m2);

        /// <summary>
        /// Свертка
        /// </summary>
        /// <param name="input">Тензор входа</param>
        /// <param name="filters">Фильтры</param>
        /// <param name="isSame"></param>
        NNValue Convolution(NNValue input, NNValue[] filters, bool isSame);

        NNValue MaxPooling(NNValue inp, int h, int w);
    }
}
