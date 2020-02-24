using SharpML.Activations;
using SharpML.DataStructs;
using SharpML.Models;
using SharpML.Networks.Base;
using System;
using System.Collections.Generic;

namespace SharpML.Networks.ConvDeconv
{
    public class ConvolutionLayer : ILayer
    {
        public bool IsSame { get; set; }
        //public NNValue Bias { get; private set;}
        public NNValue[] Filters { get; private set;}
        public INonlinearity Function { get; set; }
        public Shape InputShape { get; set; }
        public Shape OutputShape { get; private set; }
        public FilterStruct fs;

        public ConvolutionLayer(Shape inputShape, FilterStruct filterStruct, INonlinearity func, double initParamsStdDev, Random rnd)
        {
            InputShape = inputShape;
            fs = filterStruct;
            OutputShape = new Shape(inputShape.H-filterStruct.FilterH+1, inputShape.W-filterStruct.FilterW+1, filterStruct.FilterCount);

            int d = InputShape.D;
            Function = func;
            Filters = new NNValue[filterStruct.FilterCount];

            for (int i = 0; i < filterStruct.FilterCount; i++)
            {
                Filters[i] = NNValue.Random(filterStruct.FilterH, filterStruct.FilterW, d, initParamsStdDev, rnd);
            }

            //Bias = new NNValue(filterStruct.FilterCount);
        }

        public ConvolutionLayer(FilterStruct filterStruct, INonlinearity func)
        {
            Function = func;
            fs = filterStruct;
        }

        public ConvolutionLayer(INonlinearity func, int count, int h=3, int w=3)
        {
            Function = func;
            fs = new FilterStruct() { FilterW = w, FilterCount = count, FilterH = h};
        }

        public NNValue Activate(NNValue input, IGraph g)
        {
            NNValue output = g.Convolution(input, Filters, IsSame);
            output = g.Nonlin(Function, output);
            return output;
        }

        public List<NNValue> GetParameters()
        {
            List<NNValue> values = new List<NNValue>();
            values.AddRange(Filters);
           // values.Add(Bias);
            return values;
        }

        public void ResetState()
        {
            
        }

        /// <summary>
        /// Генерация слоя НС
        /// </summary>
        /// <returns></returns>
        public void Generate(Shape inpShape, Random random, double std)
        {
            Init(inpShape, fs, Function, std, random);
        }

        void Init(Shape inputShape, FilterStruct filterStruct, INonlinearity func, double initParamsStdDev, Random rnd)
        {
            InputShape = inputShape;
            fs = filterStruct;
            OutputShape = new Shape(inputShape.H - filterStruct.FilterH + 1, inputShape.W - filterStruct.FilterW + 1, filterStruct.FilterCount);

            int d = InputShape.D;
            Function = func;
            Filters = new NNValue[filterStruct.FilterCount];

            for (int i = 0; i < filterStruct.FilterCount; i++)
            {
                Filters[i] = NNValue.Random(filterStruct.FilterH, filterStruct.FilterW, d, initParamsStdDev, rnd);
            }
        }
    }

    /// <summary>
    /// Структура фильтра
    /// </summary>
    public class FilterStruct
    {
        public int FilterW { get; set; }
        public int FilterH { get; set; }
        public int FilterCount { get; set; }
    }

}
