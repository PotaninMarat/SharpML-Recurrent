using SharpML.Activations;
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

        
        public ConvolutionLayer(int d, FilterStruct filterStruct, INonlinearity func, double initParamsStdDev, Random rnd)
        {
            Function = func;
            Filters = new NNValue[filterStruct.FilterCount];

            for (int i = 0; i < filterStruct.FilterCount; i++)
            {
                Filters[i] = NNValue.Random(filterStruct.FilterH, filterStruct.FilterW, d, initParamsStdDev, rnd);
            }

            //Bias = new NNValue(filterStruct.FilterCount);
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
