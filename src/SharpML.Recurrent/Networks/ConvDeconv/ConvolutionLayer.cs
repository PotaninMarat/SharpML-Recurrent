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
        //public NNValue Bias { get; private set;}
        public NNValue[] Filters { get; private set;}
        public INonlinearity Function { get; set; }
        public Shape InputShape { get; set; }
        public Shape OutputShape { get; private set; }
        public FilterStruct fs;
        public int StrideX = 1, StrideY = 1;

        /// <summary>
        /// Pad по Y
        /// </summary>
        public int PaddingY {
            get
            {
                return _padY;
            }
            set
            {
                _padY = (value > Filters[0].H - 1) ? Filters[0].H - 1 : value;
            }
        }

        /// <summary>
        /// Pad по X
        /// </summary>
        public int PaddingX {
            get
            {
                return _padX;
            }
            set
            {
                _padX = (value>Filters[0].W-1)? Filters[0].W - 1:value;
            }
        }

        /// <summary>
        /// Сохраняется ли размерность входа
        /// </summary>
        public bool IsSame
        {
            get
            {
                return (_padY == Filters[0].H - 1) && (_padX == Filters[0].W - 1);
            }
            set
            {
                if (value)
                {
                    _padY = Filters[0].H - 1;
                    _padX = Filters[0].W - 1;
                }
                else
                {
                    _padX = 0; _padY = 0;
                }
            }
        }

        int _padY = 0;
        int _padX = 0;

        /// <summary>
        /// Число обучаемых парамметров
        /// </summary>
        public int TrainableParameters
        {
            get
            {
                return fs.FilterCount * fs.FilterH * fs.FilterW;
            }
        }


        public ConvolutionLayer(Shape inputShape, FilterStruct filterStruct, INonlinearity func, Random rnd)
        {
            

            InputShape = inputShape;
            fs = filterStruct;

            int outputH = inputShape.H - filterStruct.FilterH + 1+_padY,
                outputW = inputShape.W - filterStruct.FilterW + 1+_padX;


            OutputShape = new Shape(outputH, outputW, filterStruct.FilterCount);
            double initParamsStdDev = 1.0 / Math.Sqrt(OutputShape.Len);

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
            Filters = new NNValue[fs.FilterCount];

            for (int i = 0; i < fs.FilterCount; i++)
            {
                Filters[i] = new NNValue(fs.FilterH, fs.FilterW);
            }
        }

        public ConvolutionLayer(INonlinearity func, int count, int h=3, int w=3)
        {
            Filters = new NNValue[count];

            for (int i = 0; i < count; i++)
            {
                Filters[i] = new NNValue(h, w);
            }
        
            Function = func;
            fs = new FilterStruct() { FilterW = w, FilterCount = count, FilterH = h};
        }

        public NNValue Activate(NNValue input, IGraph g)
        {
            NNValue output = g.Convolution(input, Filters, _padX, _padY);
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
            Init(inpShape, fs, Function, random);
        }

        void Init(Shape inputShape, FilterStruct filterStruct, INonlinearity func, Random rnd)
        {

            InputShape = inputShape;
            fs = filterStruct;
            int outputH = inputShape.H - filterStruct.FilterH + 1 + _padY,
               outputW = inputShape.W - filterStruct.FilterW + 1 + _padX;


            OutputShape = new Shape(outputH, outputW, filterStruct.FilterCount);

            double initParamsStdDev =  1.0 / Math.Sqrt(OutputShape.Len);


            int d = InputShape.D;
            Function = func;
            Filters = new NNValue[filterStruct.FilterCount];

            for (int i = 0; i < filterStruct.FilterCount; i++)
            {
                Filters[i] = NNValue.Random(filterStruct.FilterH, filterStruct.FilterW, d, initParamsStdDev, rnd);
            }
        }

       
            /// <summary>
            /// Описание слоя
            /// </summary>
            /// <returns></returns>
        public override string ToString()
        {
            return string.Format("ConvLayer       \t|inp: {0} |outp: {1} |Non lin. activate: {3} |TrainParams: {2}", InputShape, OutputShape, TrainableParameters, Function);
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


   


