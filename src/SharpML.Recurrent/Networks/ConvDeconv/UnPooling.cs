﻿using SharpML.DataStructs;
using SharpML.Models;
using SharpML.Networks.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpML.Networks.ConvDeconv
{
    public class UnPooling : ILayer
    {
        public Shape InputShape { get; set; }
        public Shape OutputShape { get; private set; }
        int _h, _w;
        public int TrainableParameters => 0;


        public UnPooling(Shape inputShape, int h = 2, int w = 2)
        {
            OutputShape = new Shape(inputShape.H*h, inputShape.W*w, inputShape.D);
            InputShape = inputShape;
            _h = h;
            _w = w;
        }

        public UnPooling(int h = 2, int w = 2)
        {
            _h = h;
            _w = w;
        }


        public NNValue Activate(NNValue input, IGraph g)
        {
            NNValue res = g.UnPooling(input, _h, _w);
            return res;
        }

        public List<NNValue> GetParameters()
        {
            return new List<NNValue>();
        }

        public void ResetState()
        {

        }

        public void Generate(Shape inpShape, Random random, double std)
        {
            Init(inpShape, _h, _w);
        }

        void Init(Shape inputShape, int h = 2, int w = 2)
        {
            OutputShape = new Shape(inputShape.H*h, inputShape.W* w, inputShape.D);
            InputShape = inputShape;
            _h = h;
            _w = w;
        }

        /// <summary>
        /// Описание слоя
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return string.Format("UnPooling\t\t|inp: {0} |outp: {1}|TrainParams: {2}", InputShape, OutputShape, TrainableParameters);
        }
    }
}
