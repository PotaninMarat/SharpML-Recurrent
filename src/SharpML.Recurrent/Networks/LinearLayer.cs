using System;
using System.Collections.Generic;
using SharpML.DataStructs;
using SharpML.Models;
using SharpML.Networks.Base;

namespace SharpML.Networks
{
     [Serializable]
    public class LinearLayer : ILayer
    {
         
        NNValue _w;
        /// <summary>
        /// Входная размерность
        /// </summary>
        public Shape InputShape { get; set; }
        /// <summary>
        /// Выходная размерность
        /// </summary>
        public Shape OutputShape { get; private set; }


        //no biases

        public LinearLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rng)
        {
            InputShape = new Shape(inputDimension);
            OutputShape = new Shape(outputDimension);
            _w = NNValue.Random(outputDimension, inputDimension, initParamsStdDev, rng);
        }

        public LinearLayer(Shape inputShape, int outputDimension, double initParamsStdDev, Random rng)
        {
            InputShape = inputShape;
            OutputShape = new Shape(outputDimension);
            _w = NNValue.Random(outputDimension, inputShape.H, initParamsStdDev, rng);
        }

        public LinearLayer(int outputDimension)
        {
            OutputShape = new Shape(outputDimension);
        }


        public NNValue Activate(NNValue input, IGraph g)
        {
            NNValue returnObj = g.Mul(_w, input);
            return returnObj;
        }

        public void ResetState()
        {

        }

        public List<NNValue> GetParameters()
        {
            List<NNValue> result = new List<NNValue>();
            result.Add(_w);
            return result;
        }

        /// <summary>
        /// Генерация слоя НС
        /// </summary>
        /// <param name="inpShape"></param>
        /// <param name="random"></param>
        /// <param name="std"></param>
        /// <returns></returns>
        public void Generate(Shape inpShape, Random random, double std)
        {
            Init(inpShape, OutputShape.H, std, random);
        }

        void Init(Shape inputShape, int outputDimension, double initParamsStdDev, Random rng)
        {
            InputShape = inputShape;
            OutputShape = new Shape(outputDimension);
            _w = NNValue.Random(outputDimension, inputShape.H, initParamsStdDev, rng);
        }
    }
}
