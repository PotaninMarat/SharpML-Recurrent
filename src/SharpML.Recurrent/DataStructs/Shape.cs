using SharpML.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.DataStructs
{
    /// <summary>
    /// Форма тензора
    /// </summary>
    public class Shape
    {
        public int H { get; set; }
        public int W { get; set; }
        public int D { get; set; }
        
        public int Len
        {
            get
            {
                return H * W * D;
            }
        }

        public Shape()
        {
            H = 1;
            W = 1;
            D = 1;
        }

        public Shape(int h)
        {
            H = h;
            W = 1;
            D = 1;
        }

        public Shape(int h, int w)
        {
            H = h;
            W = w;
            D = 1;
        }

        public Shape(int h, int w, int d)
        {
            H = h;
            W = w;
            D = d;
        }


        public static NNValue ReShape(NNValue value, Shape newShape)
        {
            if(value.Len != newShape.Len)
            {
                throw new Exception("Преобразование невозможно, не совпадают объемы");
            }

            NNValue valRes = value.Clone();

            valRes.H = newShape.H;
            valRes.W = newShape.W;
            valRes.D = newShape.D;

            return valRes;
        }

        public override string ToString()
        {
            return string.Format("[H:{0}, W:{1}, D:{2}]", H, W, D);
        }

    }
}
