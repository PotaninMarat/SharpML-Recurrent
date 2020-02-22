using System;

namespace SharpML.Models
{
    public class Runnable : IRunnable
    {
        public Action Run {get;set;}
    }
}
