using System;

namespace SharpML.Models
{
    public interface IRunnable
    {
        Action Run { get; set; }
    }
}
