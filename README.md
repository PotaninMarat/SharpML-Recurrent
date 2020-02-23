# SharpML-Recurrent
Реплика проекта Андрея Корпатого "RecurrentJs" и Thomas Lahore's "RecurrentJava" на языке C#.

* [Проект на Java: (MIT)](https://github.com/evolvingstuff/RecurrentJava)

* [Оригинальный проект на C#: (MIT)](https://github.com/andrewfry/SharpML-Recurrent)

* [Дневник внесенных изменений](https://github.com/zaharPonimash/SharpML-Recurrent/blob/master/%D0%98%D0%B7%D0%BC%D0%B5%D0%BD%D0%B5%D0%BD%D0%B8%D1%8F.md)

### Слои:

* RnnLayer (Простой рекуррентный слой)
* LstmLayer
* RnnLayer
* FeedForwardLayer
* ConvolutionLayer
* MaxPooling
* Flatten
* ReShape

### Функции активации:

* AbsUnit
* ArcTanUnit
* EliotSigUnit
* GaussianRbfUnit
* LinearUnit
* RectifiedLinearUnit
* SigmoidUnit
* SineUnit
* SoftmaxUnit
* SqnlUnit
* SQRBFUnit
* TanhUnit


### Лицензия
### MIT

### Дальнейшие планы
* Перенести код из этой библиотеки в свой проект AIFramework 3.0 для более удобного использования
* Написать регуляризацию L1
* Добавить различные оптимизаторы Adam, Adadelta, Adagrad, Nesterov, SGD.
* Сделать минипакетное обучение
* Добавить same в сверточный слой и написать unpooling
* Написать более высокоуровневые реализации популярных архитектур нейронных сетей
* Создать реализации IGraph и ITrainer для GPU, на базе Cuda и OpenCL

### Благодарности за помощь в реализации
* Выражаю благодаронсть [Марату](https://github.com/PotaninMarat) за помощь в реализации прямого и обратного прохода в свертке тензоров и пуллинге
