# ML from scratch

Классические алгоритмы машинного обучения, реализованные с нуля на numpy, без готовых моделей из scikit-learn. Цель — разобраться, как модели устроены внутри: функции потерь, градиентный спуск, обновление весов.

## Проекты

| Папка | Модель | Задача |
|---|---|---|
| [linear-regression](linear-regression/) | Линейная регрессия одной переменной | Прогноз прибыли автосервиса по числу автомобилей в городе |
| [linear-regression-multi](linear-regression-multi/) | Линейная регрессия нескольких переменных | Градиентный спуск и нормальное уравнение |
| [perceptron-adaline](perceptron-adaline/) | Персептрон и Adaline | Бинарная классификация |
| [perceptron-hebb](perceptron-hebb/) | Персептрон, правило Хебба | Разделение двух множеств точек на плоскости |
| [logistic-regression](logistic-regression/) | Логистическая регрессия | Вероятность неисправности двигателя, нелинейный признак |
| [logistic-regression-regularization](logistic-regression-regularization/) | Логистическая регрессия с регуляризацией | Полиномиальные признаки, L1, L2 и Elastic Net для классификации микрочипов |

Запуск: `pip install -r requirements.txt`, затем `jupyter notebook`.
