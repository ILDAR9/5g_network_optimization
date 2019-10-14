### Interval Graph Coloring related to Optical Network Planning

My approach to solve the contest of ["Управление Информация Оптимизация"](https://ssopt.org).

The solution shared the 1st place with another solution.

### Task description
Дан неориентированный граф и набор путей на нём. Для каждого пути задана его целочисленная ширина BW.
Каждый путь надо раскрасить в BW подряд идущих цветов из множества {0,1,2,...,319} так, чтобы любые два пути,
имеющие общее ребро, не имели общих цветов. Цель – надо максимизировать число раскрашенных путей.
- Вершин <200
- Рёбра <500
- Пути <1000

Решение должно занимать <5 секунд

**Дано**:
- nodesinfo.csv -> список сетевых узлов в 5G сети
- links.csv -> список установленных каналов связи между узлами сети
- newrouting.csv -> _Путь_, его _составляющие Ребра_ и 
_Ширина пути_, указанного в первом столбце. Для этого столбца верно следующее правило: для каждого пути его ширина не меняется вдоль пути.
 Эта ширина и есть количество **цветов**, в которые надо раскрасить путь. Заголовок: number_of_slices.

**Результат**: coloring.csv -> ID пути и его минимальное кол-во цветов для раскраски.

### Более простое объяснение своим языком
Дан неориентированный граф, на нем пути.
Пути хотим закрасить абстрактными цветами то есть 'интервалы цветов при заданом bw' (bw ширина канала):
 а точнее путь должен быть раскрашен в цвета а, а+1, ..., а+bw-1 для некоторого а.

Условия:
1) если пути имеют общее ребро, то между этими двумя путями не должно быть ни одного общего цвета.

2) если не получается какой либо из путей "расскрасить", то 'минусуем' его чтобы не мешал.

В общем нужно придумать стратегию перебора **Graph Interval Coloring**, так чтобы максимизировать кол-во закрашенных путей.

По сути получается что можно забить на файлы links.csv и nodes_info.csv eсли не использовать топологические свойства исходной сети,
для решения нужен только файл с путями.

НО Использование топологических свойств - интересный путь, чтобы придумать быструю стратегию runtime которой уместится в <5секунд.