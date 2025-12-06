

# **Математические основания и алгоритмическая реализация объективных метрик для профессиональной стилометрии и атрибуции текстов**

## **Аннотация**

Разработка профессионального программного комплекса для анализа текстовых данных требует фундаментального пересмотра подходов к квантификации стиля. Традиционные метрики, используемые в поверхностном анализе, часто демонстрируют высокую чувствительность к длине текста, что делает их непригодными для сравнительного анализа корпусов с гетерогенными характеристиками. Данный отчет представляет собой исчерпывающее исследование математических моделей, необходимых для построения системы объективного сравнения авторов. В работе детально рассматриваются четыре ключевых измерения текста: лексическое разнообразие (с упором на инвариантные к масштабу метрики), синтаксическая топология (через анализ деревьев зависимостей и составляющих), дисперсионная динамика (квантификация «взрывности» и равномерности распределения) и информационно-энтропийные характеристики (включая фрактальный анализ временных рядов). Особое внимание уделяется алгоритмической реализации метрик, таких как Yule’s K, MTLD, Gries’ DP, Hurst Exponent и различных вариаций Delta-метода, с предоставлением математического обоснования и рекомендаций по их программной имплементации. Цель отчета — обеспечить теоретическую базу для создания продукта, отвечающего самым высоким стандартам математической точности в области компьютерной лингвистики.

---

## **1\. Введение: Проблема объективизации в пространстве текстовых данных**

Создание аналитической системы промышленного уровня неизбежно сталкивается с проблемой размерности и стохастической природы естественного языка. Когда мы ставим задачу «максимально полного сравнения текстов», мы фактически переходим от качественной филологии к многомерному статистическому анализу. Ключевым вызовом здесь является не столько извлечение признаков, сколько обеспечение их **робастности** (устойчивости) к внешним факторам, прежде всего к объему анализируемого материала ($N$).

Многие интуитивно понятные метрики, такие как *Type-Token Ratio* (отношение уникальных слов к общему числу слов), математически несостоятельны для профессионального сравнения. Согласно закону Хипса (Heaps' Law), рост словаря $V$ происходит нелинейно относительно длины текста $N$ ($V \\propto N^\\beta$, где $\\beta \< 1$), что приводит к тому, что более длинные тексты автоматически получают более низкие оценки лексического разнообразия, даже если они написаны более богатым языком. Для объективного сравнения текстов объемом в 2000 слов и текстов объемом в 100 000 слов необходимы метрики, которые либо асимптотически стабилизируются, либо используют процедуры нормализации, основанные на вероятностных моделях.

В данном исследовании предлагается архитектура анализа, базирующаяся на ортогональных векторах признаков. Это означает, что метрики выбраны таким образом, чтобы минимизировать их взаимную корреляцию, покрывая различные аспекты генерации текста автором: от бессознательного выбора служебных слов (Delta-методы) до когнитивной нагрузки при построении синтаксических деревьев (Yngve Depth) и динамики распределения информации во времени (Hurst Exponent).

---

## **2\. Лексическое богатство: Инвариантные метрики**

Оценка лексического разнообразия (Lexical Diversity, LD) является фундаментальным этапом стилометрии. Однако, как было отмечено, «наивные» подходы здесь неприменимы. Мы рассмотрим метрики, которые используют вероятностные модели и скользящие окна для устранения зависимости от длины текста.

### **2.1. Yule’s K: Вероятностная модель повторений**

Коэффициент $K$, предложенный статистиком Джорджем Юлом, остается одной из самых надежных метрик для оценки «лексической концентрации». В отличие от простых подсчетов уникальных слов, $K$ анализирует спектр частот — то есть, сколько слов встретилось один раз ($V\_1$, hapax legomena), сколько дважды ($V\_2$, dis legomena) и так далее.1

#### **Математическая формализация**

Основой метрики является предположение, что возникновение слов в тексте подчиняется пуассоновскому процессу. Юл вывел формулу, которая по сути представляет собой нормированный второй момент распределения частот слов.

Формула расчета:

$$K \= C \\times \\frac{S\_2 \- S\_1}{S\_1^2}$$

Где:

* $C$ — константа нормализации, обычно принимаемая за $10^4$ для приведения значений в удобный для чтения диапазон (например, 0-1000).1  
* $S\_1$ — общее количество токенов в тексте ($N \= \\sum m V\_m$).  
* $S\_2$ — сумма квадратов частот спектра: $S\_2 \= \\sum\_{m=1}^{V\_{max}} m^2 V\_m$.

Здесь $V\_m$ обозначает количество слов (типов), которые встречаются в тексте ровно $m$ раз.

#### **Интерпретация и свойства**

Метрика $K$ измеряет вероятность того, что два случайно выбранных слова из текста окажутся идентичными.

* Если текст состоит из абсолютно уникальных слов (максимальное разнообразие), то $V\_1 \= N$, $V\_{m\>1} \= 0$. Тогда $S\_2 \= 1^2 \\times N \= N$. Числитель формулы становится $N \- N \= 0$, следовательно, $K \= 0$.  
* Если текст состоит из бесконечного повторения одного слова, $S\_2 \= N^2$, и $K$ стремится к максимуму.3

Исследования показывают, что Yule's K демонстрирует исключительную стабильность при длине текста $N \> 1000$ токенов. Это делает его незаменимым при сравнении, например, коротких рассказов Чехова и романов Толстого. Важно отметить, что $K$ отражает не просто объем словаря, а *стиль повторений* автора: насколько часто он склонен возвращаться к одной и той же лексике.3

### **2.2. Алгоритмические метрики: MTLD и MATTR**

В то время как Yule's K базируется на глобальной статистике, современные подходы используют алгоритмическую сегментацию текста для оценки локального разнообразия.

#### **MTLD (Measure of Textual Lexical Diversity)**

Метрика MTLD, разработанная МакКарти и Джарвисом (2010), использует концепцию последовательного анализа. Идея заключается в том, чтобы измерить среднюю длину текстового сегмента, на котором автор способен поддерживать заданный уровень лексического разнообразия (обычно TTR \= 0.72).5

**Алгоритм реализации:**

1. **Инициализация:** Установить целевое значение TTR (фактор) $t \= 0.72$. Счетчик факторов $F \= 0$. Текущий список слов пуст.  
2. **Прямой проход:** Читать текст токен за токеном. После добавления каждого токена вычислять TTR текущего сегмента.  
3. **Сброс:** Как только TTR падает ниже $t$, инкрементировать счетчик факторов $F \= F \+ 1$, сбросить сегмент и начать накопление заново.  
4. **Обработка остатка:** Для последнего незавершенного сегмента вычисляется доля фактора (partial factor value) на основе отношения его TTR к пороговому значению, чтобы не терять данные в конце текста.7  
5. **Обратный проход:** Повторить процедуру, читая текст с конца к началу, чтобы нивелировать влияние композиции текста (например, вступления могут быть более шаблонными).  
6. **Результат:** Итоговое значение MTLD — это среднее гармоническое или арифметическое результатов прямого и обратного проходов: $MTLD \= \\frac{N}{F\_{avg}}$.

Высокое значение MTLD означает, что автор способен писать длинные пассажи без частого повторения лексики. Это метрика «выносливости» словарного запаса.

#### **MATTR (Moving Average Type-Token Ratio)**

MATTR (Covington & McFall, 2010\) решает проблему чувствительности к структуре текста методом скользящего окна. Это наиболее «гладкая» и статистически обоснованная метрика для коротких текстов.1

Формула и алгоритм:  
Для окна фиксированной длины $L$ (стандарт в индустрии — 500 слов, но для коротких текстов можно использовать 50-100 1):

$$\\text{MATTR} \= \\frac{1}{N \- L \+ 1} \\sum\_{i=1}^{N \- L \+ 1} \\text{TTR}(w\_i, w\_{i+1}, \\dots, w\_{i+L-1})$$

Где $w\_i$ — токен на позиции $i$.  
Преимущество MATTR перед MTLD заключается в том, что оно не зависит от произвольного порога (0.72) и использует все возможные сегменты текста длины $L$. Это обеспечивает микро-усреднение разнообразия и исключает артефакты, связанные с разрывом сегментов в неудачных местах.10 В реализации на Python или R следует использовать оптимизированные структуры данных (например, collections.Counter с обновлением при сдвиге окна), чтобы избежать квадратичной сложности вычислений $O(N \\cdot L)$.10

---

## **3\. Дисперсия и динамика: От статики к «Взрывности»**

Частота слова (Term Frequency) сама по себе является плохим дескриптором стиля. Слово, встречающееся 50 раз в первой главе и ни разу далее, стилистически отличается от слова, встречающегося 50 раз равномерно по всему тексту. Для профессионального анализа необходимо ввести метрики *дисперсии* и *взрывности* (burstiness).

### **3.1. Gries’ DP: Объективная оценка дисперсии**

Стефан Грис (2008) предложил метрику *Deviation of Proportions* (DP), которая на сегодняшний день считается золотым стандартом оценки равномерности распределения, так как она не чувствительна к размеру корпуса, в отличие от ранних метрик Джуилланда (Juilland's D).11

Математический смысл:  
DP вычисляет манхэттенское расстояние между вектором наблюдаемых частот в частях корпуса и вектором ожидаемых частот (при гипотезе равномерного распределения).  
Формула расчета:

$$DP \= \\frac{1}{2} \\sum\_{i=1}^{n} \\left| \\frac{v\_i}{F} \- \\frac{s\_i}{S} \\right|$$

Где:

* $n$ — количество сегментов, на которые разбит текст (или количество текстов в корпусе автора).  
* $v\_i$ — абсолютная частота исследуемого слова в сегменте $i$.  
* $F$ — общая частота слова во всем тексте ($F \= \\sum v\_i$).  
* $s\_i$ — объем сегмента $i$ в словах.  
* $S$ — общий объем текста ($S \= \\sum s\_i$).

Значения DP варьируются от $0$ (идеальная дисперсия, слово размазано ровным слоем) до $\\approx 1$ (слово сконцентрировано в одном узком месте). Для удобства интерпретации в интерфейсе продукта рекомендуется использовать инвертированную и нормализованную метрику:

$$DP\_{norm} \= 1 \- \\frac{DP}{1 \- \\min(s\_i/S)}$$

Такая метрика ($DP\_{norm} \\to 1$) будет указывать на высокую равномерность использования слова, что характерно для функциональной лексики и базовых маркеров стиля автора.12

### **3.2. Квантификация «Взрывности»: Модель Пьерхамберт**

Понятие «взрывности» (burstiness) описывает феномен, когда использование слова провоцирует его повторное использование в ближайшем контексте (эффект прайминга), что нарушает пуассоновское распределение. Джанет Пьерхамберт (2009) показала, что распределение интервалов времени (расстояний в словах) между повторами слов лучше всего описывается не экспонентой, а **растянутым экспоненциальным распределением (Weibull distribution)**.15

#### **Параметр $\\beta$ (Shape Parameter)**

Плотность вероятности интервала $t$ между вхождениями слова моделируется как:

$$P(t) \\sim e^{-t^\\beta}$$

Где $\\beta$ — параметр формы (shape parameter).

* Если $\\beta \= 1$, распределение становится экспоненциальным ($e^{-t}$), что соответствует отсутствию памяти (процесс Пуассона).  
* Если $\\beta \< 1$, распределение имеет «тяжелый хвост» (fat tail). Это означает, что слово характеризуется чередованием периодов высокой активности («взрывов») и длительных пауз.

Для вычисления $\\beta$ в программном продукте необходимо собрать массив всех расстояний $t\_1, t\_2, \\dots, t\_k$ между последовательными употреблениями каждого слова и применить метод максимального правдоподобия (MLE) для подгонки распределения Вейбулла. Параметр $\\beta$ является мощнейшим индикатором семантического класса слова: контентные слова имеют низкий $\\beta$, тогда как служебные слова приближаются к $\\beta \\approx 1$.16

#### **Коэффициент взрывности $B$**

Для более быстрой, но менее детальной оценки можно использовать коэффициент $B$, предложенный Го и Барабаши, который вычисляется через моменты распределения интервалов:

$$B \= \\frac{\\sigma\_\\tau \- \\mu\_\\tau}{\\sigma\_\\tau \+ \\mu\_\\tau}$$

Где $\\mu\_\\tau$ — среднее значение интервалов между вхождениями слова, а $\\sigma\_\\tau$ — их стандартное отклонение.18

* $B \\to 1$: Сильная взрывность.  
* $B \= 0$: Нейтральное (случайное) распределение.  
* $B \\to \-1$: Регулярное, периодическое распределение.

Включение параметра $B$ или $\\beta$ в профиль автора позволяет отличать авторов, склонных к «застреванию» на темах, от авторов с широким динамическим диапазоном лексики.21

---

## **4\. Синтаксическая сложность: Топология мысли**

Большинство стилометрических систем ограничиваются лексикой. Однако синтаксис — это каркас мышления, который гораздо труднее сознательно имитировать или изменять. Для профессионального продукта необходим глубокий парсинг (parsing) и метрики, основанные на геометрии деревьев разбора.

### **4.1. Глубина Ингве (Yngve Depth): Когнитивная нагрузка**

Виктор Ингве (1960) сформулировал гипотезу, связывающую структуру предложений с ограничениями оперативной памяти человека. Он предложил метрику глубины, основанную на асимметрии левого ветвления (в английском и многих индоевропейских языках левое ветвление создает большую нагрузку на память, так как требует удержания незавершенных составляющих).22

**Алгоритм расчета на дереве разбора:**

1. Производится синтаксический парсинг предложения (например, с использованием PCFG или нейросетевых парсеров типа Stanford CoreNLP/SpaCy), строится дерево составляющих.  
2. Для каждого терминального узла (слова) вычисляется его глубина Ингве ($d\_Y$). Глубина определяется количеством **левых ветвей**, которые необходимо пройти при спуске от корня к данному слову.  
3. **Альтернативная (вычислительно более простая) формулировка:** При обходе дерева каждой ветви, уходящей вправо, присваивается вес 0, а каждой ветви, уходящей влево — вес, равный количеству её правых сестер. Глубина слова — сумма весов на пути от корня.22  
   * Уточнение по 24: Часто используется подсчет количества правых сестер для каждого узла на пути от корня к слову.  
   * Формула: $d(w) \= \\sum\_{n \\in Path(root, w)} \\text{RightSiblings}(n)$.

Метрика позволяет вычислить среднюю глубину предложения и максимальную глубину. Высокие значения указывают на сложный, «утяжеленный» синтаксис, свойственный академическим или архаичным текстам.

### **4.2. Метрика Фрейзера (Frazier Score): Стоимость встраивания**

Лин Фрейзер (1985) предложила метрику, которая фокусируется не просто на ветвлении, а на *синтаксическом встраивании* (embedding) предложенческих структур (S-nodes). Это более тонкий инструмент для оценки грамматической сложности.26

**Алгоритм начисления баллов:**

1. Проход по дереву разбора. Каждому узлу присваивается базовый вес.  
2. Обычные узлы (NP, VP, PP) получают вес **1**.  
3. Узлы, обозначающие предложения (S, S-bar, SQ), получают вес **1.5**. Это ключевое отличие: вложенные предложения «стоят» дороже.27  
4. Счет слова вычисляется как сумма весов узлов на пути от слова вверх по дереву до тех пор, пока не будет достигнут корень или узел, который не является крайним левым потомком своего родителя.  
5. Общая оценка предложения (Frazier Score) часто рассчитывается не как простая сумма, а как среднее значение или максимальная сумма очков в скользящем окне из 3 слов, чтобы оценить локальную когнитивную перегрузку.27

### **4.3. L2SCA Метрики**

Для полноты картины необходимо включить набор из 14 метрик, стандартизированных в инструменте L2SCA (Lu, 2010), которые стали де\-факто стандартом в прикладной лингвистике. К ним относятся:

* **MLC (Mean Length of Clause):** Средняя длина клаузы.  
* **C/S (Clauses per Sentence):** Степень гипотаксиса (подчинения).  
* **DC/C (Dependent Clauses per Clause):** Насыщенность придаточными предложениями.  
* **CN/C (Complex Nominals per Clause):** Насыщенность сложными именными группами.29

Эти метрики требуют качественного POS-теггинга и определения границ клауз (clause boundaries), но они отлично дифференцируют стили (например, научный стиль характеризуется высоким CN/C, а художественный — высоким C/S и Yngve Depth).

---

## **5\. Энтропия и Теория Информации: Измерение предсказуемости**

Текст можно рассматривать как стохастический процесс передачи информации. Объективное сравнение авторов невозможно без оценки того, насколько предсказуем их текст.

### **5.1. Условная энтропия N-грамм**

Классическая энтропия Шеннона ($H$) для униграмм (отдельных слов) дает информацию только о богатстве словаря. Гораздо более информативна **условная энтропия** (Conditional Entropy), которая показывает, насколько сложно предсказать следующее слово, зная предыдущие $n-1$ слов.31

Формула:  
$$ H(X\_n | X\_{n-1}, \\dots, X\_{n-k}) \= \- \\sum p(x\_{n-k}, \\dots, x\_n) \\log\_2 p(x\_n | x\_{n-1}, \\dots, x\_{n-k}) $$  
Для профессионального продукта рекомендуется строить профили энтропии для биграмм ($n=2$) и триграмм ($n=3$). Автор с низкой условной энтропией использует клише и устойчивые обороты. Автор с высокой энтропией создает неожиданные, нетривиальные связи между словами.

### **5.2. Сложность Лемпеля-Зива (Lempel-Ziv Complexity)**

Алгоритмическая сложность (Колмогоровская сложность) невычислима, но сложность Лемпеля-Зива (LZ) служит её отличной оценкой. Она базируется на алгоритмах сжатия данных. В контексте стилометрии LZ измеряет количество уникальных паттернов в тексте.

Relative Lempel-Ziv Complexity:  
Для сравнения двух текстов $A$ и $B$ можно использовать перекрестный подход (Cross-LZ): сжимать текст $A$, используя словарь, построенный на тексте $B$. Коэффициент сжатия будет служить мерой структурного подобия (расстояния) между текстами. Чем лучше текст $A$ сжимается словарем $B$, тем ближе стили авторов. Это непараметрический метод, не требующий предварительного обучения моделей.33

---

## **6\. Фрактальный анализ и Временные ряды**

Текст — это не «мешок слов» (Bag of Words), а последовательность, разворачивающаяся во времени. Преобразование текста в числовой временной ряд (Time Series) открывает доступ к мощному аппарату нелинейной динамики.

### **6.1. Отображение текста во временной ряд**

Для анализа текст преобразуется в ряд $\\{x\_1, x\_2, \\dots, x\_N\\}$, где $x\_i$ может быть:

* Длиной $i$-го слова в символах.  
* Частотным рангом $i$-го слова.  
* Кодом ASCII (менее лингвистически обосновано, но используется).  
  Исследования 35 показывают, что ряды длин слов (Sequence of Word Lengths, SWL) наиболее информативны для стилометрии.

### **6.2. Показатель Хёрста и DFA**

Показатель Хёрста ($H$, $0 \< H \< 1$) характеризует степень персистентности (памяти) ряда.

* $H \= 0.5$: Случайный процесс (белый шум). Автор не контролирует ритм длин слов.  
* $H \> 0.5$: Персистентный процесс (длинная память). Длинные слова тяготеют к длинным, короткие — к коротким. Это признак сложной, фрактальной структуры текста, характерной для качественной художественной прозы.  
* $H \< 0.5$: Анти-персистентность. Резкие скачки (длинное слово сразу сменяется коротким).

Для расчета $H$ в профессиональном продукте следует использовать метод **DFA (Detrended Fluctuation Analysis)**, так как он устойчив к нестационарным трендам (например, изменению стиля в ходе повествования).37

**Алгоритм DFA:**

1. **Интегрирование:** Преобразовать исходный ряд $x\_t$ (с вычтенным средним) в кумулятивный профиль: $Y(k) \= \\sum\_{t=1}^k (x\_t \- \\bar{x})$.  
2. **Сегментация:** Разбить профиль $Y(k)$ на непересекающиеся окна равной длины $s$.  
3. **Детрендинг:** В каждом окне аппроксимировать данные полиномом (обычно 1-го порядка — линейный тренд, DFA1) $Y\_{trend}(k)$ и вычесть его: $Y\_{detrend}(k) \= Y(k) \- Y\_{trend}(k)$.  
4. **Расчет флуктуации:** Вычислить среднеквадратичное отклонение $F(s) \= \\sqrt{\\frac{1}{N} \\sum Y\_{detrend}^2(k)}$.  
5. **Скейлинг:** Повторить для разных масштабов $s$.  
6. **Результат:** Построить график $\\log F(s)$ от $\\log s$. Наклон прямой (аппроксимация методом наименьших квадратов) равен коэффициенту Хёрста $H$ (или $\\alpha$ в терминологии DFA).35

### **6.3. Закон Ципфа-Мандельброта: Параметризация словаря**

Распределение частот слов в тексте следует закону Ципфа, но для точного анализа используется уточнение Мандельброта:

$$f(r) \= \\frac{C}{(r \+ \\beta)^\\alpha}$$

Параметры $\\alpha$ и $\\beta$ являются уникальными характеристиками автора. $\\alpha$ (обычно $\\approx 1$) отвечает за наклон «хвоста» (богатство редкой лексики), а $\\beta$ корректирует изгиб в начале распределения (разнообразие самых частотных слов). Извлечение этих параметров через нелинейную регрессию дает компактный вектор признаков для кластеризации авторов.40

---

## **7\. Атрибуция: Векторные модели Delta**

Для задачи идентификации автора (Attribution) наиболее объективным методом является семейство метрик Delta, работающих с векторами частот наиболее частотных слов (Most Frequent Words, MFW).

### **7.1. Burrows' Delta: Классический подход**

Джон Берроуз предложил использовать Z-оценки для стандартизации частот. Это критически важно: колебание частоты слова "the" на 0.1% и слова "however" на 0.1% имеет разный вес. Z-score уравнивает их дисперсии.

Формула расстояния между текстом $T$ и корпусом кандидата $C$:

$$\\Delta(C, T) \= \\frac{1}{n} \\sum\_{i=1}^{n} |Z\_{i, C} \- Z\_{i, T}|$$

Где $Z\_{i} \= \\frac{\\text{freq}\_i \- \\mu\_i}{\\sigma\_i}$. Параметры $\\mu$ и $\\sigma$ рассчитываются по всему сравниваемому корпусу.42

### **7.2. Eder’s Delta: Взвешивание по рангу**

Мацей Эдер показал, что слова, находящиеся в «хвосте» списка MFW (например, ранги 1000-2000), часто несут больше авторского сигнала, чем топ-10 служебных слов. В классической Delta все слова имеют равный вес. Eder’s Delta вводит веса $w\_i$, зависящие от ранга слова.

В программном пакете stylo Эдер реализует различные схемы взвешивания, но наиболее эффективной для флективных языков (как русский) считается схема, где вес уменьшается с увеличением частоты (или наоборот, оптимизируется под задачу). Эмпирически было показано, что для атрибуции важно давать больший вес словам средней частотности, поэтому формула может модифицироваться как:

$$\\Delta\_{Eder}(C, T) \= \\sum\_{i=1}^{n} w\_i |Z\_{i, C} \- Z\_{i, T}|$$

Где веса $w\_i$ могут быть определены, например, как $1 \- \\frac{rank\_i}{n}$ (уменьшение влияния редких слов, если список слишком длинный) или через более сложные функции затухания, реализованные в stylo.43

### **7.3. Cosine Delta: Векторная нормализация**

Исследования (Jannidis et al., Smith & Aldridge) показали, что использование косинусного расстояния вместо Манхэттенского (L1) улучшает результаты, особенно при сравнении текстов разной длины, так как косинусное расстояние по определению нормализует длину векторов.45

$$ \\Delta\_{\\angle}(C, T) \= 1 \- \\cos(\\vec{Z\_C}, \\vec{Z\_T}) \= 1 \- \\frac{\\sum (Z\_{i,C} \\cdot Z\_{i,T})}{\\sqrt{\\sum Z\_{i,C}^2} \\cdot \\sqrt{\\sum Z\_{i,T}^2}} $$  
Эта метрика («Würzburg Delta») на текущий момент считается наиболее стабильной (state-of-the-art) для задач атрибуции на больших корпусах.46

---

## **8\. Итоговая архитектура системы**

Для реализации профессионального продукта рекомендуется следующий стек метрик, сгруппированных по аналитическим слоям. Данная комбинация обеспечивает «максимально полное сравнение», требуемое в ТЗ.

| Измерение анализа | Рекомендуемая метрика | Почему это объективно | Ключевые параметры |
| :---- | :---- | :---- | :---- |
| **Лексика** | **Yule's K** | Инвариантна к длине ($N\>1000$), оценивает повторяемость. | $C=10^4$ |
|  | **MTLD** | Интуитивна, оценивает длину уникальных сегментов. | Threshold TTR \= 0.72 |
| **Дисперсия** | **Gries' DP (norm)** | Не зависит от частоты слова, показывает равномерность. | Нормализация на $1 \- \\min(s\_i/S)$ |
| **Взрывность** | **Pierrehumbert's $\\beta$** | Глубокая оценка семантической структуры (Weibull fit). | MLE fitting для $P(t) \\sim e^{-t^\\beta}$ |
| **Синтаксис** | **Yngve Depth** | Оценка когнитивной нагрузки (левое ветвление). | Метод правых сестер |
|  | **Frazier Score** | Оценка вложенности предложений. | Penalty 1.5 для S-nodes |
| **Динамика** | **Hurst Exponent (DFA)** | Фрактальность, память текста, структура повествования. | Окна $s$, полином 1-го порядка |
| **Атрибуция** | **Cosine Delta** | Векторное сравнение стиля, устойчивое к шуму. | Top-2000 MFW, Z-scores |

### **Заключение**

Предложенный набор метрик формирует математически строгую, многомерную проекцию текста. Отказ от линейных метрик (TTR) в пользу вероятностных (Yule's K, MTLD) и внедрение методов из физики сложных систем (DFA, Burstiness) позволит вашему продукту выявлять скрытые паттерны авторского стиля, недоступные для поверхностного анализа. Реализация данных алгоритмов требует тщательной работы с препроцессингом (лемматизация для русского языка обязательна), но именно этот уровень сложности отличает профессиональный инструмент от любительского скрипта.

#### **Works cited**

1. Investigating Lexical Progression through Lexical Diversity Metrics in a Corpus of French L3, accessed November 30, 2025, [https://journals.openedition.org/discours/9950](https://journals.openedition.org/discours/9950)  
2. textstat\_lexdiv Calculate lexical diversity \- RDocumentation, accessed November 30, 2025, [https://www.rdocumentation.org/packages/quanteda/versions/1.3.4/topics/textstat\_lexdiv](https://www.rdocumentation.org/packages/quanteda/versions/1.3.4/topics/textstat_lexdiv)  
3. Computational Constancy Measures of Texts—Yule's K and Rényi's Entropy, accessed November 30, 2025, [https://direct.mit.edu/coli/article/41/3/481/1519/Computational-Constancy-Measures-of-Texts-Yule-s-K](https://direct.mit.edu/coli/article/41/3/481/1519/Computational-Constancy-Measures-of-Texts-Yule-s-K)  
4. Lexical diversity: Yule's K \- R, accessed November 30, 2025, [https://search.r-project.org/CRAN/refmans/koRpus/html/K.ld.html](https://search.r-project.org/CRAN/refmans/koRpus/html/K.ld.html)  
5. Measuring Lexical Diversity in Narrative Discourse of People With Aphasia \- PMC \- NIH, accessed November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3813439/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3813439/)  
6. Measure of Textual Lexical Diversity (MTLD), accessed November 30, 2025, [https://search.r-project.org/CRAN/refmans/koRpus/html/MTLD.html](https://search.r-project.org/CRAN/refmans/koRpus/html/MTLD.html)  
7. Algorithmic and subjective measures of lexical diversity in bilingual written corpora: a discussion \- OpenEdition Journals, accessed November 30, 2025, [https://journals.openedition.org/corela/4843](https://journals.openedition.org/corela/4843)  
8. Psychometric Evaluation of Lexical Diversity Indices: Assessing Length Effects \- PMC \- NIH, accessed November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4490052/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4490052/)  
9. Evaluating evidence for the reliability and validity of lexical diversity indices in L2 oral task responses | Studies in Second Language Acquisition | Cambridge Core, accessed November 30, 2025, [https://www.cambridge.org/core/journals/studies-in-second-language-acquisition/article/evaluating-evidence-for-the-reliability-and-validity-of-lexical-diversity-indices-in-l2-oral-task-responses/E51955F0291E21A916CD2C4787508B80](https://www.cambridge.org/core/journals/studies-in-second-language-acquisition/article/evaluating-evidence-for-the-reliability-and-validity-of-lexical-diversity-indices-in-l2-oral-task-responses/E51955F0291E21A916CD2C4787508B80)  
10. Wait, what?\! Python is quicker than Rust when calculating MATTR lexical diversity, accessed November 30, 2025, [https://www.youtube.com/watch?v=uVitm6KW-3A](https://www.youtube.com/watch?v=uVitm6KW-3A)  
11. Dispersion Measures as Predictors of Lexical Decision Time, Word Familiarity, and Lexical Complexity \- arXiv, accessed November 30, 2025, [https://arxiv.org/html/2501.06536v1](https://arxiv.org/html/2501.06536v1)  
12. Analyzing Dispersion \- Stefan Th. Gries, accessed November 30, 2025, [https://www.stgries.info/research/2020\_STG\_Dispersion\_PHCL.pdf](https://www.stgries.info/research/2020_STG_Dispersion_PHCL.pdf)  
13. disp\_DP() R function from \[tlda\] \- r packages, accessed November 30, 2025, [https://r-packages.io/packages/tlda/disp\_DP](https://r-packages.io/packages/tlda/disp_DP)  
14. Advancing our understanding of dispersion measures in corpus research | Corpora, accessed November 30, 2025, [https://www.euppublishing.com/doi/full/10.3366/cor.2025.0326](https://www.euppublishing.com/doi/full/10.3366/cor.2025.0326)  
15. Beyond Word Frequency: Bursts, Lulls, and Scaling in the Temporal Distributions of Words, accessed November 30, 2025, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0007678](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0007678)  
16. Beyond Word Frequency: Bursts, Lulls, and Scaling in the Temporal Distributions of Words, accessed November 30, 2025, [https://www.researchgate.net/publication/38084102\_Beyond\_Word\_Frequency\_Bursts\_Lulls\_and\_Scaling\_in\_the\_Temporal\_Distributions\_of\_Words](https://www.researchgate.net/publication/38084102_Beyond_Word_Frequency_Bursts_Lulls_and_Scaling_in_the_Temporal_Distributions_of_Words)  
17. Stretched Exponential Dynamics in Online Article Views \- Frontiers, accessed November 30, 2025, [https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2020.619729/full](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2020.619729/full)  
18. Burstiness \- Wikipedia, accessed November 30, 2025, [https://en.wikipedia.org/wiki/Burstiness](https://en.wikipedia.org/wiki/Burstiness)  
19. Exploring Burstiness: Evaluating Language Dynamics in LLM-Generated Texts, accessed November 30, 2025, [https://ramblersm.medium.com/exploring-burstiness-evaluating-language-dynamics-in-llm-generated-texts-8439204c75c1](https://ramblersm.medium.com/exploring-burstiness-evaluating-language-dynamics-in-llm-generated-texts-8439204c75c1)  
20. Burstiness and memory in complex systems | Request PDF \- ResearchGate, accessed November 30, 2025, [https://www.researchgate.net/publication/231033081\_Burstiness\_and\_memory\_in\_complex\_systems](https://www.researchgate.net/publication/231033081_Burstiness_and_memory_in_complex_systems)  
21. Measuring and Modeling Bursty Human Phenomena \- arXiv, accessed November 30, 2025, [https://arxiv.org/html/2412.13617v1](https://arxiv.org/html/2412.13617v1)  
22. Depth in English grammar \- Geoffrey Sampson, accessed November 30, 2025, [https://www.grsampson.net/ADie.html](https://www.grsampson.net/ADie.html)  
23. Letter to the Editor: Clues from the Depth Hypothesis: A Reply to Geoffrey Sampson's Review \- ACL Anthology, accessed November 30, 2025, [https://aclanthology.org/J98-4005.pdf](https://aclanthology.org/J98-4005.pdf)  
24. Automated Measures of Syntactic Complexity in Natural Speech Production: Older and Younger Adults as a Case Study \- ASHA Journals, accessed November 30, 2025, [https://pubs.asha.org/doi/10.1044/2023\_JSLHR-23-00009](https://pubs.asha.org/doi/10.1044/2023_JSLHR-23-00009)  
25. Spoken Language Derived Measures for Detecting Mild Cognitive Impairment \- PMC \- NIH, accessed November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3244269/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3244269/)  
26. Automated Measures of Syntactic Complexity in Natural Speech Production: Older and Younger Adults as a Case Study, accessed November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12510242/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12510242/)  
27. Syntactic complexity measures for detecting Mild Cognitive Impairment \- ACL Anthology, accessed November 30, 2025, [https://aclanthology.org/W07-1001.pdf](https://aclanthology.org/W07-1001.pdf)  
28. Automated Measures of Syntactic Complexity in Natural Speech Production: Older and Younger Adults as a Case Study | Request PDF \- ResearchGate, accessed November 30, 2025, [https://www.researchgate.net/publication/377361742\_Automated\_Measures\_of\_Syntactic\_Complexity\_in\_Natural\_Speech\_Production\_Older\_and\_Younger\_Adults\_as\_a\_Case\_Study](https://www.researchgate.net/publication/377361742_Automated_Measures_of_Syntactic_Complexity_in_Natural_Speech_Production_Older_and_Younger_Adults_as_a_Case_Study)  
29. Web-based L2 Syntactic Complexity Analyzer \- Haiyang Ai, accessed November 30, 2025, [https://aihaiyang.com/software/l2sca/](https://aihaiyang.com/software/l2sca/)  
30. Kolmogorov complexity metrics in assessing L2 proficiency: An information-theoretic approach \- PMC \- NIH, accessed November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9583672/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9583672/)  
31. N-gram Language Models \- Stanford University, accessed November 30, 2025, [https://web.stanford.edu/\~jurafsky/slp3/3.pdf](https://web.stanford.edu/~jurafsky/slp3/3.pdf)  
32. PREDICTION AND ENTROPY OF PRINTED ENGLISH \- erratic and uncertain, and they depend more critically on the type of text involved. \- Language Log, accessed November 30, 2025, [https://languagelog.ldc.upenn.edu/myl/Shannon1950.pdf](https://languagelog.ldc.upenn.edu/myl/Shannon1950.pdf)  
33. A Quick and Easy Way to Estimate Entropy and Mutual Information for Neuroscience \- NIH, accessed November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8239197/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8239197/)  
34. Estimating the Entropy Rate of Spike Trains via Lempel-Ziv Complexity \- SciSpace, accessed November 30, 2025, [https://scispace.com/pdf/estimating-the-entropy-rate-of-spike-trains-via-lempel-ziv-36iyp98428.pdf](https://scispace.com/pdf/estimating-the-entropy-rate-of-spike-trains-via-lempel-ziv-36iyp98428.pdf)  
35. Scale and time dependence of serial correlations in word-length time series of written texts, accessed November 30, 2025, [https://www.researchgate.net/publication/264827042\_Scale\_and\_time\_dependence\_of\_serial\_correlations\_in\_word-length\_time\_series\_of\_written\_texts](https://www.researchgate.net/publication/264827042_Scale_and_time_dependence_of_serial_correlations_in_word-length_time_series_of_written_texts)  
36. \[1208.6174\] Generalized Hurst exponent and multifractal function of original and translated texts mapped into frequency and length time series \- arXiv, accessed November 30, 2025, [https://arxiv.org/abs/1208.6174](https://arxiv.org/abs/1208.6174)  
37. accessed November 30, 2025, [https://www.mathworks.com/matlabcentral/fileexchange/19795-detrended-fluctuation-analysis\#:\~:text=In%20stochastic%20processes%2C%20chaos%20theory,to%20be%20long%2Dmemory%20processes.](https://www.mathworks.com/matlabcentral/fileexchange/19795-detrended-fluctuation-analysis#:~:text=In%20stochastic%20processes%2C%20chaos%20theory,to%20be%20long%2Dmemory%20processes.)  
38. On the Validity of Detrended Fluctuation Analysis at Short Scales \- PMC \- NIH, accessed November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8775092/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8775092/)  
39. Detrended fluctuation analysis \- Wikipedia, accessed November 30, 2025, [https://en.wikipedia.org/wiki/Detrended\_fluctuation\_analysis](https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis)  
40. Zipf–Mandelbrot law \- Wikipedia, accessed November 30, 2025, [https://en.wikipedia.org/wiki/Zipf%E2%80%93Mandelbrot\_law](https://en.wikipedia.org/wiki/Zipf%E2%80%93Mandelbrot_law)  
41. (PDF) Mandelbrot's Model for Zipf's Law: Can Mandelbrot's Model Explain Zipf's Law for Language? \- ResearchGate, accessed November 30, 2025, [https://www.researchgate.net/publication/220469172\_Mandelbrot's\_Model\_for\_Zipf's\_Law\_Can\_Mandelbrot's\_Model\_Explain\_Zipf's\_Law\_for\_Language](https://www.researchgate.net/publication/220469172_Mandelbrot's_Model_for_Zipf's_Law_Can_Mandelbrot's_Model_Explain_Zipf's_Law_for_Language)  
42. Introduction to stylometry with Python | Programming Historian, accessed November 30, 2025, [https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python](https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python)  
43. Beyond the black box, or: understanding the difference between various statistical distance measures \- The Dragonfly's Gaze, accessed November 30, 2025, [https://dragonfly.hypotheses.org/101](https://dragonfly.hypotheses.org/101)  
44. Towards a better understanding of Burrows's Delta in literary authorship attribution \- ACL Anthology, accessed November 30, 2025, [https://aclanthology.org/anthology-files/pdf/W/W15/W15-0709.pdf](https://aclanthology.org/anthology-files/pdf/W/W15/W15-0709.pdf)  
45. Understanding and explaining Delta measures for authorship attribution \- Oxford Academic, accessed November 30, 2025, [https://academic.oup.com/dsh/article/32/suppl\_2/ii4/3865676](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676)  
46. Stylometry and the Voice of Hildegard \- Humanities Data Analysis, accessed November 30, 2025, [https://www.humanitiesdataanalysis.org/stylometry/notebook.html](https://www.humanitiesdataanalysis.org/stylometry/notebook.html)  
47. Comparison of distance and similarity measures for stylometric analysis of Lithuanian texts \- CEUR-WS.org, accessed November 30, 2025, [https://ceur-ws.org/Vol-1852/p01.pdf](https://ceur-ws.org/Vol-1852/p01.pdf)