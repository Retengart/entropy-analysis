#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: pill, rect

#set page(width: auto, height: auto, margin: 1cm)
#set text(font: "Linux Libertine", lang: "ru", size: 10pt)

// Вспомогательная функция для контента блока: номер в углу + текст по центру
#let block_content(num, body) = {
  place(top + left, text(weight: "bold", size: 0.9em, num))
  align(center + horizon, body)
}

// Стиль для основных функциональных блоков
#let block_style = (
  shape: rect,
  width: 55mm,
  height: 24mm,
  stroke: 1pt,
  fill: white,
  inset: 8pt,
)

#diagram(
  spacing: (20mm, 15mm),
  node-stroke: 1pt,
  edge-stroke: 1pt,

  // --- УЗЛЫ (NODES) ---

  // Начало
  node((1, 0), [НАЧАЛО], shape: pill, width: 30mm, height: 12mm, stroke: 1.5pt, fill: white, name: <start>),

  // === ЛЕВАЯ КОЛОНКА (x=0) ===
  node((0, 1), block_content("1", [Ввод исходных данных\ по классам (эталонам)]), ..block_style, name: <n1>),

  node((0, 2), block_content("2", [Оценка\ информативности\ признаков $I_k$]), ..block_style, name: <n2>),

  node((0, 3), block_content("2'", [Оценка вероятности\ распознавания по\ информационным признакам]), ..block_style, name: <n2p>),

  node((0, 4), block_content("3", [Расчёт условных\ и апостериорных\ вероятностей наиболее\ вероятных реализаций]), ..block_style, name: <n3>),

  node((0, 5), block_content("4", [Определение наиболее\ вероятных реализаций\ на основе оценки\ $P(l\/b j) <= p(l)_"зад"$]), ..block_style, name: <n4>),

  // === ПРАВАЯ КОЛОНКА (x=2) ===
  node((2, 1), block_content("5", [Формирование\ исходных данных\ по виду учитываемых помех]), ..block_style, name: <n5>),

  node((2, 2), block_content("6", [Ввод исходных данных\ по классам в условиях помех\ по вероятным реализациям]), ..block_style, name: <n6>),

  node((2, 3), block_content("7", [Расчёт условных\ и апостериорных\ распределений реализаций\ в условиях помех]), ..block_style, name: <n7>),

  node((2, 4), block_content("8", [Вычисление $P^*(l)b j$\ для каждой реализации]), ..block_style, name: <n8>),

  node((2, 5), block_content("9", [Сравнение $P^*(l)b j$\ с заданной\ $P^*(l)b j >= p^*(l)_"зад"$]), ..block_style, name: <n9>),

  // === НИЖНЯЯ ЧАСТЬ (Центр x=1) ===
  node((1, 6), block_content("10", [Вычисление\ $delta P(l) = (P^*(l) - P(l)) / P^*(l)$]), ..block_style, width: 70mm, name: <n10>),

  node((1, 7), block_content("11", [Сравнение помех по $delta P(l)$\ выбор помехи]), ..block_style, width: 70mm, name: <n11>),

  node((1, 8), [КОНЕЦ], shape: pill, width: 30mm, height: 12mm, stroke: 1.5pt, fill: white, name: <end>),


  // --- СВЯЗИ (EDGES) ---

  edge(<start>, <n1>, "->", corner: left),
  edge(<start>, <n5>, "->", corner: right),

  edge(<n1>, <n2>, "->"),
  edge(<n2>, <n2p>, "->"),
  edge(<n2p>, <n3>, "->"),
  edge(<n3>, <n4>, "->"),

  edge(<n5>, <n6>, "->"),
  edge(<n6>, <n7>, "->"),
  edge(<n7>, <n8>, "->"),
  edge(<n8>, <n9>, "->"),

  // Связь 1 -> 6 (через центр, чтобы не наезжать на блоки)
  edge(<n1.east>, (1, 1), (1, 2), <n6.west>, "->"),

  // Обратная связь 4 -> 1 (через левую сторону с большим отступом)
  // Используем промежуточные точки (-1.5, ...) чтобы обойти входы
  edge(<n4.west>, (-1.5, 5), (-1.5, 1), <n1.west>, "->", corner-radius: 5pt),

  // Слияние в 10
  edge(<n4>, <n10.north>, "->", corner: left),
  edge(<n9>, <n10.north>, "->", corner: right),

  edge(<n10>, <n11>, "->"),
  edge(<n11>, <end>, "->"),


  // --- БОКОВЫЕ ВХОДЫ ---

  // Используем координаты x=-0.8 чтобы быть между блоками и линией обратной связи
  node((-0.8, 2.5), $P(x^*_l)$, stroke: none, name: <input_px>),
  // Подключаем к конкретным якорям (.west)
  edge(<input_px>, <n2.west>, "->", corner: left),
  edge(<input_px>, <n2p.west>, "->", corner: left),

  node((-0.8, 5), $p(l)_"зад"$, stroke: none, name: <input_pl>),
  edge(<input_pl>, <n4.west>, "->"),

  node((2.8, 5), $p^*(l)_"зад"$, stroke: none, name: <input_pl_star>),
  edge(<input_pl_star>, <n9.east>, "->"),
)
