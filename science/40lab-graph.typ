#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: diamond, brace
#diagram(
 debug: 3,
 node-stroke: black + 0.5pt,
 node-fill: gradient.radial(white, blue, center: (40%, 20%),
 radius: 150%),
 spacing: (10mm, 5mm),
 node((0,0), [1], name: <1>, extrude: (0, -4)), // double stroke
 edge("=>"),
 node((1,0), [2], name: <2>, shape: diamond),
 node((2,-1), [3a], name: <3a>),
 node((2,+1), [3b], name: <3b>),
 node(enclose: (<1>, <2>), shape: brace.with(dir: top, label: [12])),
 edge(<2.east>, "->", <3a>, bend: -15deg),
 edge(<2.east>, "->", <3b>, bend: +15deg),
 edge(<3b>, "~>", <3b>, bend: -130deg, loop-angle: 120deg)[loop!],
)
