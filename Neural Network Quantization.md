---


---

<h1 id="neural-network-quantization">Neural Network Quantization</h1>
<blockquote>
<p>Written with <a href="https://stackedit.io/">StackEdit</a>.</p>
</blockquote>
<h1 id="introduction">Introduction</h1>
<p>This is a reading note of <a href="https://arxiv.org/abs/1806.08342">Quantizing deep convolutional networks for efficient inference: A whitepaper</a>.</p>
<h2 id="modeling-simulated-quantization-in-forward-and-backward">Modeling simulated quantization in forward and backward</h2>
<p>The basics of Quantization aware training is how to model the quantized effect of inference during training.<br>
Let’s consider a <code>Conv</code> layer like <span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>y</mi><mo>=</mo><mi>ω</mi><mo>∗</mo><mi>x</mi><mo>+</mo><mi>b</mi></mrow><annotation encoding="application/x-tex">y = \omega * x + b</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathit" style="margin-right: 0.03588em;">y</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.46528em; vertical-align: 0em;"></span><span class="mord mathit" style="margin-right: 0.03588em;">ω</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">∗</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.66666em; vertical-align: -0.08333em;"></span><span class="mord mathit">x</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathit">b</span></span></span></span></span></span>. In the origin block, all variables are float.</p>
<div class="mermaid"><svg xmlns="http://www.w3.org/2000/svg" id="mermaid-svg-QBMuMNSsxqRyulji" width="100%" style="max-width: 322.828125px;" viewBox="0 0 322.828125 206"><g transform="translate(-12, -12)"><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath" style="opacity: 1;"><path class="path" d="M53.484375,43L81.578125,43L113.3146690327094,74.7365440327094" marker-end="url(#arrowhead4912)" style="fill:none"></path><defs><marker id="arrowhead4912" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M56.578125,139L81.578125,139L113.3146690327094,107.2634559672906" marker-end="url(#arrowhead4913)" style="fill:none"></path><defs><marker id="arrowhead4913" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M152.578125,91L177.578125,91L209.31466903270942,122.7365440327094" marker-end="url(#arrowhead4914)" style="fill:none"></path><defs><marker id="arrowhead4914" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M144.0625,187L177.578125,187L209.31466903270942,155.26345596729058" marker-end="url(#arrowhead4915)" style="fill:none"></path><defs><marker id="arrowhead4915" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M248.578125,139L273.578125,139L298.578125,139" marker-end="url(#arrowhead4916)" style="fill:none"></path><defs><marker id="arrowhead4916" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node" id="X" transform="translate(38.2890625,43)" style="opacity: 1;"><rect rx="0" ry="0" x="-15.1953125" y="-23" width="30.390625" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-5.1953125,-13)"><foreignObject width="10.390625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">X</div></foreignObject></g></g></g><g class="node" id="*" transform="translate(129.578125,91)" style="opacity: 1;"><circle x="-13.40625" y="-23" r="23"></circle><g class="label" transform="translate(0,0)"><g transform="translate(-3.40625,-13)"><foreignObject width="6.8125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">*</div></foreignObject></g></g></g><g class="node" id="W" transform="translate(38.2890625,139)" style="opacity: 1;"><rect rx="0" ry="0" x="-18.2890625" y="-23" width="36.578125" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-8.2890625,-13)"><foreignObject width="16.578125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">W</div></foreignObject></g></g></g><g class="node" id="+" transform="translate(225.578125,139)" style="opacity: 1;"><circle x="-14.640625" y="-23" r="23"></circle><g class="label" transform="translate(0,0)"><g transform="translate(-4.640625,-13)"><foreignObject width="9.28125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">+</div></foreignObject></g></g></g><g class="node" id="b" transform="translate(129.578125,187)" style="opacity: 1;"><rect rx="0" ry="0" x="-14.484375" y="-23" width="28.96875" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-4.484375,-13)"><foreignObject width="8.96875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">b</div></foreignObject></g></g></g><g class="node" id="y" transform="translate(312.703125,139)" style="opacity: 1;"><rect rx="0" ry="0" x="-14.125" y="-23" width="28.25" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-4.125,-13)"><foreignObject width="8.25" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">y</div></foreignObject></g></g></g></g></g></g></svg></div>
<p>In the simulated quantization in training, the <code>forward</code> diagram is like</p>
<div class="mermaid"><svg xmlns="http://www.w3.org/2000/svg" id="mermaid-svg-1DwoWGDjQkzlehps" width="100%" style="max-width: 580.703125px;" viewBox="0 0 580.703125 254"><g transform="translate(-12, -12)"><g class="output"><g class="clusters"></g><g class="edgePaths"><g class="edgePath" style="opacity: 1;"><path class="path" d="M58.6796875,43L86.84375,43L111.84375,43" marker-end="url(#arrowhead4950)" style="fill:none"></path><defs><marker id="arrowhead4950" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M157.84375,43L195.1875,43L237.53132254199144,76.68208773925369" marker-end="url(#arrowhead4951)" style="fill:none"></path><defs><marker id="arrowhead4951" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M61.84375,139L86.84375,139L111.84375,139" marker-end="url(#arrowhead4952)" style="fill:none"></path><defs><marker id="arrowhead4952" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M157.84375,139L195.1875,139L237.53132254199144,105.31791226074631" marker-end="url(#arrowhead4953)" style="fill:none"></path><defs><marker id="arrowhead4953" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M278.53125,91L303.53125,91L335.2677940327094,122.7365440327094" marker-end="url(#arrowhead4954)" style="fill:none"></path><defs><marker id="arrowhead4954" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M149.328125,235L195.1875,235L232.53125,235" marker-end="url(#arrowhead4955)" style="fill:none"></path><defs><marker id="arrowhead4955" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M278.53125,235L303.53125,235L341.24533730350095,159.57182539299805" marker-end="url(#arrowhead4956)" style="fill:none"></path><defs><marker id="arrowhead4956" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M374.53125,139L408.0234375,139L441.515625,139" marker-end="url(#arrowhead4957)" style="fill:none"></path><defs><marker id="arrowhead4957" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g><g class="edgePath" style="opacity: 1;"><path class="path" d="M499.890625,139L524.890625,139L549.890625,139" marker-end="url(#arrowhead4958)" style="fill:none"></path><defs><marker id="arrowhead4958" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="strokeWidth" markerWidth="8" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowheadPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker></defs></g></g><g class="edgeLabels"><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(195.1875,43)" style="opacity: 1;"><g transform="translate(-9.4140625,-13)" class="label"><foreignObject width="18.828125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel">Xq</span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(195.1875,139)" style="opacity: 1;"><g transform="translate(-12.34375,-13)" class="label"><foreignObject width="24.6875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel">Wq</span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel" transform="translate(408.0234375,139)" style="opacity: 1;"><g transform="translate(-8.4921875,-13)" class="label"><foreignObject width="16.984375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel">yq</span></div></foreignObject></g></g><g class="edgeLabel" transform="" style="opacity: 1;"><g transform="translate(0,0)" class="label"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node" id="Xf" transform="translate(40.921875,43)" style="opacity: 1;"><rect rx="0" ry="0" x="-17.7578125" y="-23" width="35.515625" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-7.7578125,-13)"><foreignObject width="15.515625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Xf</div></foreignObject></g></g></g><g class="node" id="Qx" transform="translate(134.84375,43)" style="opacity: 1;"><circle x="-20.265625" y="-23" r="23"></circle><g class="label" transform="translate(0,0)"><g transform="translate(-10.265625,-13)"><foreignObject width="20.53125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Qx</div></foreignObject></g></g></g><g class="node" id="*" transform="translate(255.53125,91)" style="opacity: 1;"><circle x="-13.40625" y="-23" r="23"></circle><g class="label" transform="translate(0,0)"><g transform="translate(-3.40625,-13)"><foreignObject width="6.8125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">*</div></foreignObject></g></g></g><g class="node" id="Wf" transform="translate(40.921875,139)" style="opacity: 1;"><rect rx="0" ry="0" x="-20.921875" y="-23" width="41.84375" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-10.921875,-13)"><foreignObject width="21.84375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Wf</div></foreignObject></g></g></g><g class="node" id="Qw" transform="translate(134.84375,139)" style="opacity: 1;"><circle x="-22.6953125" y="-23" r="23"></circle><g class="label" transform="translate(0,0)"><g transform="translate(-12.6953125,-13)"><foreignObject width="25.390625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Qw</div></foreignObject></g></g></g><g class="node" id="+" transform="translate(351.53125,139)" style="opacity: 1;"><circle x="-14.640625" y="-23" r="23"></circle><g class="label" transform="translate(0,0)"><g transform="translate(-4.640625,-13)"><foreignObject width="9.28125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">+</div></foreignObject></g></g></g><g class="node" id="b" transform="translate(134.84375,235)" style="opacity: 1;"><rect rx="0" ry="0" x="-14.484375" y="-23" width="28.96875" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-4.484375,-13)"><foreignObject width="8.96875" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">b</div></foreignObject></g></g></g><g class="node" id="Qb" transform="translate(255.53125,235)" style="opacity: 1;"><circle x="-20.8125" y="-23" r="23"></circle><g class="label" transform="translate(0,0)"><g transform="translate(-10.8125,-13)"><foreignObject width="21.625" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Qb</div></foreignObject></g></g></g><g class="node" id="deQy" transform="translate(470.703125,139)" style="opacity: 1;"><circle x="-29.1875" y="-23" r="29.1875"></circle><g class="label" transform="translate(0,0)"><g transform="translate(-19.1875,-13)"><foreignObject width="38.375" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">deQy</div></foreignObject></g></g></g><g class="node" id="Yf" transform="translate(567.296875,139)" style="opacity: 1;"><rect rx="0" ry="0" x="-17.40625" y="-23" width="34.8125" height="46"></rect><g class="label" transform="translate(0,0)"><g transform="translate(-7.40625,-13)"><foreignObject width="14.8125" height="26"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;">Yf</div></foreignObject></g></g></g></g></g></g></svg></div>
