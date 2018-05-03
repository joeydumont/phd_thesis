vardef cross_bar (expr p, len, ang) =
  ((-len/2,0)--(len/2,0))
  rotated (ang + angle direction length(p)/2 of p)
  shifted point 0 of p shifted (0,1.5mm)
enddef;
vardef rcross_bar (expr p, len, ang) =
  ((-len/2,0)--(len/2,0))
  rotated (ang + angle direction length(p)/2 of p)
  shifted point 0 of p shifted (0,-1.5mm)
enddef;
style_def crossed expr p =
  cdraw (wiggly p);
  ccutdraw cross_bar (p, 3mm, 45);
  ccutdraw cross_bar (p, 3mm, -45);
  cdraw fullcircle scaled 3mm shifted point 0 of p shifted (0,1.5mm);
enddef;
style_def rcrossed expr p =
  cdraw (wiggly p);
  ccutdraw rcross_bar (p, 3mm, 45);
  ccutdraw rcross_bar (p, 3mm, -45);
  cdraw fullcircle scaled 3mm shifted point 0 of p shifted (0,-1.5mm);
enddef;
