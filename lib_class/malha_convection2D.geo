Point(1) = {-3, -3, 0, 1.0};
Point(2) = {3, -3, 0, 1.0};
Point(3) = {3, 3, 0, 1.0};
Point(4) = {-3, 3, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};
Physical Line("neumann1 bottom") = {1};
Physical Line("neumann1 right") = {2};
Physical Line("neumann1 top") = {3};
Physical Line("neumann1 left") = {4};
Physical Surface(11) = {6};
