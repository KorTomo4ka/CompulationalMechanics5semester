E = 200000000000;
S = 0.0001;
F_val = -1000;
Nodes = zeros(16,2);
N_elements = 29;
Q = 4;
Elements = zeros(N_elements,Q);
for i = 1:9
Nodes(i,1) = 3*(i-1);
end
for i = 1:7
Nodes(9+i,1) = 24 - 3*i;
Nodes(9+i,2) = 6;
end
Coords = zeros(32,1);
for i = 1:16
Coords(2*i-1) = Nodes(i,1);
Coords(2*i) = Nodes(i,2);
end
for i = 1:15
Elements(i,1) = i;
Elements(i,3) = i+1;
end
Elements(16,1) = 1; Elements(16,3) = 16;
Elements(17,1) = 2; Elements(17,3) = 16;
Elements(18,1) = 3; Elements(18,3) = 16;
Elements(19,1) = 3; Elements(19,3) = 15;
Elements(20,1) = 3; Elements(20,3) = 14;
Elements(21,1) = 4; Elements(21,3) = 14;
Elements(22,1) = 5; Elements(22,3) = 14;
Elements(23,1) = 5; Elements(23,3) = 13;
Elements(24,1) = 5; Elements(24,3) = 12;
Elements(25,1) = 6; Elements(25,3) = 12;
Elements(26,1) = 7; Elements(26,3) = 12;
Elements(27,1) = 7; Elements(27,3) = 11;
Elements(28,1) = 7; Elements(28,3) = 10;
Elements(29,1) = 8; Elements(29,3) = 10;
El = 0;
for i = 1:29
El = Elements(i,1);
Elements(i,1) = 2*El - 1;
Elements(i,2) = 2*El;
El = Elements(i,3);
Elements(i,3) = 2*El - 1;
Elements(i,4) = 2*El;
end

T = zeros(2,4);
ke = zeros(2,2);
ke_gl = zeros(4,4);
le = 0;
F = zeros(32,1);
for i = 2:8
F(2*i) = F_val;
end
A = zeros(4,32);
Ki = zeros(32,32);
U = zeros(32,1);
K = zeros(32,32);
for i = 1:29
le = sqrt( (Coords(Elements(i,3)) - Coords(Elements(i,1))).^2 + (Coords(Elements(i,4)) - Coords(Elements(i,2))).^2 );
T(1,1) = (Coords(Elements(i,3)) - Coords(Elements(i,1)))/le;
T(2,3) = T(1,1);
T(1,2) = (Coords(Elements(i,4)) - Coords(Elements(i,2)))/le;
T(2,4) = T(1,2);
ke = (E*S./le)*[1 -1; -1 1];
ke_gl = T'*ke*T;
A = zeros(4,32);
A(1:2,Elements(i,1):Elements(i,2)) = eye(2);
A(3:4,Elements(i,3):Elements(i,4)) = eye(2);
Ki = A'*ke_gl*A;
K = K + Ki;
Ki = zeros(32,32);
end
K(1,:) = 0; K(:,1) = 0; K(1,1) = 1;
K(2,:) = 0; K(:,2) = 0; K(2,2) = 1;
K(17,:) = 0; K(:,17) = 0; K(17,17) = 1;
K(18,:) = 0; K(:,18) = 0; K(18,18) = 1;
Ul = length(U);
ux = zeros(16,1);
uy = zeros(16,1);
U = linsolve(K,F);
j = 1;
for i = 1:32
    if mod(i,2) == 0
        ux(j) = U(i);

    else uy(j) = U(i);
        j = j+1;
    end
    
end
n = 1;
u_vertical = ux(n+1) * 10^9
u_horizontal = uy(n) * 10^9



    
 