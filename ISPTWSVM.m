% 	Author : Sanchit Jalan & Rahul Choudhary

%--------------Description--------------------
% Function to calculate accuracy, non-zero
% dual variables and training time for
% ISPTWSVM  for linear case.  
%---------------------------------------------


function [accuracy, non_zero_dual_variables, training_time, lambda] = ISPTWSVM(X1_Train, X2_Train, X_Test, Y_Test, c1, c3, epsilon, tau)

% epsilon1 = epsilon2 = epsilon
% tau1 = tau2 = tau
training_time = 0;
non_zero_dual_variables = zeros(2, 1);

m1 = size(X1_Train, 1);
m2 = size(X2_Train, 1);

n = size(X1_Train, 2);

%---------Original ISPTWSVM for 1st class---------
A = X1_Train;
B = X2_Train;
e1 = ones(m1, 1);
e2 = ones(m2, 1);

tic;
q = [A*A' + c3.*eye(m1)  A*B' ; B*A'  B*B'] + ones(m1 + m2, m1 + m2);
Q = [q zeros(m1 + m2, m2); zeros(m2, m1 + m2) zeros(m2, m2)];


f = [zeros(m1, 1) ; -e2*(c3*(epsilon/tau + 1)) ; e2*(c3*(epsilon*(1 + 1/tau)))];

Aeq = [];
Beq = [];

Im2 = eye(m2);

Ain = [zeros(m2, m1) (-1/tau).*Im2 (1 + 1/tau).*Im2; zeros(m2, m1) Im2 -Im2];
Bin = [c1.*e2; zeros(m2, 1)];

lb = [-Inf(m1, 1); -tau*c1.*e2; zeros(m2, 1)];	%Here alpha belongs to [0, c1e2] and lambda belongs to [-tau*c1e2, c1e2] 
ub = [Inf(m1, 1); c1.*e2;  c1.*e2];


[U, fval] = quadprog(Q, f, Ain, Bin, Aeq, Beq, lb, ub);
toc;

training_time = training_time + toc;

mu = U(1: m1, :);
lambda = U(m1 + 1: m1 + m2, :);
alpha = U(m1 + m2 + 1: end, :);


w1 = (-(A'*mu + B'*lambda))./c3;
eta1 = mu;
b1 = -(e1'*mu + e2'*lambda)/c3;

dual_var = U(1: m1 + m2, :);

for i = 1: size(dual_var, 1)
	if abs(dual_var(i)) > 0.0000001
		non_zero_dual_variables(1) = non_zero_dual_variables(1) + 1;
	end
end


%---------Original ISPTWSVM with Sparse Pinball function for 2nd class---------

c2 = c1;
c4 = c3;

tic;
q = [B*B' + c4.*eye(m2)  -B*A' ; -A*B'  A*A'] + ones(m1 + m2, m1 + m2);
Q = [q zeros(m1 + m2, m1); zeros(m1, m1 + m2) zeros(m1, m1)];

f = [zeros(m2, 1); -e1*(c4*(epsilon/tau + 1)); e1*(c4*(epsilon*(1 + 1/tau)))];

Aeq = [e2' -e1' zeros(1, m1)];
Beq = 0;

Im1 = eye(m1);

Ain = [zeros(m1, m2) (-1/tau).*Im1 (1 + 1/tau).*Im1; zeros(m1, m2) Im1 -Im1];
Bin = [c2.*e1; zeros(m1, 1)];

lb = [-Inf(m2, 1); -tau*c2.*e1; zeros(m1, 1)];
ub = [Inf(m2, 1); c2.*e1; c2.*e1];


[U, fval] = quadprog(Q, f, Ain, Bin, Aeq, Beq, lb, ub);
toc;

training_time = training_time + toc;

mu = U(1: m2, :);
lambda = U(m2 + 1: m2 + m1, :);
alpha = U(m1 + m2 + 1: end, :);


w2 = ( A'*lambda - (B'*mu))./c4;
eta2 = mu;
b2 = (e1'*lambda - e2'*mu)/c4;

dual_var = U(1: m1 + m2, :);

for i = 1: size(dual_var, 1)
	if abs(dual_var(i)) > 0.0000001
		non_zero_dual_variables(2) = non_zero_dual_variables(2) + 1;
	end
end



%--------Evaluating accuracy of obtained SVM model---------
id = ones(size(X_Test, 1), 1);

dist1 = abs(X_Test*w1 + b1.*id);
dist2 = abs(X_Test*w2 + b2.*id);

accuracy = 0;

Y_Predicted = zeros(size(X_Test, 1), 1);

for i = 1: size(X_Test, 1)
	if dist1(i) < dist2(i)
		Y_Predicted(i) = 1;				%Class +1 
	else 
		Y_Predicted(i) = -1;			%Class -1
	end
end

accuracy = (sum(Y_Predicted == Y_Test))/(size(Y_Test, 1));

end





























