%% Authors: Rahul Choudhary  & Sanchit Jalan

%-----------------Description-------------------
% This file is used for calculating sparsity 
% (number of non-zero support vectors) in the 
% dual problem of ISPTWSVM (for non-linear case). 
%-----------------------------------------------



%-------------------------------Loading dataset-------------------------------

%----------The snippet below is used when dataset has a test-train split (uncomment when using)--------------
% load('monks3_train.mat');
% X_Train = data(:, 1: end - 1);
% X1_Train = data(data(:, end) == 1, 1: end - 1);
% X2_Train = data(data(:, end) == -1, 1: end - 1);
% Y_Train = [data(data(:, end) == 1, end); data(data(:, end) == -1, end)];

% load('monks3_test.mat');
% X_Test = data(:, 1: end - 1);
% Y_Test = data(:, end);
%------------------------------------------------------------------------------------------------------------


%-------The snippet below is used when dataset does not have a test-train split(uncomment when using)-------- 
load('sonar.mat');

[M N] = size(data);                       			%Size of original dataset, M are the number of samples and N - 1 are the number of features, last column are the labels

percentage = 50;									%Percentage of samples used for training

m = floor(M*(percentage/100));                      %Total training samples

n = N - 1;                                			%Number of features

x1 = data(data(:, end) == 1, 1: end - 1);			%Samples in data belonging to +1 class
x2 = data(data(:, end) == -1, 1: end - 1);			%Samples in data belonging to -1 class

y1 = data(data(:, end) == 1, end);					%Samples' labels in data belonging to +1 class
y2 = data(data(:, end) == -1, end);					%Samples' labels in data belonging to -1 class

M1 = size(x1, 1);	               					%Total training samples of +1 class
M2 = size(x2, 1);	               					%Total training samples of -1 class

m1 = floor((percentage/100)*M1);                    %No. of Training Samples in +1 class
m2 = floor((percentage/100)*M2);                    %No. of Training Samples of -1 class


X1_Train = x1(1:m1, :);								%Training data for +1 class
X2_Train = x2(1:m2, :);								%Training data for -1 class
Y1_Train = y1(1:m1, :);								%Labels of training data for +1 class 
Y2_Train = y2(1:m2, :);								%Labels of training data for -1 class 

X_Train = [X1_Train; X2_Train];
Y_Train = [Y1_Train; Y2_Train];

X1_Test = x1(m1 + 1: end, :);
X2_Test = x2(m2 + 1: end, :);
Y1_Test = y1(m1 + 1: end, :);
Y2_Test = y2(m2 + 1: end, :);

X_Test = [X1_Test; X2_Test];
Y_Test = [Y1_Test; Y2_Test];
%-----------------------------------------------------------------------------------------------------------





%--------Setting ranges for epsilon, tau and values of c1 and c3----------
epsilon = [0; 0.05; 0.1; 0.2; 0.3; 0.5];			%Here epsilon = epsilon1 (for subproblem 1) = epsilon2 (for subproblem 2)
tau = [0.01; 0.1; 0.2; 0.5; 1.0];					%Here tau = tau1 (for subproblem 1) = tau2 (for subproblem 2)
c1 = power(10, -7);									%Here c1 (for subproblem 1) = c2 (for subproblem 2)
c3 = power(10, -7);									%Here c3 (for subproblem 1) = c4 (for subproblem 2)
gamma = power(10, -8);								%gamma in RBF kernel, same in both subproblems
%-------------------------------------------------------------------------





%-------Cross validation for ISPTWSVM (choosing optimal c1, c3 for each (epsilon, tau) combination) for non-linear case-------
maxx = 0;

for k = 1: 13										%Iterate on 13 values of c1
	c1 = c1*10;
	for q = 1: 13									%Iterate on 13 values of c3
		c3 = c3*10;
		gamma = power(10, -8);
		for l = 1: 12								%Iterate on 12 values of gamma
			gamma = gamma*10;
			for i = 1: size(epsilon, 1)
				for j = 1: size(tau, 1)
					[accuracy] = ISPTWSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, c1, c3, gamma, epsilon(i), tau(j));			%Call to function ISPTWSVM_Kernel (non-linear ISPTWSVM)
					if(maxx < accuracy)
						maxx = accuracy;
						finalc1 = c1;
						finalc3 = c3;
						finalgamma = gamma;
					end
				end
			end
		end
	end
end

ans = [];
sparsity = [];
time = [];
for i = 1: size(epsilon, 1)
	temp = [];
	tempSparsity = [];
	tempTime = 0;
	% for j = 1: size(tau, 1)
	[accuracy, non_zero_dual_variables, training_time, lambda] = ISPTWSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, finalc1, finalc3, finalgamma, epsilon(i), 0.5);				%Calculating number of non-zero SVs for tau = 0.5
	temp = [temp accuracy];
		% if(tau(j) == 0.5)
	tempSparsity = non_zero_dual_variables;
	tempTime = training_time;
		% end
	% end
	ans = [ans; temp];
	sparsity = [sparsity; tempSparsity'];
	time = [time; tempTime];
end



ans = 100.*ans;
ans = round(ans, 3);
ans = single(ans);

disp(sparsity);
%-------------------------------------------------------------------------------------------------------


